"""Phân loại câu hỏi người dùng bằng LLM (JSON) — bổ sung / thay thế regex routing."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.config import (
    OPENAI_API_KEY,
    QUERY_UTTERANCE_CLASSIFIER_ENABLED,
    QUERY_UTTERANCE_CLASSIFIER_MAX_TOKENS,
    QUERY_UTTERANCE_CLASSIFIER_MODEL,
    QUERY_UTTERANCE_MERGE_MIN_CONFIDENCE,
    QUERY_UTTERANCE_OOS_MIN_CONFIDENCE,
)
from app.services.legal_scope import query_has_strong_legal_scope_signals

_SUBSTANTIVE_CONTENT_PATTERNS: re.Pattern = re.compile(
    r"""
    (các\s+loại|các\s+hạng|các\s+nhóm|các\s+mức|các\s+trường\s+hợp
    |gồm\s+những\s+(gì|loại|hạng|mức|trường\s+hợp)
    |có\s+những\s+(loại|hạng|mức|trường\s+hợp)
    |liệt\s+kê\s+(điều\s+kiện|quy\s+định|mức\s+phạt|tiêu\s+chuẩn|yêu\s+cầu|hành\s+vi)
    |điều\s+kiện\s+(để|mở|thành\s+lập|cấp\s+phép|hoạt\s+động)
    |thủ\s+tục\s+(đăng\s+ký|cấp\s+phép|xin|làm)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
from app.services.llm_client import generate_json_object
from app.services.intent_detector import VALID_INTENTS, _RAG_LEGAL_LOOKUP_INTENTS
from app.services.query_intent import query_requires_multi_document_synthesis

log = logging.getLogger(__name__)

_CLASSIFIER_SYSTEM = """Bạn là bộ phân loại câu hỏi tiếng Việt cho chatbot pháp luật / hành chính.
Trả về ĐÚNG một JSON với các khóa sau (boolean hoặc số), không markdown:

- is_legal_or_admin_query: true nếu câu liên quan luật, nghị định, thông tư, điều khoản, mức phạt, thủ tục hành chính, UBND, xã, tình huống cán bộ, thiết chế văn hóa (nhà văn hóa, thư viện), chính sách ưu tiên đầu tư của Nhà nước, đề xuất giải pháp gắn vận hành cơ sở công. false CHỈ khi rõ ràng là chat chung (thời tiết, giải trí không liên hành chính) hoặc không liên quan pháp luật/hành chính.

- is_checklist_catalog_only: true CHỈ KHI người dùng muốn biết TÊN / SỐ HIỆU các văn bản pháp luật (nghị định, thông tư, luật…) mà KHÔNG cần trích nội dung điều khoản.
  PHẢI false khi:
  • Câu hỏi về NỘI DUNG pháp lý dù có dùng từ "các", "liệt kê", "gồm những gì", "có những loại nào" — ví dụ: "các loại xe ưu tiên", "liệt kê điều kiện mở spa", "mức phạt gồm những mức nào" → đây là tra cứu Điều/Khoản, cần trích dẫn nguyên văn.
  • Câu hỏi về thủ tục hành chính, điều kiện, tiêu chuẩn, mức phạt, quy định cụ thể.
  • Câu có tổng hợp/so sánh/đối chiếu quy định giữa nhiều văn bản.
  Ví dụ is_checklist_catalog_only=true: "có những văn bản nào về giao thông", "liệt kê các nghị định về môi trường".
  Ví dụ is_checklist_catalog_only=false: "các loại xe ưu tiên", "thủ tục đăng ký kinh doanh gồm những bước nào", "liệt kê điều kiện mở spa".

- needs_substantive_legal_answer: true nếu cần trả lời nội dung pháp lý cụ thể (mức phạt, điều kiện, quy định thế nào, thủ tục gồm gì, danh mục loại/hạng/nhóm theo luật, trách nhiệm, căn cứ pháp lý…). Phải true cho mọi câu hỏi về nội dung Điều/Khoản kể cả khi câu dùng từ "liệt kê" hay "các loại".

- references_prior_message_context: true nếu câu tham chiếu hội thoại trước (văn bản trên, cái đó, kế hoạch vừa soạn…). Nếu không có ngữ cảnh hội thoại, luôn false.

- confidence: số 0–1 (độ chắc chắn tổng thể).

Quy tắc bắt buộc: nếu needs_substantive_legal_answer=true thì is_checklist_catalog_only phải false."""


@dataclass
class UtteranceLabels:
    is_legal_or_admin_query: bool = True
    is_checklist_catalog_only: bool = False
    needs_substantive_legal_answer: bool = False
    references_prior_message_context: bool = False
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> UtteranceLabels:
        if not d:
            return cls()
        return cls(
            is_legal_or_admin_query=bool(d.get("is_legal_or_admin_query", True)),
            is_checklist_catalog_only=bool(d.get("is_checklist_catalog_only", False)),
            needs_substantive_legal_answer=bool(d.get("needs_substantive_legal_answer", False)),
            references_prior_message_context=bool(d.get("references_prior_message_context", False)),
            confidence=float(d.get("confidence") or 0.0),
        )


async def classify_user_utterance(
    query: str,
    *,
    has_conversation: bool = False,
) -> Optional[UtteranceLabels]:
    """Một lần gọi LLM JSON. Trả None nếu tắt cấu hình / không API key / lỗi."""
    if not QUERY_UTTERANCE_CLASSIFIER_ENABLED or not (query or "").strip():
        return None
    if not OPENAI_API_KEY:
        return None

    ctx_note = (
        "Người dùng đang trong một cuộc hội thoại (có conversation_id)."
        if has_conversation
        else "Không có ngữ cảnh hội thoại trước; references_prior_message_context nên false trừ khi câu rõ ràng tham chiếu 'trên/đó/vừa rồi'."
    )
    user = f"{ctx_note}\n\nCâu hỏi:\n\"\"\"{(query or '').strip()[:4000]}\"\"\""

    try:
        data = await generate_json_object(
            user,
            system=_CLASSIFIER_SYSTEM,
            model=QUERY_UTTERANCE_CLASSIFIER_MODEL,
            max_tokens=QUERY_UTTERANCE_CLASSIFIER_MAX_TOKENS,
            temperature=0.0,
        )
        if not data:
            return None
        labels = UtteranceLabels.from_dict(data)
        if _SUBSTANTIVE_CONTENT_PATTERNS.search((query or "").strip()):
            if labels.is_checklist_catalog_only:
                log.info(
                    "utterance_classifier post-guard: '%s' matches substantive pattern "
                    "— overriding checklist=True → substantive=True",
                    (query or "")[:80],
                )
            labels.is_checklist_catalog_only = False
            labels.needs_substantive_legal_answer = True
        if query_requires_multi_document_synthesis((query or "").strip()):
            labels.is_checklist_catalog_only = False
            labels.needs_substantive_legal_answer = True
        if labels.needs_substantive_legal_answer:
            labels.is_checklist_catalog_only = False
        log.info(
            "utterance_classifier: legal=%s checklist=%s substantive=%s ctx_ref=%s conf=%.2f",
            labels.is_legal_or_admin_query,
            labels.is_checklist_catalog_only,
            labels.needs_substantive_legal_answer,
            labels.references_prior_message_context,
            labels.confidence,
        )
        return labels
    except Exception as exc:
        log.warning("classify_user_utterance failed: %s", exc)
        return None


def merge_utterance_labels_into_analysis(
    analysis: Dict[str, Any],
    labels: UtteranceLabels,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Sao chép analysis và điều chỉnh intent + rag_flags theo nhãn LLM."""
    out = dict(analysis)
    rf = dict(out.get("rag_flags") or {})
    intent = out.get("intent", "legal_lookup")
    q = (query or "").strip()
    multi_syn = bool(q and query_requires_multi_document_synthesis(q))

    merge_min = float(QUERY_UTTERANCE_MERGE_MIN_CONFIDENCE)

    if labels.needs_substantive_legal_answer:
        if intent == "checklist_documents":
            out["intent"] = "legal_lookup"
        rf["needs_expansion"] = True
        if labels.confidence >= merge_min:
            rf["use_multi_article"] = rf.get("use_multi_article", True)

    if (
        labels.is_checklist_catalog_only
        and not labels.needs_substantive_legal_answer
        and labels.confidence >= merge_min
        and not multi_syn
    ):
        out["intent"] = "checklist_documents"

    out["rag_flags"] = rf
    out["utterance_labels"] = {
        "is_legal_or_admin_query": labels.is_legal_or_admin_query,
        "is_checklist_catalog_only": labels.is_checklist_catalog_only,
        "needs_substantive_legal_answer": labels.needs_substantive_legal_answer,
        "references_prior_message_context": labels.references_prior_message_context,
        "confidence": labels.confidence,
    }
    if multi_syn:
        if out.get("intent") == "checklist_documents":
            out["intent"] = "legal_lookup"
        rf2 = dict(out.get("rag_flags") or {})
        rf2["needs_expansion"] = True
        out["rag_flags"] = rf2

    det_f = out.get("detector_intent", "")
    if det_f in VALID_INTENTS:
        rf3 = dict(out.get("rag_flags") or {})
        lk3 = det_f in _RAG_LEGAL_LOOKUP_INTENTS
        rf3["is_legal_lookup"] = lk3
        rf3["use_multi_article"] = not lk3
        out["rag_flags"] = rf3

    oos_min = float(QUERY_UTTERANCE_OOS_MIN_CONFIDENCE)
    if (
        not labels.is_legal_or_admin_query
        and labels.confidence >= oos_min
        and not multi_syn
        and not query_has_strong_legal_scope_signals(q)
    ):
        out["intent"] = "out_of_scope"
        out["rag_flags"] = {
            "is_legal_lookup": False,
            "use_multi_article": False,
            "needs_expansion": False,
        }
    return out
