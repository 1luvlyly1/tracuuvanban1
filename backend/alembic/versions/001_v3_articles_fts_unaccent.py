"""V3: unaccent + pg_trgm + articles.search_vector GIN

Revision ID: v3_001
Revises:
Create Date: 2026-03-20
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "v3_001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        ALTER TABLE articles ADD COLUMN IF NOT EXISTS search_vector tsvector
        """
    )
    op.execute(
        """
        UPDATE articles SET search_vector =
          setweight(to_tsvector('simple', coalesce(title,'')), 'A') ||
          setweight(to_tsvector('simple', coalesce(content,'')), 'B')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_articles_search_vector
        ON articles USING GIN(search_vector)
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION articles_search_vector_update() RETURNS trigger AS $$
        BEGIN
          NEW.search_vector :=
            setweight(to_tsvector('simple', coalesce(NEW.title,'')), 'A') ||
            setweight(to_tsvector('simple', coalesce(NEW.content,'')), 'B');
          RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER articles_search_vector_trigger
        BEFORE INSERT OR UPDATE ON articles
        FOR EACH ROW EXECUTE FUNCTION articles_search_vector_update()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS articles_search_vector_trigger ON articles")
    op.execute("DROP FUNCTION IF EXISTS articles_search_vector_update")
    op.execute("DROP INDEX IF EXISTS idx_articles_search_vector")
    op.execute("ALTER TABLE articles DROP COLUMN IF EXISTS search_vector")
