from sqlalchemy import text
from sqlalchemy.orm import Session
from .embeddings import embed_texts


async def vector_search(db: Session, document_id: str, query: str, k: int = 20):
vec = (await embed_texts([query]))[0]
# SQLAlchemy raw SQL for pgvector cosine distance
q = text("""
SELECT id, chunk_text, page_start, page_end, para_index,
1 - (embeddings.vec <=> :qvec) AS score
FROM embeddings
WHERE document_id = :doc
ORDER BY embeddings.vec <=> :qvec ASC
LIMIT :k
""")
rows = db.execute(q, {"qvec": vec, "
