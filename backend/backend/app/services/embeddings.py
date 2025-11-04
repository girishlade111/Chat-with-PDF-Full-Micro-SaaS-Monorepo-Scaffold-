from openai import OpenAI
from .config import settings
client = OpenAI(api_key=settings.OPENAI_API_KEY)


async def embed_texts(texts: list[str]) -> list[list[float]]:
# batching is recommended; simplified here
resp = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
return [d.embedding for d in resp.data]
