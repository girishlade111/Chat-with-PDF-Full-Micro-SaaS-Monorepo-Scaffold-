from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
document_id: str
status: str


class AskRequest(BaseModel):
document_id: str
query: str
stream: Optional[bool] = True


class MessageOut(BaseModel):
role: str
content: str
citations: List[dict] = []
