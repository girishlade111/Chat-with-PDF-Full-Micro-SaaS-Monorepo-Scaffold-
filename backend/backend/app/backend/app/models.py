from sqlalchemy import Column, Integer, Text, JSON, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import Base


class User(Base):
__tablename__ = "users"
id = Column(UUID(as_uuid=True), primary_key=True)
email = Column(Text, unique=True, nullable=False)
name = Column(Text)
password_hash = Column(Text)
created_at = Column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
__tablename__ = "documents"
id = Column(UUID(as_uuid=True), primary_key=True)
user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
filename = Column(Text, nullable=False)
file_url = Column(Text, nullable=False)
page_count = Column(Integer)
status = Column(Text, default="processing")
created_at = Column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
__tablename__ = "messages"
id = Column(Integer, primary_key=True, autoincrement=True)
document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"))
user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
role = Column(Text)
content = Column(Text)
citations = Column(JSON)
latency_ms = Column(Integer)
created_at = Column(DateTime(timezone=True), server_default=func.now())
