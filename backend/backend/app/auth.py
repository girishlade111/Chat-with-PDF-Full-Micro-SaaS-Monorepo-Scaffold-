from fastapi import Depends, HTTPException, Header
import jwt, datetime
from .config import settings


class UserIdentity:
def __init__(self, user_id: str):
self.id = user_id


# For demo: accept X-Demo-User header or Bearer token


def create_token(user_id: str) -> str:
payload = {"sub": user_id, "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)}
return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


async def current_user(x_demo_user: str | None = Header(default=None), authorization: str | None = Header(default=None)) -> UserIdentity:
if x_demo_user:
return UserIdentity(x_demo_user)
if authorization and authorization.startswith("Bearer "):
token = authorization.split(" ",1)[1]
try:
payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
return UserIdentity(payload["sub"])
except Exception:
raise HTTPException(401, "Invalid token")
raise HTTPException(401, "Missing auth")
