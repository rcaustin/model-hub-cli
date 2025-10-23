from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class ModelCreate(BaseModel):
    urls: List[HttpUrl]

class ModelInfo(BaseModel):
    id: str
    name: str
    version: Optional[str] = None
    size_bytes: Optional[int] = None
    net_score: Optional[float] = None
