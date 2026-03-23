from pydantic import BaseModel
from typing import List, Dict, Any, Optional
class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]] 
    stream: bool = False
    modelConfig: Dict[str, Any] = {}

class TokenCountRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]

class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    text: str