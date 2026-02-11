from pydantic import BaseModel, Field
from typing import Optional

class GenerateRequest(BaseModel):
    model: str = Field(..., description="The model to use for generation", example="gemma-3-1b")
    prompt: str = Field(..., description="The input prompt for text generation", example="Once upon a time")
    max_new_tokens: int = Field(256, description="The maximum number of new tokens")
    stream: bool = Field(False, description="Whether to stream the output tokens")

class GenerateResponse(BaseModel):
    model: str
    text: str
    usage: dict

# Used for stream=True (Each chunk follows this format)
class GenerateStreamChunk(BaseModel):
    text: str
    finish_reason: Optional[str] = None