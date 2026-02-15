from pydantic import BaseModel, Field
from typing import Optional, Literal, List

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
    model: Optional[str] = None
    finish_reason: Optional[str] = None

# Schema for chat-based generation (TBD, for future multi-turn support)
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")

# Request schema for chat-based generation
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage] # List of messages for multi-turn conversation
    max_new_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

# The Chunk format for Streaming (Delta)
class ChatStreamChoice(BaseModel):
    index: int
    delta: dict # {"role": "assistant", "content": "Hello"}
    finish_reason: Optional[str] = None

# The response format
class ChatStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatStreamChoice]
