from pydantic import BaseModel
from typing import Dict

class InputSchema(BaseModel):
    function_name: str
    query: str
    question: str

class SystemPromptSchema(BaseModel):
    """Schema for Nobel Prize agent system prompts."""
    role: str = "You are an AI assistant specialized in Nobel Prize history and laureates. Your role is to provide accurate information about Nobel Prize winners, their achievements, and the significance of their work."