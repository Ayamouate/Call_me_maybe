from typing import Any
from pydantic import BaseModel, ConfigDict


class PromptItem(BaseModel):
    """Represent one natural-language function-calling prompt."""

    prompt: str


class ParameterDefinition(BaseModel):
    """Represent the type of one function parameter."""

    type: str


class ReturnDefinition(BaseModel):
    """Represent the return type of a function."""

    type: str


class FunctionDefinition(BaseModel):
    """Represent one available function definition."""

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class FunctionCall(BaseModel):
    """Represent one generated function call."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
