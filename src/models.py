from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator


class PromptItem(BaseModel):
    """Represent one natural-language function-calling prompt."""

    model_config = ConfigDict(extra="forbid")
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, value: str) -> str:
        """Reject empty or whitespace-only prompts."""
        if not value.strip():
            raise ValueError("Prompt cannot be an empty string.")
        return value


class ParameterDefinition(BaseModel):
    """Represent the type of one function parameter."""

    model_config = ConfigDict(extra="forbid")
    type: str


class ReturnDefinition(BaseModel):
    """Represent the return type of a function."""

    type: str


class FunctionDefinition(BaseModel):
    """Represent one available function definition."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, value: str) -> str:
        """Reject empty/whitespace-only names and the reserved sentinel."""
        if not value.strip():
            raise ValueError("Function name cannot be an empty string.")
        if value == "__no_match__":
            raise ValueError(
                "'__no_match__' is a reserved name and cannot be used "
                "as a function name."
            )
        return value

    @field_validator("parameters")
    @classmethod
    def parameter_names_must_not_be_empty(
        cls, value: dict[str, ParameterDefinition]
    ) -> dict[str, ParameterDefinition]:
        """Reject empty/whitespace-only parameter names."""
        for key in value:
            if not key.strip():
                raise ValueError(
                    "Parameter name cannot be an empty string."
                )
        return value


class FunctionCall(BaseModel):
    """Represent one generated function call."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
