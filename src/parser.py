import json
from typing import List
from .models import PromptItem, FunctionDefinition
from pydantic import ValidationError


class Parser:
    """Load and validate project input JSON files."""

    def __init__(self) -> None:
        self.functions: List[FunctionDefinition] = []
        self.prompts: List[PromptItem] = []

    def _load_json(self, file: str) -> object:
        """Load Json data from a file"""

        try:
            with open(file, "r") as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise ValueError(
                f"File '{file} not found!"
            ) from exc
        except PermissionError as exc:
            raise ValueError(
                f"Permission denied for '{file}'!"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"File '{file}' contains invalid JSON!"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Could not read file '{file}'!"
            ) from exc

    def load_functions(self, file: str) -> None:
        """Load and validate function definitions."""

        data = self._load_json(file)
        if not isinstance(data, list):
            raise ValueError(
                "Functions definition file must contain a JSON array."
            )
        try:
            self.functions = [
                FunctionDefinition.model_validate(item)
                for item in data
            ]
        except ValidationError as exc:
            raise ValueError(
                f"Invalid function definition: {exc}"
            ) from exc

    def load_prompts(self, file: str) -> None:
        """Load and validate function-calling prompts."""

        data = self._load_json(file)
        if not isinstance(data, list):
            raise ValueError(
                "Prompt file must contain a JSON array."
            )
        try:
            self.prompts = [
                PromptItem.model_validate(item)
                for item in data
            ]
        except ValidationError as exc:
            raise ValueError(
                f"Invalid prompt definition: {exc}"
            ) from exc
