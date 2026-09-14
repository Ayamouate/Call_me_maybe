import re
from typing import Any
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, PrivateAttr
from .generator import TokenGenerator
from .llm import build_prompt
from .models import FunctionCall, FunctionDefinition


_MATCH_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "can", "do",
    "for", "from", "how", "i", "in", "is", "it", "me", "of",
    "on", "please", "the", "this", "to", "what", "with", "you",
}
_MATCH_ALIASES = {
    "backward": "reverse",
    "backwards": "reverse",
    "flip": "reverse",
    "greeting": "greet",
    "hello": "greet",
    "hi": "greet",
    "plus": "add",
    "replace": "substitute",
    "sqrt": "root",
    "sum": "add",
    "total": "add",
}


class ConstrainedDecoder(BaseModel):
    """Generate function calls using constrained LLM decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model
    _generator: TokenGenerator = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize the constrained token generator."""
        self._generator = TokenGenerator(model=self.model)

    @staticmethod
    def _match_terms(text: str) -> set[str]:
        """Extract meaningful terms used for a conservative match check."""

        words = re.findall(r"[a-z]+", text.lower())
        return {
            _MATCH_ALIASES.get(word, word)
            for word in words
            if word not in _MATCH_STOP_WORDS
        }

    def _possible_functions(
        self,
        prompt: str,
        functions: list[FunctionDefinition],
    ) -> list[FunctionDefinition]:
        """Keep only functions whose definition relates to the prompt."""

        prompt_terms = self._match_terms(prompt)
        return [
            function
            for function in functions
            if prompt_terms.intersection(
                self._match_terms(
                    f"{function.name} {function.description}"
                )
            )
        ]

    def generate_parameters(
        self,
        function_name: str,
        functions: list[FunctionDefinition],
        input_ids: list[int],
        prompt: str,
    ) -> dict[str, object]:
        """Generate parameters required by the selected function."""

        number_choices = re.findall(r"-?(?:\d+(?:\.\d+)?|\.\d+)", prompt)
        number_choices = [
            "-0" + value[1:]
            if value.startswith("-.")
            else "0" + value
            if value.startswith(".")
            else value
            for value in number_choices
        ]

        selected = next(
            (f for f in functions if f.name == function_name),
            None,
        )
        if selected is None:
            raise ValueError("Selected function does not exist.")

        parameters: dict[str, object] = {}
        items = list(selected.parameters.items())

        if not items:
            self._generator.force_text("}", input_ids)
            return parameters

        for index, (name, definition) in enumerate(items):
            self._generator.force_text(f'"{name}":', input_ids)
            last_parameter = index == len(items) - 1
            stop_character = "}" if last_parameter else ","

            if definition.type in ("number", "float", "integer"):
                integer_only = definition.type == "integer"

                if number_choices:
                    number = self._generator.generate_choice(
                        input_ids,
                        number_choices,
                    )
                    number_choices.remove(number)

                else:
                    number = self._generator.generate_number(
                        input_ids,
                        stop_character,
                        integer_only=integer_only,
                    )

                if integer_only:
                    parameters[name] = int(number)
                elif "." in number:
                    parameters[name] = float(number)
                else:
                    integer_value = int(number)
                    if abs(integer_value) <= 2**53:
                        parameters[name] = float(number)
                    else:
                        parameters[name] = integer_value

            elif definition.type == "string":
                self._generator.force_text('"', input_ids)
                if name == "regex":
                    source_string = parameters.get("source_string", "")
                    value = self._generator.generate_regex(
                        input_ids,
                        str(source_string),
                    )
                else:
                    value = self._generator.generate_string(
                        input_ids,
                        stop_symbol_tail=name == "replacement",
                    )
                self._generator.force_text('"', input_ids)
                parameters[name] = value

            elif definition.type == "boolean":
                value = self._generator.generate_choice(
                    input_ids, ["true", "false"])
                parameters[name] = value == "true"

            else:
                raise ValueError(
                    f"Unsupported parameter type: {definition.type}"
                )
            self._generator.force_text(stop_character, input_ids)
        return parameters

    def decode(self, prompt: str,
               functions: list[FunctionDefinition]) -> FunctionCall:
        """Generate one constrained function call."""

        candidate_functions = self._possible_functions(prompt, functions)
        if not candidate_functions:
            return FunctionCall(
                prompt=prompt,
                name="__no_match__",
                parameters={},
            )

        llm_prompt = build_prompt(prompt, candidate_functions)
        input_ids = self._generator.encode(llm_prompt)

        self._generator.force_text('{"name":"', input_ids)

        function_name = self._generator.generate_choice(
            input_ids,
            [function.name for function in candidate_functions]
            + ["__no_match__"],
            suffix='"',
        )

        if function_name == "__no_match__":
            return FunctionCall(
                prompt=prompt,
                name="__no_match__",
                parameters={},
            )
        self._generator.force_text(',"parameters":{', input_ids,)

        parameters = self.generate_parameters(
            function_name,
            candidate_functions,
            input_ids,
            prompt,
        )
        self._generator.force_text("}", input_ids)

        return FunctionCall(
            prompt=prompt,
            name=function_name,
            parameters=parameters,
        )
