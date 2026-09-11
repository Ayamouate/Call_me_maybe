import json
import re
from llm_sdk import Small_LLM_Model
from .llm import build_prompt
from .models import FunctionCall, FunctionDefinition
from pydantic import BaseModel, ConfigDict, PrivateAttr


class ConstrainedDecoder(BaseModel):
    """Generate function calls using constrained LLM decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model
    _safe_string_tokens: dict[int, str] = PrivateAttr(
        default_factory=dict
    )

    def _encode(self, text: str) -> list[int]:
        """Encode text and return a simple list of token IDs."""
        raw: object = self.model.encode(text).tolist()
        if (
            not isinstance(raw, list)
            or not raw
            or not isinstance(raw[0], list)
        ):
            raise ValueError("Could not encode text.")
        result: list[int] = []
        for token_id in raw[0]:
            if not isinstance(token_id, int):
                raise ValueError("Tokenizer returned an invalid token ID.")
            result.append(token_id)
        return result

    def filter_logits(
        self,
        input_ids: list[int],
        allowed_tokens: set[int],
    ) -> int:
        """Choose the highest-scoring allowed token."""

        if not allowed_tokens:
            raise ValueError("No allowed tokens available.")
        logits = self.model.get_logits_from_input_ids(input_ids)
        return max(
            allowed_tokens,
            key=lambda token_id: logits[token_id],
        )

    def force_text(self, text: str, input_ids: list[int]) -> None:
        """Append fixed text to the current generation."""
        input_ids.extend(self._encode(text))

    def generate_choice(
        self,
        input_ids: list[int],
        choices: list[str],
        suffix: str = "",
    ) -> str:
        """Let the LLM choose one value from a fixed set."""

        candidates = {
            choice: self._encode(choice + suffix)
            for choice in choices
        }
        generated: list[int] = []
        while candidates:
            allowed_tokens: set[int] = set()
            for tokens in candidates.values():
                if len(generated) < len(tokens):
                    allowed_tokens.add(tokens[len(generated)])
            next_token = self.filter_logits(
                input_ids,
                allowed_tokens,
            )

            input_ids.append(next_token)
            generated.append(next_token)

            candidates = {
                choice: tokens
                for choice, tokens in candidates.items()
                if tokens[:len(generated)] == generated
            }
            for choice, tokens in candidates.items():
                if generated == tokens:
                    return choice
        raise ValueError("The LLM could not choose a valid value.")

    def _single_token(self, text: str) -> int:
        """Return the token ID for one allowed character."""
        token_ids = self._encode(text)
        if len(token_ids) != 1:
            raise ValueError(
                f"'{text}' does not map to exactly one token."
            )
        return token_ids[0]

    def generate_number(
        self,
        input_ids: list[int],
        stop_character: str,
        integer_only: bool = False,
    ) -> str:
        """Generate a constrained JSON number."""

        ids = {
            char: self._single_token(char)
            for char in "0123456789-."
        }
        digits = {ids[char] for char in "0123456789"}
        stop_id = self._single_token(stop_character)
        result = ""

        for _ in range(128):
            allowed = set(digits)

            if not result:
                allowed.add(ids["-"])
            elif result in ("0", "-0"):
                allowed = {stop_id}
                if not integer_only:
                    allowed.add(ids["."])
            elif result != "-" and not result.endswith("."):
                allowed.add(stop_id)
                if not integer_only and "." not in result:
                    allowed.add(ids["."])

            token = self.filter_logits(input_ids, allowed)

            if token == stop_id:
                return result

            input_ids.append(token)
            result += next(
                char for char, token_id in ids.items()
                if token_id == token
            )

        raise ValueError("Number generation did not stop.")

    def _is_valid_regex(self, pattern: str) -> bool:
        """Check whether a regex pattern is valid."""

        try:
            re.compile(pattern)
        except re.error:
            return False
        return True

    def _should_stop_regex(
        self,
        current: str,
        next_text: str,
    ) -> bool:
        """Decide whether a completed regex should stop."""

        if not current:
            return False

        if not self._is_valid_regex(current):
            return False

        if current.endswith("]") and not next_text.startswith(
            ("+", "*", "?", "{")
        ):
            return True

        if current.endswith(")") and next_text[:1].isspace():
            return True

        if "|" in next_text:
            return True

        if current.isalnum() and next_text.startswith("."):
            return True

        return False

    def generate_regex(self, input_ids: list[int]) -> str:
        """Generate a short valid regex string."""

        content_tokens = self._get_safe_string_tokens()
        quote_id = self._single_token('"')
        backslash_id = self._single_token("\\")
        escape_tokens = {
            self._single_token(char): char
            for char in '"\\/'
        }
        allowed = set(content_tokens)
        allowed.add(quote_id)
        allowed.add(backslash_id)

        encoded = ""
        last_valid = ""
        for _ in range(32):
            next_token = self.filter_logits(input_ids, allowed)
            if next_token == quote_id:
                value = self._decode_json_string(encoded)
                if self._is_valid_regex(value):
                    return value
                break
            if next_token == backslash_id:
                input_ids.append(next_token)
                encoded += "\\"

                escape_token = self.filter_logits(
                    input_ids,
                    set(escape_tokens),
                )
                input_ids.append(escape_token)
                encoded += escape_tokens[escape_token]
            else:
                text = content_tokens[next_token]
                current = self._decode_json_string(encoded)

                if self._should_stop_regex(current, text):
                    return current

                input_ids.append(next_token)
                encoded += text
            value = self._decode_json_string(encoded)
            if self._is_valid_regex(value):
                last_valid = value
        if last_valid:
            return last_valid
        raise ValueError("Could not generate a valid regex.")

    def _get_safe_string_tokens(self) -> dict[int, str]:
        """Return vocabulary tokens safe inside a JSON string."""

        if self._safe_string_tokens:
            return self._safe_string_tokens

        vocab_path = self.model.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as file:
            raw: object = json.load(file)

        if not isinstance(raw, dict):
            raise ValueError("Invalid vocabulary file.")

        safe_tokens: dict[int, str] = {}
        for value in raw.values():
            if not isinstance(value, int):
                continue
            text = self.model.decode([value])
            if (
                text
                and '"' not in text
                and "\\" not in text
                and "\ufffd" not in text
                and all(ord(character) >= 32 for character in text)
            ):
                safe_tokens[value] = text

        if not safe_tokens:
            raise ValueError("No safe string tokens found.")

        self._safe_string_tokens = safe_tokens
        return safe_tokens

    def _decode_json_string(self, encoded: str) -> str:
        """Decode JSON string content and ensure it is a string."""

        value: object = json.loads(f'"{encoded}"')
        if not isinstance(value, str):
            raise ValueError("Decoded JSON value is not a string.")
        return value

    def generate_string(
        self,
        input_ids: list[int],
        stop_symbol_tail: bool = False,
    ) -> str:
        """Generate a constrained JSON string."""

        content_tokens = self._get_safe_string_tokens()
        quote_id = self._single_token('"')
        backslash_id = self._single_token("\\")
        escape_tokens = {
            self._single_token(char): char
            for char in '"\\/bfnrt'
        }
        allowed = set(content_tokens)
        allowed.add(quote_id)
        allowed.add(backslash_id)
        encoded = ""

        for _ in range(32):
            next_token = self.filter_logits(input_ids, allowed)
            if next_token == quote_id:
                return self._decode_json_string(encoded)

            if next_token != backslash_id:
                text = content_tokens[next_token]
                current = self._decode_json_string(encoded)

                if (
                    stop_symbol_tail
                    and current
                    and all(
                        not char.isalnum() and not char.isspace()
                        for char in current
                    )
                    and all(
                        not char.isalnum() and not char.isspace()
                        for char in text
                    )
                ):
                    return current
                input_ids.append(next_token)
                encoded += text
                continue

            input_ids.append(next_token)
            encoded += "\\"
            escape_token = self.filter_logits(
                input_ids,
                set(escape_tokens),
            )
            input_ids.append(escape_token)
            encoded += escape_tokens[escape_token]

        return self._decode_json_string(encoded)

    def generate_parameters(
        self,
        function_name: str,
        functions: list[FunctionDefinition],
        input_ids: list[int],
        prompt: str,
    ) -> dict[str, object]:
        """Generate parameters required by the selected function."""

        number_choices = re.findall(r"-?\d+(?:\.\d+)?", prompt)
        selected = next(
            (f for f in functions if f.name == function_name),
            None,
        )
        if selected is None:
            raise ValueError("Selected function does not exist.")

        parameters: dict[str, object] = {}
        items = list(selected.parameters.items())
        if not items:
            self.force_text("}", input_ids)
            return parameters

        for index, (name, definition) in enumerate(items):
            self.force_text(f'"{name}":', input_ids)
            last_parameter = index == len(items) - 1
            stop_character = "}" if last_parameter else ","

            if definition.type in ("number", "float", "integer"):
                integer_only = definition.type == "integer"

                if number_choices:
                    number = self.generate_choice(
                        input_ids,
                        number_choices,
                    )
                    number_choices.remove(number)
                else:
                    number = self.generate_number(
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
                self.force_text('"', input_ids)
                if name == "regex":
                    value = self.generate_regex(input_ids)
                else:
                    value = self.generate_string(
                        input_ids,
                        stop_symbol_tail=name == "replacement",
                    )
                self.force_text('"', input_ids)
                parameters[name] = value

            elif definition.type == "boolean":
                value = self.generate_choice(input_ids, ["true", "false"])
                parameters[name] = value == "true"

            else:
                raise ValueError(
                    f"Unsupported parameter type: {definition.type}"
                )
            self.force_text(stop_character, input_ids)
        return parameters

    def decode(self, prompt: str,
               functions: list[FunctionDefinition]) -> FunctionCall:
        """Generate one constrained function call."""

        llm_prompt = build_prompt(prompt, functions)
        input_ids = self._encode(llm_prompt)

        self.force_text('{"name":"', input_ids)
        function_name = self.generate_choice(
            input_ids,
            [function.name for function in functions]
            + ["__no_match__"],
            suffix='"',
        )
        if function_name == "__no_match__":
            return FunctionCall(
                prompt=prompt,
                name="__no_match__",
                parameters={},
            )
        self.force_text(',"parameters":{', input_ids,)
        parameters = self.generate_parameters(
            function_name,
            functions,
            input_ids,
            prompt,
        )
        self.force_text("}", input_ids)

        return FunctionCall(
            prompt=prompt,
            name=function_name,
            parameters=parameters,
        )
