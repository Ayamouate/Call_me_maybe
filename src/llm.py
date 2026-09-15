import json
from .models import FunctionDefinition


def build_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the prompt that will be given to the LLM."""

    functions_data = [function.model_dump() for function in functions]
    return (
        "Choose the function that CLEARLY matches the user request.\n"
        "Only select a function if the request is UNAMBIGUOUS and directly"
        "relates to that function.\n"
        "If the request is unclear, vague, nonsensical, or does not "
        "clearly match any function, you MUST use "
        "'__no_match__' as the function name.\n"

        "Extract the exact required parameter values.\n"
        "Do not explain parameter values or add extra text.\n"
        "Use valid JSON escaping for string values.\n\n"

        "REGEX RULES:\n"
        "- The regex parameter describes WHAT TO MATCH.\n"
        "- The replacement parameter describes WHAT TO INSERT.\n"
        "- The substitution function already replaces ALL regex matches.\n"
        "- The replacement is inserted once for EACH match.\n"

        "Return a JSON object with 'name' and 'parameters'.\n\n"
        f"Available functions:\n"
        f"{json.dumps(functions_data, indent=2)}\n\n"
        f"User request:\n{user_prompt}\n"
    )
