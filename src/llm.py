import json
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition


def build_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the prompt that will be given to the LLM."""

    functions_data = [function.model_dump() for function in functions]
    return (
        "Choose the function that best matches the user request.\n"
        "Extract the exact required parameter values.\n"
        "Do not explain parameter values or add extra text.\n"
        "Use valid JSON escaping for string values.\n"
        "Return a JSON object with 'name' and 'parameters'.\n\n"
        f"Available functions:\n"
        f"{json.dumps(functions_data, indent=2)}\n\n"
        f"User request:\n{user_prompt}\n"
    )


def tokenizer(model: Small_LLM_Model, text: str) -> list[int]:
    """Convert text into token IDs using the LLM tokenizer."""

    encoded = model.encode(text)
    token_ids = encoded.tolist()
    if not token_ids or not isinstance(token_ids[0], list):
        raise ValueError("Could not tokenize the prompt.")
    return [int(token_id) for token_id in token_ids[0]]


def get_logits(model: Small_LLM_Model, token_ids: list[int]) -> list[float]:
    """Get the model scores for the next possible token."""

    return model.get_logits_from_input_ids(token_ids)
