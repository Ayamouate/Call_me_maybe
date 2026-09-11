from .parser import Parser
from llm_sdk import Small_LLM_Model
from .models import FunctionCall
from .decoder import ConstrainedDecoder
from .output import write_results
from .cli import parse_arguments
import sys


def main() -> None:
    """Run the function-calling pipeline."""

    arguments = parse_arguments()
    parser = Parser()
    parser.load_functions(arguments.functions_definition)
    parser.load_prompts(arguments.input)

    if not parser.functions:
        raise ValueError(
            "No function definitions were provided."
        )

    if not parser.prompts:
        raise ValueError(
            "No prompts were provided."
        )

    model = Small_LLM_Model()
    decoder = ConstrainedDecoder(
        model=model,
    )

    results: list[FunctionCall] = []

    for index, prompt_item in enumerate(parser.prompts):
        print(
            f"Processing {index + 1}/{len(parser.prompts)}: "
            f"{prompt_item.prompt}"
        )
        try:
            result = decoder.decode(prompt_item.prompt, parser.functions)
            if result.name == "__no_match__":
                print("  -> no function matches this prompt")
        except Exception as exc:
            print(f"  -> failed, marking as unknown: {exc}")
            result = FunctionCall(
                prompt=prompt_item.prompt,
                name="unknown",
                parameters={},
            )
        results.append(result)
        print(
            "\nResult:\n\n"
            f"Function name: {result.name}\n"
            f"Parameters: {result.parameters}\n"
        )

    write_results(results, arguments.output)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
