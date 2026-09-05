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
        result = decoder.decode(prompt_item.prompt, parser.functions)
        results.append(result)
        print(
            f"\nResult:\n\nFonction name: {
                result.name}\nParameters: {
                result.parameters}\n")

    # for prompt_item in parser.prompts:
    #     result = decoder.decode(
    #         prompt_item.prompt,
    #         parser.functions,
    #     )
    #     results.append(result)

    write_results(results, arguments.output)

    print(
        f"Generated {len(results)} function calls "
        f"in '{arguments.output}'."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
