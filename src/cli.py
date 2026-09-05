import argparse
from pydantic import BaseModel


DEFAULT_FUNCTIONS = (
    "data/input/functions_definition.json"
)
DEFAULT_INPUT = (
    "data/input/function_calling_tests.json"
)
DEFAULT_OUTPUT = (
    "data/output/function_calling_results.json"
)


class Arguments(BaseModel):
    """Represent validated command-line arguments."""

    functions_definition: str
    input: str
    output: str


def parse_arguments() -> Arguments:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate structured function calls "
            "from natural-language prompts."
        )
    )

    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS,
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the prompt input JSON file.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to the generated output JSON file.",
    )

    namespace = parser.parse_args()

    return Arguments(
        functions_definition=namespace.functions_definition,
        input=namespace.input,
        output=namespace.output,
    )
