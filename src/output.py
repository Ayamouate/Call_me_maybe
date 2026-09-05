import json
from pathlib import Path
from .models import FunctionCall


def write_results(results: list[FunctionCall], output_file: str) -> None:
    """Write generated function calls to a JSON file."""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        result.model_dump()
        for result in results
    ]
    try:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")
    except OSError as exc:
        raise ValueError(
            f"Could not write output file '{output_file}'."
        ) from exc
