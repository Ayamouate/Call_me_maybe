*This project has been created as part of the 42 curriculum by aymouate.*

# Call Me Maybe

## Description

**Call Me Maybe** is a function-calling project built around a small language model. Its goal is to translate a natural-language request into a structured function call containing:

- the function name to use;
- the required parameters;
- parameter values with the correct types.

For example, given a prompt such as:

```text
What is the sum of 2 and 3?
```

the program does not calculate the answer itself. Instead, it produces a structured call similar to:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The main objective of the project is to make this generation reliable by using **constrained decoding**. Rather than trusting the model to spontaneously produce correct JSON, the decoder restricts which tokens are allowed at each generation step so that the produced function call follows the expected structure and parameter types.

The project uses the provided `Small_LLM_Model` wrapper with **Qwen/Qwen3-0.6B** as the default model.

---

## Project Structure

```text
call_me/
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── llm_sdk/
├── src/
│   ├── __main__.py
│   ├── cli.py
│   ├── decoder.py
│   ├── generator.py
│   ├── llm.py
│   ├── models.py
│   ├── output.py
│   └── parser.py
├── .gitignore
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

### Main responsibilities

| File | Responsibility |
| --- | --- |
| `src/__main__.py` | Runs the complete pipeline and handles per-prompt failures. |
| `src/cli.py` | Parses command-line options and provides default paths. |
| `src/parser.py` | Loads and validates JSON input files. |
| `src/models.py` | Defines Pydantic models for prompts, function definitions, parameters, and results. |
| `src/llm.py` | Builds the LLM prompt from the user request and available function definitions. |
| `src/generator.py` | Implements low-level constrained token generation. |
| `src/decoder.py` | Coordinates function selection and parameter generation. |
| `src/output.py` | Creates the output directory and writes valid JSON results. |

---

# Algorithm Explanation

## 1. Load and validate the inputs

The program receives two JSON files:

1. a list of available function definitions;
2. a list of natural-language prompts.

`Parser` loads both files and validates their structure with Pydantic. Invalid JSON, missing files, duplicate function names, empty prompts, and invalid definitions are converted into clear errors instead of causing uncontrolled crashes.

Each function definition contains:

```text
name
Description
parameters
return type
```

The decoder therefore does not depend on a fixed list of callable functions: the available functions are read at runtime.

## 2. Build the model prompt

`build_prompt()` provides the model with:

- the available function definitions;
- instructions to choose the best matching function;
- instructions to extract the required argument values;
- the original user request;
- the `__no_match__` option when none of the available functions can satisfy the request.

This prompt gives the model the semantic context it needs, while constrained decoding controls the actual output generation.

## 3. Encode the prompt

The prompt is passed to the SDK tokenizer:

```python
model.encode(text)
```

The resulting token IDs become the context used for the first generation step.

## 4. Read logits and restrict token choices

For each generated token, `TokenGenerator.filter_logits()` calls:

```python
model.get_logits_from_input_ids(input_ids)
```

The model returns a score for every token in the vocabulary. Instead of selecting from the whole vocabulary, the decoder creates a set of **allowed token IDs** and chooses only the highest-scoring token inside that set.

Conceptually:

```text
Prompt
  ↓
Token IDs
  ↓
LLM
  ↓
Logits for every possible next token
  ↓
Keep only tokens allowed by the current constraint
  ↓
Select the highest-scoring valid token
  ↓
Append it to the context
  ↓
Repeat
```

This is the core constrained-decoding mechanism used by the project.

## 5. Constrain function selection

The function name is not chosen with keyword-based `if` statements. The decoder gives the LLM a restricted candidate set containing:

```text
all function names from functions_definition.json
+
__no_match__
```

`generate_choice()` encodes every candidate and progressively removes candidates whose token prefixes no longer match the model's selected tokens.

At every step, only token IDs that can still lead to one of the valid candidates are allowed. The model therefore still decides which function is the best semantic match, but it cannot invent a function name that is outside the permitted set.

## 6. Force the JSON structure

Structural fragments such as:

```text
{"name":"
,"parameters":{
"parameter_name":
}
```

are appended directly through `force_text()`.

The model is used for decisions and values, while deterministic JSON syntax is inserted by the decoder. This reduces unnecessary generation and prevents the model from corrupting fixed structural parts of the result.

## 7. Generate parameters according to their schema

After selecting a function, `ConstrainedDecoder.generate_parameters()` reads the selected function's parameter definitions and generates every required value according to its declared type.

### Numbers

Numeric values present in the original request are extracted as candidate values. When such candidates exist, the LLM chooses between them using constrained choice generation instead of freely producing an unlimited sequence of digits.

A generic constrained number generator is also kept as a fallback. It only allows characters that can belong to a valid JSON number:

```text
0-9
-
.
```

and the appropriate delimiter when the number may legally stop.

For JSON `number` parameters, ordinary whole values are represented as floats when safely representable, which preserves expected results such as:

```text
2 → 2.0
```

Very large whole numbers are kept as Python integers when conversion to a float would risk precision loss, for example:

```text
64646464646464646 → 64646464646464646
```

### Integers

When the declared type is `integer`, decimal points are not allowed and the generated value is converted to `int`.

### Booleans

Boolean parameters are restricted to exactly:

```text
true
false
```

The selected value is then converted to a Python boolean.

### Strings

String generation uses the model vocabulary to build a cache of tokens that are safe inside a JSON string. Tokens containing invalid control characters, an unescaped quote, an unescaped backslash, or the replacement character are excluded.

The decoder also handles valid JSON escape sequences when a backslash is generated.

### Regex strings

Regex parameters use constrained string generation with additional validation through Python's `re` module. Intermediate patterns are checked with `re.compile()`, and the generator keeps track of valid candidates so that it can return a concise valid regular expression.

## 8. Produce the final result

Each successful generation is stored as a `FunctionCall` object containing exactly:

```text
prompt
name
parameters
```

The complete list is serialized with `json.dump()` to the configured output file.

---

# Design Decisions

## Separate decoder and token generator

The project separates high-level schema logic from low-level token generation:

- `ConstrainedDecoder` decides **what** must be generated: function name, parameter order, and expected parameter type.
- `TokenGenerator` decides **how** valid values are generated from model logits under constraints.

This keeps the decoder easier to understand and makes the constrained-generation utilities reusable.

## Use the LLM for semantic selection

Function selection remains an LLM decision. The implementation does not choose functions by checking for hardcoded keywords in the user prompt. Constraints only limit the model to valid function names.

## Force deterministic structure

Known JSON syntax is inserted directly instead of spending model-generation steps on characters whose value is already known. This improves reliability and reduces the number of opportunities for malformed output.

## Dynamic function definitions

Functions and their parameter schemas are loaded from `functions_definition.json`. The main pipeline therefore works from supplied definitions instead of embedding the provided demonstration functions directly into the main application flow.

## Explicit no-match behavior

`__no_match__` is reserved as an internal sentinel. It lets the LLM express that none of the supplied functions can satisfy a prompt instead of forcing an unrelated function call.

## Per-prompt failure isolation

If an unexpected error occurs while generating one prompt, the main loop records that item as:

```json
{
  "name": "unknown",
  "parameters": {}
}
```

and continues processing the remaining prompts. This prevents a single edge case from cancelling an entire batch.

## Pydantic validation

Pydantic is used for project data models so malformed prompt objects, function definitions, duplicated/reserved names, and invalid field shapes are detected before decoding begins.

---

# Instructions

## Requirements

- Python **3.10 or later**
- `uv`
- Internet access the first time the model weights are downloaded

The project uses the local `llm_sdk` workspace package included in the repository.

## Installation

Clone the repository and enter the project directory:

```bash
git clone <your-repository-url>
cd call_me
```

Install and synchronize dependencies:

```bash
make install
```

Equivalent command:

```bash
uv sync
```

## Run with the default files

```bash
make run
```

Equivalent command:

```bash
uv run python -m src
```

By default, the program reads:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
```

and writes:

```text
data/output/function_calling_results.json
```

The output directory is created automatically and is ignored by Git.

## Run with custom files

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/results.json
```

## Lint and type checking

```bash
make lint
```

This runs `flake8` and `mypy` on the source code with the project's configured checks.

## Clean caches

```bash
make clean
```

This removes Python, mypy, and pytest cache directories.

---

# Input Format

## Function definitions

Example:

```json
[
  {
    "name": "add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {
      "type": "number"
    }
  }
]
```

## Prompt file

Example:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Greet shrek"
  }
]
```

---

# Output Format

The generated output is a JSON array. Every result contains:

- `prompt`: the original request;
- `name`: the selected function name;
- `parameters`: the generated argument object.

Example:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

---

# Example Usage

Running:

```bash
make run
```

produces console progress similar to:

```text
Processing 1/11: What is the sum of 2 and 3?

Result:

Function name: add_numbers
Parameters: {'a': 2.0, 'b': 3.0}
```

The complete machine-readable result is written to:

```text
data/output/function_calling_results.json
```

Another example using explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

---

# Performance Analysis

On the current development environment and the provided 11-prompt test set:

- **Accuracy:** 11/11 expected function calls and arguments are produced correctly.
- **Observed runtime:** approximately **4 minutes** for the full test set after model startup/download requirements are satisfied.
- **JSON reliability:** result serialization is handled by Python's `json` module, while constrained decoding prevents the model from freely generating the fixed JSON structure.
- **Failure isolation:** an error on one prompt does not terminate processing of the remaining prompts.

These measurements describe the current local test environment and supplied test set; performance can vary depending on hardware, model cache state, and the complexity of alternative function definitions or prompts.

The largest runtime cost comes from repeated LLM inference during token-by-token constrained generation. The project favors correctness and controlled output over unconstrained generation speed.

---

# Challenges Faced

## Reliable JSON from a small LLM

A small language model can understand the task but may produce extra prose, malformed JSON, invalid function names, or values with the wrong type when left unconstrained.

**Solution:** fixed JSON fragments are forced directly, while generated parts are limited to schema-compatible tokens and candidates.

## Choosing a valid function without hardcoding the answer

Function names must remain an LLM decision, but the model must not invent names.

**Solution:** `generate_choice()` keeps only token prefixes belonging to the available function names plus the reserved no-match sentinel.

## Numeric generation that does not terminate

Allowing every digit after every generated digit can make the model prefer another number token indefinitely rather than choosing the delimiter.

**Solution:** numeric values found in the user request are used as constrained candidates when available, while a bounded generic number generator remains as a fallback.

## Large-number precision

Converting every JSON `number` directly to `float` can change very large integer values or display them in scientific notation.

**Solution:** safely representable whole numbers remain compatible with float output such as `2.0`, while very large whole values are kept as integers to preserve their exact value.

## Safe JSON strings

Normal vocabulary tokens can contain characters that would break a JSON string.

**Solution:** the generator builds a safe-token vocabulary, excludes unsafe token text, and explicitly handles JSON escape sequences.

## Regex generation

A partially generated regular expression may be syntactically invalid or may continue into unnecessary repeated alternatives.

**Solution:** patterns are checked incrementally with `re.compile()`, a last valid pattern is retained, and stopping rules are used to keep generated regex values concise.

## Keeping the decoder maintainable

The original decoding logic became large because function selection, token filtering, number generation, string escaping, and regex generation were all related but distinct responsibilities.

**Solution:** the implementation was split into `ConstrainedDecoder` for high-level schema orchestration and `TokenGenerator` for low-level constrained generation.

---

# Testing Strategy

The implementation is validated in several layers.

## Provided functional test set

The current input file contains **11 prompts** covering:

- addition;
- greetings;
- string reversal;
- square roots;
- regex-based substitutions.

The current implementation produces the expected function selection and parameters for **11/11** of these prompts.

## Input-validation tests

The parser and Pydantic models are designed to reject or report:

- missing files;
- invalid JSON;
- permission/read errors;
- non-array top-level input;
- malformed function definitions;
- duplicate function names;
- the reserved `__no_match__` function name;
- empty prompts;
- empty parameter names.

## Generation edge cases

During development, the decoder was tested against cases involving:

- large numeric values;
- floats and integers;
- strings requiring JSON-safe generation;
- regex patterns;
- function definitions with different parameter types;
- prompts that do not match an available function;
- per-prompt generation failures.

## Static checks

Run:

```bash
make lint
```

to apply `flake8` and `mypy` checks to the source code.

## Manual output validation

After a run, `data/output/function_calling_results.json` should be checked for:

1. valid JSON syntax;
2. one result object per input prompt;
3. correct `prompt`, `name`, and `parameters` keys;
4. correct argument names;
5. correct argument types;
6. correct function selection and extracted values.

---

# Error Handling

The program is designed to report clear failures instead of crashing without context.

Examples include:

- missing or unreadable input files;
- invalid JSON;
- invalid Pydantic input models;
- missing functions or prompts;
- unsupported parameter types;
- invalid tokenizer results;
- empty constrained-token sets;
- number generation that fails to terminate;
- regex generation that cannot produce a valid pattern;
- output-file write errors.

At the top level, fatal setup errors are printed as a clear `Error:` message. During batch decoding, an individual generation failure is isolated so the remaining prompts can still be processed.

---

# Resources

The following resources were useful for understanding the concepts used in this project:

- Python documentation — `json`: https://docs.python.org/3/library/json.html
- Python documentation — `argparse`: https://docs.python.org/3/library/argparse.html
- Python documentation — `re`: https://docs.python.org/3/library/re.html
- Pydantic documentation: https://docs.pydantic.dev/
- Qwen3-0.6B model card: https://huggingface.co/Qwen/Qwen3-0.6B
- Hugging Face LLM course — tokenizers: https://huggingface.co/learn/llm-course/
- JSON specification: https://www.json.org/json-en.html

The project subject and the provided `llm_sdk` wrapper were also primary references for the required generation pipeline and SDK interface.

## Use of AI

AI tools were used as a learning and development aid during the project. In particular, AI was used to:

- clarify how tokenization, input IDs, logits, and next-token selection work;
- explain constrained decoding and function-calling concepts;
- help reason about edge cases in number, string, and regex generation;
- debug errors observed during development;
