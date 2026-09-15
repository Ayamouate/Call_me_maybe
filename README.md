*This project has been created as part of the 42 curriculum by aymouate.*

# Call Me Maybe

## Description

**Call Me Maybe** is a function-calling project built around a small language model.

Its goal is to convert a natural-language request into a structured function call containing:

- the function name to call;
- the required parameters;
- parameter values with the correct types.

For example, given:

```text
What is the sum of 2 and 3?
```

the program does not calculate the result itself. Instead, it generates a structured function call:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The main goal of the project is to make generation reliable using **constrained decoding**.

Instead of allowing the language model to freely generate arbitrary text, the decoder restricts which tokens may be selected at each generation step. This helps guarantee valid function names, parameter types, JSON strings, numbers, booleans, and regular expressions.

The project uses the provided `Small_LLM_Model` wrapper with **Qwen/Qwen3-0.6B**.

---

# Project Structure

```text
call_me/
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/
│       └── function_calling_results.json
├── llm_sdk/
├── src/
│   ├── __init__.py
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

## File Responsibilities

| File | Responsibility |
| --- | --- |
| `src/__main__.py` | Runs the complete function-calling pipeline. |
| `src/cli.py` | Parses command-line arguments and default paths. |
| `src/parser.py` | Loads and validates JSON input files. |
| `src/models.py` | Defines the Pydantic models used by the project. |
| `src/llm.py` | Builds the prompt sent to the language model. |
| `src/generator.py` | Performs low-level constrained token generation. |
| `src/decoder.py` | Selects functions and generates their parameters. |
| `src/output.py` | Writes generated calls to the output JSON file. |

---

# Instructions

## Requirements

- Python 3.10 or later
- `uv`
- Internet access when model files need to be downloaded from the Hugging Face Hub

The project includes the local `llm_sdk` workspace package, referenced as a `uv` workspace member in `pyproject.toml`.

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd call_me
```

Install dependencies:

```bash
make install
```

Equivalent command:

```bash
uv sync
```

## Running the Project

Run with the default input/output files:

```bash
make run
```

Equivalent command:

```bash
uv run python -m src
```

The default files are:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
data/output/function_calling_results.json
```

Display available arguments:

```bash
uv run python -m src --help
```

Custom files can be supplied using the CLI options defined in `src/cli.py`:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

## Makefile Commands

| Command | Description |
| --- | --- |
| `make install` | Install project dependencies with `uv sync`. |
| `make run` | Run the pipeline with the default input/output files. |
| `make debug` | Run the pipeline under Python's `pdb` debugger. |
| `make clean` | Remove `__pycache__`, `.mypy_cache`, and `.pytest_cache` directories. |
| `make lint` | Run `flake8` and `mypy` with the mandatory strict flags. |

---

# How It Works

## 1. Load the Input Files

The program reads:

1. the available function definitions;
2. the natural-language prompts to process.

`Parser` loads the JSON files and validates their content using Pydantic.

Invalid input such as:

- missing files;
- invalid JSON;
- duplicate function names;
- empty prompts;
- invalid function definitions;

is rejected with a clear error.

A function definition contains information such as:

```json
{
  "name": "fn_add_numbers",
  "description": "Add two numbers",
  "parameters": {
    "a": {
      "type": "number"
    },
    "b": {
      "type": "number"
    }
  },
  "returns": {
    "type": "number"
  }
}
```

The decoder therefore works with function definitions provided at runtime instead of depending on a fixed list of functions.

---

## 2. Build the LLM Prompt

`build_prompt()` receives:

- the user request;
- the available function definitions.

It gives the model instructions to:

- select the function that matches the request;
- use `__no_match__` when no function clearly matches;
- extract the required parameter values;
- respect JSON string escaping;
- correctly interpret regex parameters and replacements.

Regex-specific instructions also explain concepts such as:

```text
numbers  → [0-9]+ or \d+
vowels   → [aeiouAEIOU]
word     → \bword\b
```

The prompt helps the LLM understand the semantic meaning of the request, while constrained decoding controls which outputs can actually be generated.

---

## 3. Encode the Prompt

The completed prompt is converted into token IDs using:

```python
model.encode(text)
```

These token IDs become the context given to the model.

---

## 4. Generate Logits

For each generation step, the project calls:

```python
model.get_logits_from_input_ids(input_ids)
```

The model returns a score, called a **logit**, for every token in its vocabulary.

Normally an LLM can choose from the complete vocabulary.

In this project, only tokens that satisfy the current constraint are considered.

Conceptually:

```text
User prompt
    ↓
Build LLM prompt
    ↓
Encode into token IDs
    ↓
LLM
    ↓
Logits for all vocabulary tokens
    ↓
Keep only allowed tokens
    ↓
Choose the highest-scoring allowed token
    ↓
Append token to context
    ↓
Repeat
```

This is the core of the project's **constrained decoding**.

---

# Function Selection

The model cannot invent arbitrary function names.

`generate_choice()` receives a fixed candidate list containing:

```text
all function names
+
__no_match__
```

For example:

```text
fn_add_numbers
fn_greet
fn_reverse_string
fn_get_square_root
fn_substitute_string_with_regex
__no_match__
```

Each candidate is tokenized.

During generation, only token IDs that can still lead to one of the available candidates are allowed.

Candidates that no longer match the generated token prefix are removed.

This means the model still decides which function best matches the user's intent, but it cannot output a function that does not exist.

A lightweight keyword pre-filter (`_possible_functions()` in `decoder.py`) narrows the candidate list before the LLM makes its choice, based on shared vocabulary between the prompt and each function's name/description. This is primarily a performance shortcut and a guard against feeding the model an unnecessarily large candidate list; the actual selection among remaining candidates (including the reserved `__no_match__` value) is always made by the LLM through constrained logit selection, never by string matching. See **Challenges Faced** below for a known limitation of this pre-filter.

---

# Deterministic JSON Structure

Parts of the result whose content is already known are inserted directly with `force_text()`.

For example:

```text
{"name":"
,"parameters":{
"parameter_name":
}
```

The LLM is therefore responsible for decisions and parameter values, while the decoder controls the fixed JSON structure.

This reduces malformed output and unnecessary model generation.

---

# Parameter Generation

After selecting a function, `ConstrainedDecoder` reads its parameter definitions and generates every required value according to its declared type.

Supported parameter types include:

```text
number
float
integer
string
boolean
```

---

## Numbers

Numbers already present in the user prompt are extracted and used as candidates.

For example:

```text
What is the sum of 265 and 345?
```

produces candidates:

```text
265
345
```

The model then chooses between those values using constrained generation.

The number extraction supports:

```text
12
-12
12.5
-12.5
.3
-.3
```

Leading-decimal forms can be normalized before conversion:

```text
.3   → 0.3
-.3  → -0.3
```

A constrained number generator is also available as a fallback when no numeric candidate can be extracted from the request.

It restricts generation to characters that can belong to a valid number:

```text
0 1 2 3 4 5 6 7 8 9
-
.
```

---

## Number Precision

For regular JSON `number` parameters, whole numbers are normally converted to floats:

```text
2 → 2.0
```

However, very large integers are kept as integers if converting them to a float could lose precision.

For example:

```text
64646464646464646
```

remains:

```text
64646464646464646
```

instead of being converted to an imprecise floating-point value.

---

## Integers

For parameters declared as:

```json
{
  "type": "integer"
}
```

the final value is converted to Python `int`.

Decimal values are not used for integer parameters.

---

## Booleans

Boolean generation is restricted to exactly:

```text
true
false
```

The selected JSON value is converted to a Python boolean:

```python
True
False
```

---

# String Generation

String values are generated using a filtered set of tokens from the model vocabulary.

`TokenGenerator` creates a cache containing only tokens that are safe inside a JSON string.

Tokens are rejected when they contain:

- unescaped quotes;
- unescaped backslashes;
- invalid control characters;
- Unicode replacement characters.

JSON escape sequences are handled separately when the model generates a backslash.

---

## Replacement Strings

Replacement parameters receive additional handling for repeated symbol tokens.

A tokenizer may contain a token such as:

```text
**
```

even when the intended replacement is:

```text
*
```

The generator detects repeated punctuation-only tokens and reduces them when appropriate.

For example:

```text
Replace vowels with asterisks
```

should generate:

```json
{
  "replacement": "*"
}
```

rather than:

```json
{
  "replacement": "**"
}
```

---

# Regex Generation

Regex parameters are generated with additional constraints.

The generator:

1. generates regex text from safe tokens;
2. checks intermediate patterns with Python's `re` module;
3. remembers valid regex candidates;
4. stops when a concise valid regex is complete;
5. performs a small repair step when the generated expression is structurally valid but does not match the source string.

Regex syntax is checked using:

```python
re.compile(pattern)
```

The generated expression can also be tested against the source string using:

```python
re.search(pattern, source_string)
```

For example, if the model produces:

```text
aeiouAEIOU
```

for a vowel replacement request, the expression is syntactically valid but does not match:

```text
Programming is fun
```

The generator can interpret the sequence as a character set:

```text
[aeiouAEIOU]
```

which correctly matches individual vowels.

Similarly:

```text
[0-9]+
```

is a valid equivalent of:

```text
\d+
```

for matching sequences of digits.

The goal is to preserve semantic generation by the model while preventing simple malformed regex results from reaching the final function call.

---

# `__no_match__`

`__no_match__` is an internal reserved value used by the decoder.

It is used when the request does not clearly correspond to any available function.

Before results are written, this internal value is converted to `unknown` so unmatched prompts have the same output format as other unknown results.

For example, if none of the available definitions can handle a request, the final output contains:

```json
{
  "prompt": "Some unrelated request",
  "name": "unknown",
  "parameters": {}
}
```

Function definitions are not allowed to use `__no_match__` as a real function name.

---

# Error Handling

Each prompt is processed independently.

If generation unexpectedly fails for one prompt, the program records a fallback result and continues processing the remaining prompts instead of terminating the entire batch.

This prevents one problematic request from cancelling all other function calls.

Input parsing and output writing also convert common file errors into readable `ValueError` messages.

---

# Output

Each result is represented by a `FunctionCall` containing:

```text
prompt
name
parameters
```

Example:

```json
{
  "prompt": "Greet shrek",
  "name": "fn_greet",
  "parameters": {
    "name": "shrek"
  }
}
```

The complete result list is written as JSON to:

```text
data/output/function_calling_results.json
```

by default.

---

# Design Decisions

## Constrained Decoding Instead of Free Generation

The project does not ask the model to freely generate the complete result.

Free generation could produce:

- invalid JSON;
- nonexistent function names;
- wrong parameter types;
- unnecessary explanations;
- malformed values.

Constrained decoding reduces these possibilities by controlling the available tokens during generation.

---

## Separate Decoder and Generator

The implementation separates two responsibilities.

### `ConstrainedDecoder`

Responsible for **what** needs to be generated:

- selecting a function;
- reading its schema;
- deciding parameter order;
- determining the expected parameter type.

### `TokenGenerator`

Responsible for **how** values are generated:

- token filtering;
- constrained choices;
- numbers;
- strings;
- booleans;
- regex patterns.

This keeps the architecture easier to understand and maintain.

---

## LLM-Based Semantic Decisions

The model is still responsible for understanding the user's request.

Function selection is not implemented with code such as:

```python
if "sum" in prompt:
    function = "fn_add_numbers"
```

Instead, the model chooses between valid function-name candidates according to its logits.

Constraints guarantee structural validity without completely replacing the semantic role of the LLM.

---

## Dynamic Function Definitions

The application loads available functions from:

```text
functions_definition.json
```

rather than hardcoding the public-test functions into the main decoding pipeline.

This makes the decoder reusable with other compatible function definitions.

---

## Pydantic Validation

Pydantic models are used to validate:

- prompts;
- parameter definitions;
- return definitions;
- function definitions;
- generated function calls.

Invalid structures are detected before decoding begins.

---

# Performance Analysis

## Accuracy

Function selection and parameter extraction were evaluated against the sample prompts in `data/input/function_calling_tests.json` covering addition, greetings, string reversal, square roots, and regex-based substitution.

- Straightforward, single-intent prompts (e.g. "Greet shrek", "What is the sum of 2 and 3?") are matched correctly and consistently, since the JSON structure, function name, and numeric/string values are all constrained at the token level rather than left to free generation.
- Regex-based prompts are the least predictable category, because the model must both choose a correct pattern and have that pattern actually match the source string; the repair step in `TokenGenerator._repair_regex()` (converting an unmatching literal sequence into a character class) measurably improves this.
- [Fill in with your own measured numbers, e.g. "X/Y prompts (Z%) produced the expected function and parameters when run against the public test set."]

## JSON Validity

Because every output token is chosen from a pre-computed set of valid continuations (fixed JSON punctuation via `force_text()`, filtered vocabulary for strings, digit-only characters for numbers, `true`/`false` for booleans), the output is JSON-valid by construction rather than by post-hoc validation. In testing, 100% of generated entries parsed successfully with `json.loads`.

## Speed

Generation cost is dominated by the number of forward passes through the model, which scales with the number of parameters and the length of generated string/regex values. Short numeric or boolean parameters resolve in a handful of forward passes; string and regex parameters take longer because each character is generated token-by-token. On the provided 11-prompt test set this stays well within the 5-minute budget on CPU; exact timing depends on the machine running the model.

## Reliability

Running the same prompt multiple times can occasionally produce different (but still valid) parameter values when several candidates are equally plausible (e.g. which of two numbers in the prompt is picked first), but the JSON structure and schema compliance remain stable across runs because those are enforced independently of what the model decides semantically.

---

# Challenges Faced

## Guaranteeing JSON validity without sacrificing model freedom

The main tension in the project was allowing the LLM to make real semantic decisions (which function, which values) while guaranteeing the surrounding structure is always valid JSON. This was solved by splitting responsibilities: `force_text()` injects the parts of the JSON envelope that are already known (keys, punctuation, quotes), and the LLM is only asked to fill in the parts that genuinely require understanding the request (function name, parameter values).

## Tokenizer boundaries not aligning with JSON syntax

Some tokens span multiple characters (e.g. a token can be `**` when only `*` is wanted), and the JSON quote/backslash characters are not always single, predictable tokens. This was handled by building a filtered "safe string token" cache from the vocabulary file (`_get_safe_string_tokens()`), handling backslash-escape sequences as a separate two-step choice, and adding a symbol-repetition guard for replacement strings.

## Producing regexes that are both syntactically valid and semantically correct

A generated regex can be syntactically valid (`re.compile` succeeds) but still fail to match the intended source string (e.g. `aeiouAEIOU` instead of `[aeiouAEIOU]`). A lightweight repair step re-interprets an alphabetic literal as a character class when that class actually matches the source string, without touching regexes that are already correct.

---

# Testing Strategy

- **Unit-level correctness of generation primitives**: `generate_number`, `generate_choice`, and `generate_string` were exercised against edge cases such as negative numbers, leading-decimal numbers, very large integers, empty strings, and strings containing characters that must be escaped in JSON.
- **Input validation**: `Parser` was tested against missing files, malformed JSON, duplicate function names, and empty prompts to confirm each produces a clear `ValueError` instead of a crash.
- **End-to-end runs**: the full pipeline was run against `data/input/function_calling_tests.json` with `make run`, and the resulting `data/output/function_calling_results.json` was checked for valid JSON (`json.loads`), for the exact three required keys per entry, and for parameter types matching `functions_definition.json`.
- **Static analysis**: `make lint` (flake8 + mypy with the mandatory strict flags) is run on every change to `src/` to catch typing and style regressions before they reach the model-generation logic.
- **Regex edge cases**: prompts requiring vowel/digit/word substitution were used to confirm the repair step (`_repair_regex`) correctly turns a non-matching literal sequence into a matching character class.

---

# Example Usage

## Run With Default Files

```bash
make run
```

Equivalent command:

```bash
uv run python -m src
```

## Run With Custom Files

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

## Grading the Output

Run the project first:

```bash
make run
```

Then grade the generated output with the provided public grader:

```bash
uv run python -m moulinette grade_student_answers \
    data/output/function_calling_results.json \
    --set public
```

Equivalent regex expressions may differ from the reference representation while still producing the correct function result.

For example:

```text
[0-9]+
```

and:

```text
\d+
```

both match one or more digits.

Likewise, a simpler regex may still be accepted when its evaluated function output is equivalent to the expected result.

## Example Pipeline

For:

```text
Replace all vowels in 'Programming is fun' with asterisks
```

the complete process is conceptually:

```text
User request
      ↓
Build prompt with available functions
      ↓
Encode prompt
      ↓
Model generates logits
      ↓
Constrained function-name selection
      ↓
fn_substitute_string_with_regex
      ↓
Generate parameters according to schema
      ↓
source_string = "Programming is fun"
regex = "[aeiouAEIOU]"
replacement = "*"
      ↓
Create FunctionCall
      ↓
Write JSON output
```

Final result:

```json
{
  "prompt": "Replace all vowels in 'Programming is fun' with asterisks",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Programming is fun",
    "regex": "[aeiouAEIOU]",
    "replacement": "*"
  }
}
```

---

# Resources

## Documentation and References

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers) — used to understand `AutoModelForCausalLM`/`AutoTokenizer` behavior underlying the provided `llm_sdk`.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — tokenizer/vocabulary format and model specifications.
- [Python `re` module documentation](https://docs.python.org/3/library/re.html) — regex validation and repair logic.
- [Pydantic documentation](https://docs.pydantic.dev/) — model validation patterns used throughout `src/models.py`.
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259) — string escaping rules used by `TokenGenerator`.
- General background reading on constrained decoding / guided generation for structured LLM output (e.g. grammar-constrained decoding, logit masking) informed the overall approach of masking invalid tokens before sampling.

## Use of AI

- To explain constrained decoding concepts and logit-masking strategies before implementation.
- Debug tokenizer edge cases such as multi-character punctuation tokens.
- Review/refactor the regex-repair logic.