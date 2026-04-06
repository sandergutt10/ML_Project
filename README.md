# AST Code Completion Project

Project for Python code completion through AST token generation.

## Requirements

- Python 3.11+ is recommended.
- External dependencies: `torch`, `streamlit`
- All other imports used in the project come from the Python standard library.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Main Files

- `MLCode.py` - model, training loop, checkpoint loading, inference entry points.
- `parse_python.py` - Python `tokenize` + AST conversion pipeline and code completion post-processing.
- `tests.py` - smoke tests that write logs to `test_outputs/`.
- `frontend/app.py` - Streamlit frontend with tabs for custom input and built-in examples.
- `checkpoints_code_lm/` - model checkpoints, including `best.pt`.
- `data/` - local datasets such as `python100k_train.json` and `python50k_eval.json`.
- `docs/` - local project documents.
- `requirements.txt` - minimal runtime dependency list.

## Local Artifacts

- `data/` stores local datasets and is ignored by git.
- `test_outputs/` stores generated test logs and is ignored by git.
- `.venv/`, `.idea/`, and `__pycache__/` are local environment files.

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run tests:

```powershell
python tests.py
```

Run the frontend:

```powershell
streamlit run frontend/app.py
```

Default local layout:

```text
Project/
|- checkpoints_code_lm/
|  |- best.pt
|- data/
|  |- python100k_train.json
|  |- python50k_eval.json
|- frontend/
|  |- app.py
|- docs/
|  |- *.docx
|- test_outputs/
|- MLCode.py
|- parse_python.py
|- tests.py
```

Compile-check the main scripts:

```powershell
python -m py_compile MLCode.py parse_python.py tests.py
```

## Notes

- Inference for incomplete user input goes through `tokenize` first and then into AST processing.
- The completion API is configured to return only the generated continuation, while full reconstructed code is still available in test logs.
- The frontend provides two tabs: one for custom user input and one for the curated examples from `tests.py`.
