from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path

import MLCode
import parse_python

importlib.reload(MLCode)
importlib.reload(parse_python)

from parse_python import continue_real_code_safe


SAMPLE_GROUPS = {
    "Simple Inline Cases": [
        ("return_simple", "def add(a, b):\n    return "),
        ("return_branch", "def fact(n):\n    if n <= 1:\n        return 1\n    return "),
        ("return_bool", "def is_even(x):\n    if x % 2 == 0:\n        return "),
        ("call_print", "def greet(name):\n    message = f'Hello, {name}'\n    print("),
        ("dict_get", "data = {'a': 1, 'b': 2}\nvalue = data.get("),
        ("attribute_access", "import math\n\ndef area(r):\n    return math."),
        ("string_split", "text = 'hello world'\nparts = text.split("),
    ],
    "Indented Blocks": [
        ("with_return", "def read_text(path):\n    with open(path, 'r', encoding='utf8') as f:\n        return "),
        ("for_body", "numbers = [1, 2, 3]\nfor x in numbers:\n    "),
        ("list_comp_filter", "items = [1, 2, 3]\nresult = [x * 2 for x in items if "),
        ("append_call", "def square_all(xs):\n    out = []\n    for x in xs:\n        out.append("),
        ("except_clause", "def safe_div(a, b):\n    try:\n        return a / b\n    except "),
        ("method_attr", "class User:\n    def __init__(self, name):\n        self.name = name\n        self."),
        ("binary_update", "class Counter:\n    def inc(self):\n        self.value = self.value + "),
    ],
    "Expressions And Literals": [
        ("dict_literal", "def make_user(name, age):\n    return {'name': name, 'age': "),
        ("startswith_call", "def starts_with_a(s):\n    return s.startswith("),
        ("normalize_comp", "def normalize(items):\n    return [x.strip().lower() for x in "),
        ("fstring_expr", "def build_message(name):\n    return f'Hello, {"),
    ],
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def set_if_present(cfg, name: str, value) -> None:
    if hasattr(cfg, name):
        setattr(cfg, name, value)


def build_tokenizer(cfg):
    return MLCode.Python100kASTTokenizer(
        keep_string_values=cfg.keep_string_values,
        keep_identifier_values=cfg.keep_identifier_values,
        keep_module_values=cfg.keep_module_values,
        normalize_numbers=cfg.normalize_numbers,
        max_string_length=cfg.max_string_length,
    )


def print_case_result(title: str, sample_name: str, index: int, code: str, result: dict) -> None:
    print("=" * 90)
    print(f"{title} :: {index}. {sample_name}")
    print("=" * 90)
    print("[INPUT]")
    print(code)

    print("\n[ERROR]")
    print(result.get("error"))

    print("\n[FULL RECONSTRUCTED CODE]")
    print(result.get("full_generated_code") or "<EMPTY>")

    print("\n[GENERATED COMPLETION]")
    print(result.get("generated_completion_text") or "<EMPTY>")

    print("\n[NEW TOKENS COUNT]")
    print(len(result.get("new_tokens") or []))


def run_group(group_name: str, samples: list[tuple[str, str]], model, vocab, ivocab, tokenizer, cfg) -> None:
    print("\n" + "#" * 90)
    print(group_name)
    print("#" * 90)

    for index, (sample_name, code) in enumerate(samples, 1):
        result = continue_real_code_safe(
            model=model,
            code=code,
            vocab=vocab,
            ivocab=ivocab,
            tokenizer=tokenizer,
            cfg=cfg,
            prefix_len=getattr(cfg, "infer_prefix_len", 256),
            allow_incomplete_prefix=getattr(cfg, "infer_allow_incomplete_prefix", True),
            max_new_tokens=getattr(cfg, "infer_max_new_tokens", 96),
            temperature=getattr(cfg, "infer_temperature", 0.2),
            top_k=getattr(cfg, "infer_top_k", 8),
            top_p=getattr(cfg, "infer_top_p", 0.8),
            repetition_penalty=getattr(cfg, "infer_repetition_penalty", 1.02),
            fallback_to_original=False,
        )
        print_case_result(
            title=group_name,
            sample_name=sample_name,
            index=index,
            code=code,
            result=result,
        )


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    log_dir = project_dir / "test_outputs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"tests_{timestamp}.log"

    checkpoint_path = project_dir / "checkpoints_code_lm" / "best.pt"
    device = "cuda" if MLCode.torch.cuda.is_available() else "cpu"

    model, vocab, ivocab, cfg = MLCode.load_model_for_inference(
        checkpoint_path=str(checkpoint_path),
        device=device,
    )
    MLCode.set_seed(getattr(cfg, "seed", 42))

    set_if_present(cfg, "infer_allow_incomplete_prefix", True)
    set_if_present(cfg, "infer_temperature", 0.2)
    set_if_present(cfg, "infer_top_k", 8)
    set_if_present(cfg, "infer_top_p", 0.8)
    set_if_present(cfg, "infer_repetition_penalty", 1.02)
    set_if_present(cfg, "infer_max_new_tokens", 96)

    tokenizer = build_tokenizer(cfg)

    original_stdout = sys.stdout
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            print(f"[INFO] log_file={log_path}")
            print(f"[INFO] device={device}")
            print(f"[INFO] infer_allow_incomplete_prefix={getattr(cfg, 'infer_allow_incomplete_prefix', None)}")
            print(f"[INFO] infer_temperature={getattr(cfg, 'infer_temperature', None)}")
            print(f"[INFO] infer_top_k={getattr(cfg, 'infer_top_k', None)}")
            print(f"[INFO] infer_top_p={getattr(cfg, 'infer_top_p', None)}")
            print(f"[INFO] infer_repetition_penalty={getattr(cfg, 'infer_repetition_penalty', None)}")
            print(f"[INFO] infer_max_new_tokens={getattr(cfg, 'infer_max_new_tokens', None)}")

            for group_name, samples in SAMPLE_GROUPS.items():
                run_group(group_name, samples, model, vocab, ivocab, tokenizer, cfg)
        finally:
            sys.stdout.flush()
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
