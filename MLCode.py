from clearml import Task
#test
import json
import errno
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

task = Task.init(project_name='projectML', task_name='Train')
logger = task.get_logger()
PROJECT_ROOT = Path(__file__).resolve().parent
# DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints_code_lm"
# DEFAULT_TRAIN_PATH = DATA_DIR / "python100k_train.json"
# DEFAULT_VAL_PATH = DATA_DIR / "python50k_eval.json"

from clearml import Dataset

train_ds = Dataset.get(dataset_project="projectML", dataset_name="python100k_train")
val_ds = Dataset.get(dataset_project="projectML", dataset_name="python50k_eval")

DEFAULT_TRAIN_PATH = Path(train_ds.get_local_copy()) / "data" / "python100k_train.json"
DEFAULT_VAL_PATH = Path(val_ds.get_local_copy()) / "data" / "python50k_eval.json"


############################################################
# SPECIAL TOKENS
############################################################

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PROMPT_TOKEN = "<PROMPT>"
COMPLETION_TOKEN = "<COMPLETION>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
PROMPT_ID = 4
COMPLETION_ID = 5


############################################################
# CONFIG
############################################################

@dataclass
class Config:
    train_path: str = str(DEFAULT_TRAIN_PATH)
    val_path: Optional[str] = str(DEFAULT_VAL_PATH)

    seq_len: int = 1024

    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1

    epochs: int = 10
    batch_size: int = 8
    accum_steps: int = 8
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    seed: int = 42
    num_workers: int = 2
    early_stopping_patience: int = 3

    min_freq: int = 2
    max_vocab_size: Optional[int] = 20000

    # ВАЖНО: для code -> code значения должны сохраняться
    keep_string_values: bool = True
    keep_identifier_values: bool = True
    keep_module_values: bool = True
    normalize_numbers: bool = False
    max_string_length: int = 120

    checkpoint_dir: str = str(CHECKPOINT_DIR)
    resume_from: Optional[str] = None
    compile_model: bool = False

    validate_json_lines: bool = True
    train_random_crop: bool = True
    eval_random_crop: bool = False
    completion_min_prefix_len: int = 64
    completion_max_prefix_len: int = 384

    gen_temperature: float = 0.8
    gen_top_k: int = 40
    gen_top_p: float = 0.95
    gen_repetition_penalty: float = 1.05
    gen_max_new_tokens: int = 128

    # Отдельные настройки именно для code -> code inference
    infer_prefix_len: int = 256
    infer_min_completion_tokens: int = 16
    infer_min_ast_tokens: int = 8
    infer_allow_incomplete_prefix: bool = True
    infer_max_new_tokens: int = 96
    infer_temperature: float = 0.6
    infer_top_k: int = 24
    infer_top_p: float = 0.9
    infer_repetition_penalty: float = 1.15


CFG = Config()


def normalize_config_paths(cfg: Config) -> Config:
    """
    Keeps older checkpoints usable after reorganizing the project layout.
    """
    path_aliases = {
        "python100k_train.json": DEFAULT_TRAIN_PATH,
        "python50k_eval.json": DEFAULT_VAL_PATH,
        "best.pt": CHECKPOINT_DIR / "best.pt",
        "last.pt": CHECKPOINT_DIR / "last.pt",
    }

    def normalize_path(raw_path: Optional[str], default_path: Optional[Path] = None) -> Optional[str]:
        if raw_path is None:
            return str(default_path) if default_path is not None else None

        path = Path(raw_path)
        if path.is_absolute() and path.exists():
            return str(path)

        alias = path_aliases.get(path.name)
        if alias is not None:
            return str(alias)

        project_relative = PROJECT_ROOT / path
        if project_relative.exists():
            return str(project_relative)

        if default_path is not None:
            return str(default_path)
        return str(path)

    cfg.train_path = normalize_path(cfg.train_path, DEFAULT_TRAIN_PATH)
    cfg.val_path = normalize_path(cfg.val_path, DEFAULT_VAL_PATH)
    cfg.checkpoint_dir = normalize_path(cfg.checkpoint_dir, CHECKPOINT_DIR)
    cfg.resume_from = normalize_path(cfg.resume_from) if cfg.resume_from else None
    return cfg


############################################################
# UTILS
############################################################

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_json_loads(line: str):
    try:
        return json.loads(line)
    except Exception:
        return None


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def is_retryable_io_error(exc: BaseException) -> bool:
    if not isinstance(exc, OSError):
        return False
    return exc.errno in {
        errno.ENOTCONN,      # Transport endpoint is not connected
        errno.EIO,           # I/O error
        errno.ESTALE,        # Stale file handle
        errno.ETIMEDOUT,     # Timed out
        121,                 # Remote I/O error on some mounts
    }


def iter_jsonl_lines_resilient(
    path: str,
    encoding: str = "utf8",
    max_retries: int = 5,
    retry_delay: float = 2.0,
):
    """
    Iterate over a JSONL file and transparently reopen it after transient
    filesystem failures. This is especially helpful for mounted Google Drive
    paths in Colab.
    """
    line_num = 0
    byte_offset = 0
    retries = 0
    f = None

    try:
        while True:
            try:
                if f is None:
                    f = open(path, "rb")
                    if byte_offset:
                        f.seek(byte_offset)

                raw = f.readline()
                if not raw:
                    break

                byte_offset += len(raw)
                line_num += 1
                retries = 0
                yield line_num, raw, raw.decode(encoding)
            except UnicodeDecodeError:
                line_num += 1
                retries = 0
                continue
            except OSError as exc:
                if not is_retryable_io_error(exc) or retries >= max_retries:
                    raise

                retries += 1
                print(
                    f"[WARN] Read error at line {line_num + 1} "
                    f"(offset={byte_offset}): {exc}. "
                    f"Retry {retries}/{max_retries} in {retry_delay:.1f}s..."
                )

                if f is not None:
                    try:
                        f.close()
                    except Exception:
                        pass
                    f = None

                time.sleep(retry_delay)
    finally:
        if f is not None:
            try:
                f.close()
            except Exception:
                pass


############################################################
# AST TOKENIZER
############################################################

class Python100kASTTokenizer:
    """
    Converts one JSON line (list of nodes) into a structural token sequence.
    """

    IDENTIFIER_TYPES = {
        "NameLoad", "NameStore", "NameParam", "alias", "attr", "keyword",
        "ClassDef", "FunctionDef", "AsyncFunctionDef",
        "identifier", "vararg", "kwarg", "arg"
    }

    STRING_TYPES = {"Str"}
    NUMBER_TYPES = {"Num"}
    BOOL_LIKE_VALUES = {"True", "False", "None"}

    def __init__(
        self,
        keep_string_values: bool = True,
        keep_identifier_values: bool = True,
        keep_module_values: bool = True,
        normalize_numbers: bool = True,
        max_string_length: int = 80,
    ):
        self.keep_string_values = keep_string_values
        self.keep_identifier_values = keep_identifier_values
        self.keep_module_values = keep_module_values
        self.normalize_numbers = normalize_numbers
        self.max_string_length = max_string_length

    def _sanitize_value(self, value) -> str:
        text = str(value)
        text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        if len(text) > self.max_string_length:
            text = text[:self.max_string_length] + "..."
        return text

    def _value_tokens(self, node: dict) -> List[str]:
        out = []
        node_type = node.get("type", "UNKNOWN")
        value = node.get("value", None)

        if value is None:
            return out

        if isinstance(value, str) and value in self.BOOL_LIKE_VALUES:
            out.append(f"VALUE_CONST_{value}")
            return out

        if node_type in self.NUMBER_TYPES and self.normalize_numbers:
            try:
                int(value)
                out.append("VALUE_INT")
                return out
            except Exception:
                pass
            try:
                float(value)
                out.append("VALUE_FLOAT")
                return out
            except Exception:
                pass

        if node_type in self.STRING_TYPES:
            out.append("VALUE_STR")
            if self.keep_string_values:
                out.append(f"STR_{self._sanitize_value(value)}")
            return out

        if node_type in self.IDENTIFIER_TYPES:
            out.append(f"HAS_VALUE_{node_type}")
            if self.keep_identifier_values:
                out.append(f"ID_{self._sanitize_value(value)}")
            return out

        if node_type == "ImportFrom":
            out.append("VALUE_MODULE")
            if self.keep_module_values:
                out.append(f"MODULE_{self._sanitize_value(value)}")
            return out

        if isinstance(value, int):
            out.append("VALUE_INT")
            if not self.normalize_numbers:
                out.append(f"GENVAL_{self._sanitize_value(value)}")
        elif isinstance(value, float):
            out.append("VALUE_FLOAT")
            if not self.normalize_numbers:
                out.append(f"GENVAL_{self._sanitize_value(value)}")
        elif isinstance(value, str):
            out.append("VALUE_STR_GENERIC")
            if self.keep_string_values:
                out.append(f"GENVAL_{self._sanitize_value(value)}")
        else:
            out.append("VALUE_OTHER")

        return out

    def linearize(self, nodes: List[dict]) -> List[str]:
        if not nodes:
            return []

        visited = set()
        out: List[str] = []

        def dfs(idx: int):
            if idx < 0 or idx >= len(nodes):
                return
            if idx in visited:
                out.append("REF")
                return

            visited.add(idx)
            node = nodes[idx]
            node_type = node.get("type", "UNKNOWN")
            children = node.get("children", [])

            out.append(f"ENTER_{node_type}")
            out.extend(self._value_tokens(node))

            if children:
                out.append(f"ARITY_{min(len(children), 16)}")
                for child_idx in children:
                    dfs(child_idx)
            else:
                out.append("LEAF")

            out.append(f"EXIT_{node_type}")

        dfs(0)

        for idx in range(len(nodes)):
            if idx not in visited:
                out.append("EXTRA_ROOT")
                dfs(idx)

        return out


############################################################
# DATASET
############################################################

class StreamingASTDataset(Dataset):
    """
    JSONL dataset with byte offsets.
    Each line is one AST example.
    File is opened lazily per worker.
    """

    def __init__(
        self,
        path: str,
        vocab: Dict[str, int],
        tokenizer: Python100kASTTokenizer,
        seq_len: int,
        random_crop: bool = True,
        validate_json_lines: bool = False,
        min_prefix_len: int = 64,
        max_prefix_len: Optional[int] = None,
    ):
        self.path = path
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.random_crop = random_crop
        self.validate_json_lines = validate_json_lines
        self.min_prefix_len = max(1, min_prefix_len)
        self.max_prefix_len = max_prefix_len

        self.offsets: List[int] = []
        self.file = None
        self.total_nonempty_lines = 0
        self.skipped_invalid_json = 0

        self._build_offsets()

    def _build_offsets(self) -> None:
        current_offset = 0
        for _line_num, raw, line in iter_jsonl_lines_resilient(self.path):
            line_offset = current_offset
            current_offset += len(raw)

            if not line.strip():
                continue

            self.total_nonempty_lines += 1

            if self.validate_json_lines:
                try:
                    json.loads(line)
                    self.offsets.append(line_offset)
                except Exception:
                    self.skipped_invalid_json += 1
            else:
                self.offsets.append(line_offset)

        print(
            f"[INFO] Dataset offsets built for {self.path} | "
            f"kept={len(self.offsets)} | "
            f"skipped_invalid={self.skipped_invalid_json}"
        )

    def _ensure_open(self) -> None:
        if self.file is None:
            self.file = open(self.path, "rb")

    def close(self) -> None:
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
            self.file = None

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_raw_line_at(self, offset: int, max_retries: int = 5) -> bytes:
        retries = 0
        while True:
            try:
                self._ensure_open()
                self.file.seek(offset)
                return self.file.readline()
            except OSError as exc:
                self.close()
                if not is_retryable_io_error(exc) or retries >= max_retries:
                    raise
                retries += 1
                print(
                    f"[WARN] Sample read error at offset={offset}: {exc}. "
                    f"Retry {retries}/{max_retries}..."
                )
                time.sleep(1.0)

    def _build_completion_example(self, token_ids: List[int]) -> List[int]:
        if len(token_ids) < 2:
            return [BOS_ID, PROMPT_ID, COMPLETION_ID, EOS_ID]

        max_prefix_by_seq = max(1, self.seq_len - 3)
        max_prefix = min(max_prefix_by_seq, len(token_ids) - 1)
        if self.max_prefix_len is not None:
            max_prefix = min(max_prefix, self.max_prefix_len)

        min_prefix = min(self.min_prefix_len, max_prefix)

        if self.random_crop and min_prefix < max_prefix:
            prefix_len = random.randint(min_prefix, max_prefix)
        else:
            prefix_len = max_prefix

        prefix_ids = token_ids[:prefix_len]
        tail_ids = token_ids[prefix_len:]
        return [BOS_ID, PROMPT_ID] + prefix_ids + [COMPLETION_ID] + tail_ids + [EOS_ID]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self._read_raw_line_at(self.offsets[idx])

        try:
            line = raw.decode("utf8")
            nodes = json.loads(line)
        except Exception:
            x = torch.full((self.seq_len,), PAD_ID, dtype=torch.long)
            y = torch.full((self.seq_len,), PAD_ID, dtype=torch.long)
            x[0] = BOS_ID
            y[0] = EOS_ID
            return x, y

        tokens = self.tokenizer.linearize(nodes)
        token_ids = [self.vocab.get(tok, UNK_ID) for tok in tokens]
        ids = self._build_completion_example(token_ids)

        target_len = self.seq_len + 1
        ids = ids[:target_len]
        if len(ids) < target_len:
            ids = ids + [PAD_ID] * (target_len - len(ids))

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        completion_positions = (x == COMPLETION_ID).nonzero(as_tuple=False)
        if completion_positions.numel() > 0:
            completion_idx = completion_positions[0].item()
            y[:completion_idx] = PAD_ID
        else:
            y[:] = PAD_ID
        return x, y


############################################################
# VOCAB
############################################################

def build_vocab_streaming(
    path: str,
    tokenizer: Python100kASTTokenizer,
    min_freq: int = 2,
    max_vocab_size: Optional[int] = None,
    seq_len_for_stats: Optional[int] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter()

    num_examples = 0
    total_tokenized_len = 0
    max_tokenized_len = 0
    truncated_examples = 0
    invalid_json = 0

    target_len = None if seq_len_for_stats is None else seq_len_for_stats + 1

    for line_num, _raw, line in iter_jsonl_lines_resilient(path):
        if not line.strip():
            continue

        nodes = safe_json_loads(line)
        if nodes is None:
            invalid_json += 1
            if invalid_json <= 10:
                print(f"[WARN] Invalid JSON at line {line_num}")
            continue

        toks = tokenizer.linearize(nodes)
        counter.update(toks)

        sample_len = len(toks) + 2
        num_examples += 1
        total_tokenized_len += sample_len
        max_tokenized_len = max(max_tokenized_len, sample_len)

        if target_len is not None and sample_len > target_len:
            truncated_examples += 1

        if line_num % 100000 == 0:
            print(f"[INFO] vocab pass processed {line_num} lines")

    vocab = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
        BOS_TOKEN: BOS_ID,
        EOS_TOKEN: EOS_ID,
        PROMPT_TOKEN: PROMPT_ID,
        COMPLETION_TOKEN: COMPLETION_ID,
    }

    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        room = max(0, max_vocab_size - len(vocab))
        items = items[:room]

    for tok, _freq in items:
        if tok not in vocab:
            vocab[tok] = len(vocab)

    ivocab = {idx: tok for tok, idx in vocab.items()}

    avg_len = total_tokenized_len / max(1, num_examples)
    trunc_ratio = truncated_examples / max(1, num_examples)

    print(
        f"[INFO] Vocab stats | examples={num_examples} | "
        f"invalid_json={invalid_json} | avg_len={avg_len:.2f} | "
        f"max_len={max_tokenized_len} | trunc_ratio@seq_len={trunc_ratio:.2%} | "
        f"vocab_size={len(vocab)}"
    )

    return vocab, ivocab


############################################################
# MODEL
############################################################

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(
                torch.ones(t, t, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            att = att.masked_fill(causal_mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.dropout, training=self.training)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out_proj(out)
        out = self.resid_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden = d_model * ff_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ff_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ASTGPT(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                ff_mult=cfg.ff_mult,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t = x.shape
        if t > self.cfg.seq_len:
            raise ValueError(f"Sequence length {t} > seq_len={self.cfg.seq_len}")

        pos = torch.arange(0, t, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.drop(h)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        return self.head(h)


############################################################
# SCHEDULER
############################################################

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_num = 0
        self.apply_lr(self.get_lr(0))

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.max_lr - self.min_lr)

    def apply_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def step(self) -> None:
        self.step_num += 1
        self.apply_lr(self.get_lr(self.step_num))

    def state_dict(self) -> Dict:
        return {
            "step_num": self.step_num,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.step_num = state["step_num"]
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.max_lr = state["max_lr"]
        self.min_lr = state["min_lr"]
        self.apply_lr(self.get_lr(self.step_num))


############################################################
# CHECKPOINTS
############################################################

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineScheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    vocab: Dict[str, int],
    ivocab: Dict[int, str],
    cfg: Config,
) -> None:
    raw_model = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "vocab": vocab,
        "ivocab": ivocab,
        "config": asdict(cfg),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[WarmupCosineScheduler] = None,
    map_location: str = "cpu",
) -> Dict:
    ckpt = torch.load(path, map_location=map_location)
    raw_model = unwrap_model(model)
    raw_model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt


############################################################
# EVAL
############################################################

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    use_amp = device == "cuda"

    for x, y in loader:
        x = x.to(device, non_blocking=(device == "cuda"))
        y = y.to(device, non_blocking=(device == "cuda"))

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=PAD_ID,
                reduction="sum",
            )

        total_loss += loss_sum.item()
        total_tokens += (y != PAD_ID).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


############################################################
# GENERATION
############################################################

def apply_top_k_top_p(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k)
        masked = torch.full_like(logits, float("-inf"))
        masked[indices] = values
        logits = masked

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        cutoff = cumulative_probs > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False

        sorted_logits[cutoff] = float("-inf")
        new_logits = torch.full_like(logits, float("-inf"))
        new_logits[sorted_indices] = sorted_logits
        logits = new_logits

    return logits


def compute_ast_depth(tokens: List[str]) -> int:
    depth = 0
    for tok in tokens:
        if tok.startswith("ENTER_"):
            depth += 1
        elif tok.startswith("EXIT_"):
            depth -= 1
    return depth


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    start_tokens: List[str],
    vocab: Dict[str, int],
    ivocab: Dict[int, str],
    device: str,
    cfg: Optional[Config] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    stop_on_ast_completion: bool = False,
    initial_ast_depth: int = 0,
    min_ast_tokens_before_eos: int = 0,
) -> List[str]:
    model.eval()
    cfg = cfg or getattr(unwrap_model(model), "cfg", None)
    if cfg is None:
        raise ValueError("Config must be passed to generate_tokens or exist as model.cfg")

    token_ids = [BOS_ID] + [vocab.get(tok, UNK_ID) for tok in start_tokens]
    ast_depth = initial_ast_depth
    generated_ast_tokens = 0

    for _ in range(max_new_tokens):
        x = torch.tensor(token_ids[-cfg.seq_len:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1]
        # запрещаем бесполезные спецтокены при генерации
        logits[PAD_ID] = float("-inf")
        logits[UNK_ID] = float("-inf")
        logits[BOS_ID] = float("-inf")
        logits[PROMPT_ID] = float("-inf")
        logits[COMPLETION_ID] = float("-inf")
        if generated_ast_tokens < min_ast_tokens_before_eos:
            logits[EOS_ID] = float("-inf")

        if repetition_penalty > 1.0:
            for tid in set(token_ids[-128:]):
                if 0 <= tid < logits.numel():
                    if logits[tid] > 0:
                        logits[tid] /= repetition_penalty
                    else:
                        logits[tid] *= repetition_penalty

        logits = logits / max(temperature, 1e-6)
        logits = apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)

        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            probs = torch.zeros_like(probs)
            probs[EOS_ID] = 1.0

        next_id = torch.multinomial(probs, num_samples=1).item()
        token_ids.append(next_id)

        next_tok = ivocab.get(next_id, UNK_TOKEN)
        if next_tok.startswith("ENTER_"):
            ast_depth += 1
            generated_ast_tokens += 1
        elif next_tok.startswith("EXIT_"):
            ast_depth -= 1
            generated_ast_tokens += 1
        elif next_tok not in {
            EOS_TOKEN,
            PAD_TOKEN,
            UNK_TOKEN,
            BOS_TOKEN,
            PROMPT_TOKEN,
            COMPLETION_TOKEN,
        }:
            generated_ast_tokens += 1

        if stop_on_ast_completion and generated_ast_tokens > 0 and ast_depth <= 0:
            break

        if next_id == EOS_ID:
            break

    return [ivocab.get(tid, UNK_TOKEN) for tid in token_ids]


############################################################
# CODE -> CODE INFERENCE API
############################################################

@torch.no_grad()
def generate_code_from_source(
    model: nn.Module,
    code: str,
    vocab: Dict[str, int],
    ivocab: Dict[int, str],
    cfg: Config,
):
    """
    High-level wrapper:
        raw Python code -> generated Python code

    Требует parse_python.py с функцией continue_real_code(...).
    """
    from parse_python import continue_real_code

    tokenizer = Python100kASTTokenizer(
        keep_string_values=cfg.keep_string_values,
        keep_identifier_values=cfg.keep_identifier_values,
        keep_module_values=cfg.keep_module_values,
        normalize_numbers=cfg.normalize_numbers,
        max_string_length=cfg.max_string_length,
    )

    result = continue_real_code(
        model=model,
        code=code,
        vocab=vocab,
        ivocab=ivocab,
        tokenizer=tokenizer,
        cfg=cfg,
        prefix_len=cfg.infer_prefix_len,
        allow_incomplete_prefix=cfg.infer_allow_incomplete_prefix,
        max_new_tokens=cfg.infer_max_new_tokens,
        temperature=cfg.infer_temperature,
        top_k=cfg.infer_top_k,
        top_p=cfg.infer_top_p,
        repetition_penalty=cfg.infer_repetition_penalty,
    )

    return result


def load_model_for_inference(
    checkpoint_path: str,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    saved_cfg = ckpt.get("config", None)
    if saved_cfg is not None:
        cfg = Config(**saved_cfg)
    else:
        cfg = CFG
    cfg = normalize_config_paths(cfg)

    vocab = ckpt["vocab"]
    ivocab = ckpt["ivocab"]

    model = ASTGPT(vocab_size=len(vocab), cfg=cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed in inference mode: {e}")

    return model, vocab, ivocab, cfg


def complete_code(
    checkpoint_path: str,
    code: str,
    device: Optional[str] = None,
) -> str:
    model, vocab, ivocab, cfg = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    result = generate_code_from_source(
        model=model,
        code=code,
        vocab=vocab,
        ivocab=ivocab,
        cfg=cfg,
    )

    return result.get("generated_code", "") or result.get("generated_completion_text", "")


############################################################
# TRAIN
############################################################

def train(cfg: Config):
    cfg = normalize_config_paths(cfg)
    set_seed(cfg.seed)
    ensure_dir(cfg.checkpoint_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    print(f"[INFO] Device: {device}")

    tokenizer = Python100kASTTokenizer(
        keep_string_values=cfg.keep_string_values,
        keep_identifier_values=cfg.keep_identifier_values,
        keep_module_values=cfg.keep_module_values,
        normalize_numbers=cfg.normalize_numbers,
        max_string_length=cfg.max_string_length,
    )

    print("[INFO] Building vocab...")
    vocab, ivocab = build_vocab_streaming(
        cfg.train_path,
        tokenizer=tokenizer,
        min_freq=cfg.min_freq,
        max_vocab_size=cfg.max_vocab_size,
        seq_len_for_stats=cfg.seq_len,
    )

    train_ds = StreamingASTDataset(
        path=cfg.train_path,
        vocab=vocab,
        tokenizer=tokenizer,
        seq_len=cfg.seq_len,
        random_crop=cfg.train_random_crop,
        validate_json_lines=cfg.validate_json_lines,
        min_prefix_len=cfg.completion_min_prefix_len,
        max_prefix_len=cfg.completion_max_prefix_len,
    )

    val_ds = None
    if cfg.val_path:
        val_ds = StreamingASTDataset(
            path=cfg.val_path,
            vocab=vocab,
            tokenizer=tokenizer,
            seq_len=cfg.seq_len,
            random_crop=cfg.eval_random_crop,
            validate_json_lines=cfg.validate_json_lines,
            min_prefix_len=cfg.completion_min_prefix_len,
            max_prefix_len=cfg.completion_max_prefix_len,
        )

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(cfg.num_workers > 0),
            worker_init_fn=seed_worker,
        )

    model = ASTGPT(len(vocab), cfg).to(device)

    if cfg.compile_model and hasattr(torch, "compile"):
        print("[INFO] Compiling model...")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    updates_per_epoch = math.ceil(len(train_loader) / cfg.accum_steps)
    total_updates = cfg.epochs * updates_per_epoch

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=cfg.warmup_steps,
        total_steps=total_updates,
        max_lr=cfg.lr,
        min_lr=cfg.min_lr,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loss_fn = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=cfg.label_smoothing,
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    bad_epochs = 0

    if cfg.resume_from:
        ckpt = load_checkpoint(cfg.resume_from, model, optimizer, scheduler, map_location=device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[INFO] Resumed from epoch={start_epoch}, global_step={global_step}")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_start = time.time()
        total_train_loss = 0.0
        train_steps = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=(device == "cuda"))
            y = y.to(device, non_blocking=(device == "cuda"))

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = train_loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / cfg.accum_steps

            scaler.scale(loss).backward()

            total_train_loss += loss.item() * cfg.accum_steps
            train_steps += 1

            do_step = ((batch_idx + 1) % cfg.accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

        avg_train_loss = total_train_loss / max(1, train_steps)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 20 else float("inf")
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.report_scalar("loss", "train", iteration=epoch + 1, value=avg_train_loss)
        logger.report_scalar("ppl", "train", iteration=epoch + 1, value=train_ppl)
        logger.report_scalar("lr", "train", iteration=epoch + 1, value=current_lr)
        logger.report_scalar("time_sec", "train", iteration=epoch + 1, value=epoch_time)
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"train_loss={avg_train_loss:.4f} | train_ppl={train_ppl:.2f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6e} | "
            f"time={epoch_time:.1f}s"
        )

        avg_val_loss = None
        val_ppl = None

       if val_loader is not None:
    avg_val_loss, val_ppl = evaluate(model, val_loader, device)

    logger.report_scalar("loss", "val", iteration=epoch + 1, value=avg_val_loss)
    logger.report_scalar("ppl", "val", iteration=epoch + 1, value=val_ppl)

    print(
        f"Epoch {epoch + 1}/{cfg.epochs} | "
        f"val_loss={avg_val_loss:.4f} | "
        f"val_ppl={val_ppl:.2f}"
    )

        last_path = str(Path(cfg.checkpoint_dir) / "last.pt")
        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            vocab=vocab,
            ivocab=ivocab,
            cfg=cfg,
        )

        improved = avg_val_loss is not None and avg_val_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss
            bad_epochs = 0

            best_path = str(Path(cfg.checkpoint_dir) / "best.pt")
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
                vocab=vocab,
                ivocab=ivocab,
                cfg=cfg,
            )
            print(f"[INFO] Saved new best checkpoint: {best_path}")
        elif val_loader is not None:
            bad_epochs += 1
            print(f"[INFO] No improvement. patience={bad_epochs}/{cfg.early_stopping_patience}")
            if bad_epochs >= cfg.early_stopping_patience:
                print("[INFO] Early stopping triggered.")
                break

    train_ds.close()
    if val_ds is not None:
        val_ds.close()

    return model, vocab, ivocab


############################################################
# TESTS / DEMO
############################################################

def test_ast_tokens(
    model: nn.Module,
    vocab: Dict[str, int],
    ivocab: Dict[int, str],
    cfg: Optional[Config] = None,
) -> None:
    raw_model = unwrap_model(model)
    cfg = cfg or raw_model.cfg
    device = next(model.parameters()).device

    start_tokens = [
        "ENTER_Module",
        "ARITY_3",
        "ENTER_Import",
        "ARITY_1",
        "ENTER_alias",
        "HAS_VALUE_alias",
    ]

    out = generate_tokens(
        model=model,
        start_tokens=start_tokens,
        vocab=vocab,
        ivocab=ivocab,
        device=device,
        cfg=cfg,
        max_new_tokens=cfg.gen_max_new_tokens,
        temperature=cfg.gen_temperature,
        top_k=cfg.gen_top_k,
        top_p=cfg.gen_top_p,
        repetition_penalty=cfg.gen_repetition_penalty,
    )

    print("\nGenerated AST token continuation:\n")
    print(" ".join(out))


def test_code_to_code(
    checkpoint_path: str,
    sample_code: str,
    device: Optional[str] = None,
) -> None:
    try:
        generated = complete_code(
            checkpoint_path=checkpoint_path,
            code=sample_code,
            device=device,
        )
        print("===== INPUT CODE =====")
        print(sample_code)
        print()
        print("===== GENERATED CODE =====")
        print(generated)
    except Exception as e:
        print(f"[ERROR] code generation failed: {e}")


############################################################
# MAIN
############################################################

if __name__ == "__main__":
    mode = "train"  # "train" | "infer"

    if mode == "train":
        model, vocab, ivocab = train(CFG)
        test_ast_tokens(model, vocab, ivocab, CFG)

    elif mode == "infer":
        checkpoint_path = CFG.resume_from or str(Path(CFG.checkpoint_dir) / "best.pt")

        sample_code = """
def add(a, b):
    return a + b
""".strip()

        if Path(checkpoint_path).exists():
            test_code_to_code(
                checkpoint_path=checkpoint_path,
                sample_code=sample_code,
            )
        else:
            print(f"[WARN] Checkpoint not found: {checkpoint_path}")
