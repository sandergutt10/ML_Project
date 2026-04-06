import ast
import io
import keyword
import math
import re
import token as token_module
import tokenize
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from MLCode import (
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
    PROMPT_TOKEN,
    COMPLETION_TOKEN,
    compute_ast_depth,
    generate_tokens,
)


TOLERANT_EXPR_HOLE = "__CODEX_CONTINUE_EXPR__"
TOLERANT_STMT_HOLE = "__CODEX_CONTINUE_STMT__"
TOLERANT_ATTR_HOLE = "__codex_continue_attr__"


############################################################
# CODE -> DATASET-STYLE JSON TREE
############################################################

def tokenize_python_code(code: str) -> List[tokenize.TokenInfo]:
    """
    Tokenizes Python source with the stdlib tokenizer so raw input first goes
    through lexical analysis before we build any AST nodes.
    """
    reader = io.StringIO(code).readline
    return list(tokenize.generate_tokens(reader))


def rebuild_source_from_tokens(tokens: List[tokenize.TokenInfo]) -> str:
    """
    Rebuilds source code from tokenizer output. This keeps the pipeline
    explicitly `input -> tokenize -> source -> ast.parse`.
    """
    significant_tokens = [
        tok
        for tok in tokens
        if tok.type != token_module.ENDMARKER
    ]
    return tokenize.untokenize(significant_tokens)


def tokenize_and_parse_code(
    code: str,
    filename: str = "<string>",
) -> Dict[str, Any]:
    """
    Full frontend pipeline required by the project review:
    raw code -> stdlib tokenize -> normalized source -> ast.parse
    """
    python_tokens = tokenize_python_code(code)
    normalized_source = rebuild_source_from_tokens(python_tokens)
    tree = ast.parse(normalized_source, filename=filename)
    return {
        "python_tokens": python_tokens,
        "normalized_source": normalized_source,
        "ast_tree": tree,
    }


def parse_code_to_json_tree(
    code: str,
    filename: str = "<string>",
    parsed: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Python code string -> dataset-style json_tree (list[dict]).
    Structure is intentionally close to your original parser.
    """
    parsed = parsed or tokenize_and_parse_code(code, filename=filename)
    tree = parsed["ast_tree"]
    json_tree: List[dict] = []

    def gen_identifier(identifier, node_type="identifier"):
        pos = len(json_tree)
        json_node = {
            "type": node_type,
            "value": str(identifier),
        }
        json_tree.append(json_node)
        return pos

    def traverse_list(items, node_type="list"):
        pos = len(json_tree)
        json_node = {"type": node_type}
        json_tree.append(json_node)

        children = []
        for item in items:
            children.append(traverse(item))

        if children:
            json_node["children"] = children
        return pos

    def traverse(node):
        pos = len(json_tree)
        json_node = {"type": type(node).__name__}
        json_tree.append(json_node)

        children = []

        # value fields
        if isinstance(node, ast.Name):
            json_node["value"] = node.id

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                json_node["type"] = "Str"
                json_node["value"] = node.value
            elif isinstance(node.value, (int, float)):
                json_node["type"] = "Num"
                json_node["value"] = str(node.value)
            elif node.value in (True, False, None):
                json_node["type"] = "Name"
                json_node["value"] = str(node.value)

        elif hasattr(ast, "Num") and isinstance(node, ast.Num):
            json_node["value"] = str(node.n)

        elif hasattr(ast, "Str") and isinstance(node, ast.Str):
            json_node["value"] = node.s

        elif isinstance(node, ast.alias):
            json_node["value"] = str(node.name)
            if node.asname:
                children.append(gen_identifier(node.asname))

        elif isinstance(node, ast.arg):
            json_node["value"] = str(node.arg)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            json_node["value"] = str(node.name)

        elif isinstance(node, ast.ClassDef):
            json_node["value"] = str(node.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                json_node["value"] = str(node.module)

        elif isinstance(node, ast.Global):
            for n in node.names:
                children.append(gen_identifier(n))

        elif isinstance(node, ast.keyword):
            if node.arg is not None:
                json_node["value"] = str(node.arg)

        # structured child handling
        if isinstance(node, ast.For):
            children.append(traverse(node.target))
            children.append(traverse(node.iter))
            children.append(traverse_list(node.body, "body"))
            if node.orelse:
                children.append(traverse_list(node.orelse, "orelse"))

        elif isinstance(node, (ast.If, ast.While)):
            children.append(traverse(node.test))
            children.append(traverse_list(node.body, "body"))
            if node.orelse:
                children.append(traverse_list(node.orelse, "orelse"))

        elif isinstance(node, ast.With):
            for item in node.items:
                children.append(traverse(item.context_expr))
                if item.optional_vars is not None:
                    children.append(traverse(item.optional_vars))
            children.append(traverse_list(node.body, "body"))

        elif isinstance(node, ast.Try):
            if node.body:
                children.append(traverse_list(node.body, "body"))
            if node.handlers:
                children.append(traverse_list(node.handlers, "handlers"))
            if node.orelse:
                children.append(traverse_list(node.orelse, "orelse"))
            if node.finalbody:
                children.append(traverse_list(node.finalbody, "finalbody"))

        elif isinstance(node, ast.arguments):
            # Python 3 full arguments support
            children.append(traverse_list(getattr(node, "posonlyargs", []), "posonlyargs"))
            children.append(traverse_list(node.args, "args"))
            children.append(traverse_list(node.kwonlyargs, "kwonlyargs"))
            children.append(traverse_list(node.defaults, "defaults"))
            children.append(traverse_list(node.kw_defaults, "kw_defaults"))

            if getattr(node, "vararg", None):
                vararg_name = node.vararg.arg if isinstance(node.vararg, ast.arg) else node.vararg
                children.append(gen_identifier(vararg_name, "vararg"))

            if getattr(node, "kwarg", None):
                kwarg_name = node.kwarg.arg if isinstance(node.kwarg, ast.arg) else node.kwarg
                children.append(gen_identifier(kwarg_name, "kwarg"))

        elif isinstance(node, ast.ExceptHandler):
            if node.type:
                children.append(traverse_list([node.type], "type"))

            if node.name:
                if isinstance(node.name, str):
                    children.append(traverse_list([ast.Name(id=node.name, ctx=ast.Load())], "name"))
                else:
                    children.append(traverse_list([node.name], "name"))

            children.append(traverse_list(node.body, "body"))

        elif isinstance(node, ast.ClassDef):
            children.append(traverse_list(node.bases, "bases"))
            children.append(traverse_list(getattr(node, "keywords", []), "keywords"))
            children.append(traverse_list(node.body, "body"))
            children.append(traverse_list(node.decorator_list, "decorator_list"))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            children.append(traverse(node.args))
            children.append(traverse_list(node.body, "body"))
            children.append(traverse_list(node.decorator_list, "decorator_list"))
            if getattr(node, "returns", None) is not None:
                children.append(traverse_list([node.returns], "returns"))

        else:
            for child in ast.iter_child_nodes(node):
                if isinstance(
                    child,
                    (
                        ast.expr_context,
                        ast.operator,
                        ast.boolop,
                        ast.unaryop,
                        ast.cmpop,
                    ),
                ):
                    json_node["type"] = json_node["type"] + type(child).__name__
                else:
                    children.append(traverse(child))

        if isinstance(node, ast.Attribute):
            children.append(gen_identifier(node.attr, "attr"))

        if children:
            json_node["children"] = children

        return pos

    traverse(tree)
    return json_tree


############################################################
# SCORING REAL CODE
############################################################

@torch.no_grad()
def score_real_code(model, code: str, vocab, tokenizer, cfg):
    """
    Scores a real Python code snippet with the trained AST LM.
    """
    device = next(model.parameters()).device

    parsed = tokenize_and_parse_code(code)
    json_tree = parse_code_to_json_tree(code, parsed=parsed)
    tokens = tokenizer.linearize(json_tree)
    ids = [BOS_ID] + [vocab.get(tok, UNK_ID) for tok in tokens] + [EOS_ID]

    total_loss = 0.0
    total_tokens = 0

    start = 0
    while start < len(ids) - 1:
        chunk = ids[start:start + cfg.seq_len + 1]
        if len(chunk) < 2:
            break

        if len(chunk) < cfg.seq_len + 1:
            chunk = chunk + [PAD_ID] * (cfg.seq_len + 1 - len(chunk))

        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)

        logits = model(x)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=PAD_ID,
            reduction="sum",
        )

        valid = (y != PAD_ID).sum().item()
        total_loss += loss_sum.item()
        total_tokens += valid

        start += cfg.seq_len

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return {
        "avg_loss": avg_loss,
        "ppl": ppl,
        "python_tokens": parsed["python_tokens"],
        "normalized_source": parsed["normalized_source"],
        "num_json_nodes": len(json_tree),
        "num_tokens": len(tokens),
        "tokens_preview": tokens[:80],
    }


############################################################
# TOKENS -> JSON TREE
############################################################

def _desanitize_value(text: str) -> str:
    return (
        text.replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
    )


def _normalize_node_type(node_type: str) -> str:
    aliases = {
        "TryExcept": "Try",
        "TryFinally": "Try",
    }
    return aliases.get(node_type, node_type)


def decode_linearized_tokens_to_json_tree(tokens: List[str]) -> List[dict]:
    """
    Rebuilds a dataset-style json_tree from structural AST tokens.

    Supported token families:
      ENTER_X / EXIT_X
      ARITY_n
      LEAF
      HAS_VALUE_*
      VALUE_STR / STR_*
      VALUE_MODULE / MODULE_*
      VALUE_STR_GENERIC / GENVAL_*
      VALUE_INT / VALUE_FLOAT / VALUE_CONST_*
      ID_*
      REF / EXTRA_ROOT (ignored as best effort)

    This is a best-effort structural decoder.
    """
    nodes: List[dict] = []

    class Frame:
        def __init__(self, node_idx: int, node_type: str):
            self.node_idx = node_idx
            self.node_type = node_type
            self.children: List[int] = []

    def attach_finished_frame(finished: Frame) -> None:
        if finished.children:
            nodes[finished.node_idx]["children"] = finished.children
        if stack:
            stack[-1].children.append(finished.node_idx)

    stack: List[Frame] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok in ("<BOS>", "<EOS>", "<PAD>", "<UNK>", "REF", "EXTRA_ROOT"):
            i += 1
            continue

        if tok.startswith("ENTER_"):
            node_type = _normalize_node_type(tok[len("ENTER_"):])
            node = {"type": node_type}
            node_idx = len(nodes)
            nodes.append(node)
            stack.append(Frame(node_idx, node_type))
            i += 1
            continue

        if not stack:
            i += 1
            continue

        frame = stack[-1]
        node = nodes[frame.node_idx]

        if tok.startswith("HAS_VALUE_"):
            i += 1
            continue

        if tok == "VALUE_STR":
            if i + 1 < len(tokens) and tokens[i + 1].startswith("STR_"):
                node["value"] = _desanitize_value(tokens[i + 1][len("STR_"):])
                i += 2
            else:
                node["value"] = ""
                i += 1
            continue

        if tok == "VALUE_MODULE":
            if i + 1 < len(tokens) and tokens[i + 1].startswith("MODULE_"):
                node["value"] = _desanitize_value(tokens[i + 1][len("MODULE_"):])
                i += 2
            else:
                node["value"] = ""
                i += 1
            continue

        if tok == "VALUE_STR_GENERIC":
            if i + 1 < len(tokens) and tokens[i + 1].startswith("GENVAL_"):
                node["value"] = _desanitize_value(tokens[i + 1][len("GENVAL_"):])
                i += 2
            else:
                node["value"] = ""
                i += 1
            continue

        if tok == "VALUE_INT":
            if i + 1 < len(tokens) and tokens[i + 1].startswith("GENVAL_"):
                node["value"] = _desanitize_value(tokens[i + 1][len("GENVAL_"):])
                i += 2
            else:
                node["value"] = "0"
                i += 1
            continue

        if tok == "VALUE_FLOAT":
            if i + 1 < len(tokens) and tokens[i + 1].startswith("GENVAL_"):
                node["value"] = _desanitize_value(tokens[i + 1][len("GENVAL_"):])
                i += 2
            else:
                node["value"] = "0.0"
                i += 1
            continue

        if tok == "VALUE_CONST_True":
            node["value"] = "True"
            i += 1
            continue

        if tok == "VALUE_CONST_False":
            node["value"] = "False"
            i += 1
            continue

        if tok == "VALUE_CONST_None":
            node["value"] = "None"
            i += 1
            continue

        if tok.startswith("ID_"):
            node["value"] = _desanitize_value(tok[len("ID_"):])
            i += 1
            continue

        if tok.startswith("ARITY_"):
            i += 1
            continue

        if tok == "LEAF":
            i += 1
            continue

        if tok.startswith("EXIT_"):
            exit_type = _normalize_node_type(tok[len("EXIT_"):])
            if not stack:
                i += 1
                continue

            if stack[-1].node_type == exit_type:
                finished = stack.pop()
                attach_finished_frame(finished)
                i += 1
                continue

            matching_index = None
            for frame_idx in range(len(stack) - 1, -1, -1):
                if stack[frame_idx].node_type == exit_type:
                    matching_index = frame_idx
                    break

            if matching_index is None:
                i += 1
                continue

            while len(stack) - 1 >= matching_index:
                finished = stack.pop()
                attach_finished_frame(finished)

            i += 1
            continue

        i += 1

    while stack:
        finished = stack.pop()
        attach_finished_frame(finished)

    return nodes


############################################################
# JSON TREE -> PYTHON AST
############################################################

def _node(nodes: List[dict], idx: Optional[int]) -> Optional[dict]:
    if idx is None:
        return None
    if idx < 0 or idx >= len(nodes):
        return None
    return nodes[idx]


def _children(nodes: List[dict], idx: int) -> List[int]:
    return list(nodes[idx].get("children", []))


def _value(nodes: List[dict], idx: int, default: str = "") -> str:
    return str(nodes[idx].get("value", default))


def _ctx_from_type(node_type: str) -> ast.expr_context:
    if "Store" in node_type:
        return ast.Store()
    if "Param" in node_type:
        return ast.Param()
    return ast.Load()


def _sanitize_identifier(name: str, default: str = "x") -> str:
    name = str(name or "").strip()
    if not name:
        return default
    if name in {"True", "False", "None"}:
        return default
    if keyword.iskeyword(name):
        return default
    if not name.isidentifier():
        return default
    return name


def _op_from_type(node_type: str):
    mapping = {
        "Add": ast.Add,
        "Sub": ast.Sub,
        "Mult": ast.Mult,
        "Div": ast.Div,
        "Mod": ast.Mod,
        "Pow": ast.Pow,
        "FloorDiv": ast.FloorDiv,
        "MatMult": ast.MatMult,
        "LShift": ast.LShift,
        "RShift": ast.RShift,
        "BitOr": ast.BitOr,
        "BitXor": ast.BitXor,
        "BitAnd": ast.BitAnd,
    }
    for key, cls in mapping.items():
        if key in node_type:
            return cls()
    return ast.Add()


def _cmp_ops_from_type(node_type: str):
    mapping = {
        "Eq": ast.Eq,
        "NotEq": ast.NotEq,
        "LtE": ast.LtE,
        "Lt": ast.Lt,
        "GtE": ast.GtE,
        "Gt": ast.Gt,
        "IsNot": ast.IsNot,
        "Is": ast.Is,
        "NotIn": ast.NotIn,
        "In": ast.In,
    }
    ops = []
    for key, cls in mapping.items():
        if key in node_type:
            ops.append(cls())
    return ops or [ast.Eq()]


def _boolop_from_type(node_type: str):
    if "Or" in node_type:
        return ast.Or()
    return ast.And()


def _unaryop_from_type(node_type: str):
    if "USub" in node_type:
        return ast.USub()
    if "UAdd" in node_type:
        return ast.UAdd()
    if "Invert" in node_type:
        return ast.Invert()
    return ast.Not()


def _safe_expr(x: Any) -> ast.expr:
    if isinstance(x, ast.expr):
        return x
    if isinstance(x, str):
        return ast.Name(id=_sanitize_identifier(x), ctx=ast.Load())
    return ast.Constant(value=None)


def _safe_store_expr(x: Any, default: str = "x") -> ast.expr:
    if isinstance(x, ast.Name):
        return ast.Name(id=_sanitize_identifier(x.id, default), ctx=ast.Store())
    if isinstance(x, ast.Attribute):
        return ast.Attribute(value=x.value, attr=_sanitize_identifier(x.attr, "attr"), ctx=ast.Store())
    if isinstance(x, ast.Subscript):
        return ast.Subscript(value=x.value, slice=x.slice, ctx=ast.Store())
    if isinstance(x, ast.Tuple):
        return ast.Tuple(elts=[_safe_store_expr(e, default) for e in x.elts], ctx=ast.Store())
    if isinstance(x, ast.List):
        return ast.List(elts=[_safe_store_expr(e, default) for e in x.elts], ctx=ast.Store())
    if isinstance(x, str):
        return ast.Name(id=_sanitize_identifier(x, default), ctx=ast.Store())
    return ast.Name(id=default, ctx=ast.Store())


def _safe_stmt(x: Any) -> ast.stmt:
    if isinstance(x, ast.stmt):
        return x
    if isinstance(x, ast.expr):
        return ast.Expr(value=x)
    return ast.Pass()


def _safe_stmt_list(xs: Any) -> List[ast.stmt]:
    if isinstance(xs, list):
        out = []
        for x in xs:
            out.append(_safe_stmt(x))
        return out or [ast.Pass()]
    return [_safe_stmt(xs)]


def _safe_expr_list(xs: Any) -> List[ast.expr]:
    if isinstance(xs, list):
        out = []
        for x in xs:
            out.append(_safe_expr(x))
        return out
    return [_safe_expr(xs)]


def json_tree_to_python_ast(json_tree: List[dict]) -> ast.AST:
    """
    Main reverse builder.
    Best effort: supports common Python node types and returns valid code when possible.
    """

    def build(idx: int) -> Any:
        n = json_tree[idx]
        t = n["type"]
        ch = _children(json_tree, idx)

        # list-like containers
        if t in {
            "body", "orelse", "finalbody", "handlers", "bases", "decorator_list",
            "args", "defaults", "type", "name", "list", "keywords",
            "posonlyargs", "kwonlyargs", "kw_defaults", "returns"
        }:
            return [build(c) for c in ch]

        # module
        if t == "Module":
            body = [_safe_stmt(build(c)) for c in ch]
            return ast.Module(body=body, type_ignores=[])

        # identifiers / literals
        if t.startswith("Name"):
            v = _value(json_tree, idx, "x")
            if t in {"Name", "NameLoad"} and v == "True":
                return ast.Constant(value=True)
            if t in {"Name", "NameLoad"} and v == "False":
                return ast.Constant(value=False)
            if t in {"Name", "NameLoad"} and v == "None":
                return ast.Constant(value=None)
            return ast.Name(id=_sanitize_identifier(v), ctx=_ctx_from_type(t))

        if t == "Str":
            return ast.Constant(value=_value(json_tree, idx, ""))

        if t == "Num":
            raw = _value(json_tree, idx, "0")
            try:
                if "." in raw:
                    return ast.Constant(value=float(raw))
                return ast.Constant(value=int(raw))
            except Exception:
                return ast.Constant(value=0)

        if t == "Constant":
            raw = _value(json_tree, idx, "")
            return ast.Constant(value=raw)

        if t in {"identifier", "vararg", "kwarg"}:
            return _sanitize_identifier(_value(json_tree, idx, "x"))

        if t == "attr":
            return _sanitize_identifier(_value(json_tree, idx, "attr"), "attr")

        if t == "arg":
            name = _sanitize_identifier(_value(json_tree, idx, "x"))
            return ast.arg(arg=name, annotation=None)

        # statements
        if t == "Expr":
            return ast.Expr(value=_safe_expr(build(ch[0])) if ch else ast.Constant(value=None))

        if t == "Print":
            values = [_safe_expr(build(c)) for c in ch] if ch else [ast.Constant(value="")]
            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="print", ctx=ast.Load()),
                    args=values,
                    keywords=[],
                )
            )

        if t == "Assign":
            if not ch:
                return ast.Assign(
                    targets=[ast.Name(id="x", ctx=ast.Store())],
                    value=ast.Constant(value=None),
                )
            targets = [_safe_store_expr(build(ch[0]))]
            value = _safe_expr(build(ch[1])) if len(ch) > 1 else ast.Constant(value=None)
            return ast.Assign(targets=targets, value=value)

        if t == "AnnAssign":
            target = _safe_store_expr(build(ch[0])) if len(ch) > 0 else ast.Name(id="x", ctx=ast.Store())
            annotation = _safe_expr(build(ch[1])) if len(ch) > 1 else ast.Name(id="Any", ctx=ast.Load())
            value = _safe_expr(build(ch[2])) if len(ch) > 2 else None
            return ast.AnnAssign(target=target, annotation=annotation, value=value, simple=1)

        if t == "AugAssign":
            target = _safe_store_expr(build(ch[0])) if len(ch) > 0 else ast.Name(id="x", ctx=ast.Store())
            value = _safe_expr(build(ch[1])) if len(ch) > 1 else ast.Constant(value=0)
            return ast.AugAssign(target=target, op=_op_from_type(t), value=value)

        if t == "Return":
            return ast.Return(value=_safe_expr(build(ch[0])) if ch else None)

        if t == "Pass":
            return ast.Pass()

        if t == "Break":
            return ast.Break()

        if t == "Continue":
            return ast.Continue()

        if t == "If":
            test = _safe_expr(build(ch[0])) if len(ch) > 0 else ast.Constant(value=True)
            body = _safe_stmt_list(build(ch[1])) if len(ch) > 1 else [ast.Pass()]
            orelse = _safe_stmt_list(build(ch[2])) if len(ch) > 2 else []
            return ast.If(test=test, body=body, orelse=orelse)

        if t == "While":
            test = _safe_expr(build(ch[0])) if len(ch) > 0 else ast.Constant(value=True)
            body = _safe_stmt_list(build(ch[1])) if len(ch) > 1 else [ast.Pass()]
            orelse = _safe_stmt_list(build(ch[2])) if len(ch) > 2 else []
            return ast.While(test=test, body=body, orelse=orelse)

        if t == "For":
            target = _safe_expr(build(ch[0])) if len(ch) > 0 else ast.Name(id="x", ctx=ast.Store())
            iter_ = _safe_expr(build(ch[1])) if len(ch) > 1 else ast.Name(id="xs", ctx=ast.Load())
            body = _safe_stmt_list(build(ch[2])) if len(ch) > 2 else [ast.Pass()]
            orelse = _safe_stmt_list(build(ch[3])) if len(ch) > 3 else []
            return ast.For(target=target, iter=iter_, body=body, orelse=orelse)

        if t == "AsyncFor":
            target = _safe_expr(build(ch[0])) if len(ch) > 0 else ast.Name(id="x", ctx=ast.Store())
            iter_ = _safe_expr(build(ch[1])) if len(ch) > 1 else ast.Name(id="xs", ctx=ast.Load())
            body = _safe_stmt_list(build(ch[2])) if len(ch) > 2 else [ast.Pass()]
            orelse = _safe_stmt_list(build(ch[3])) if len(ch) > 3 else []
            return ast.AsyncFor(target=target, iter=iter_, body=body, orelse=orelse)

        if t == "With":
            items = []
            body = [ast.Pass()]
            if ch:
                child_types = [json_tree[child_idx]["type"] for child_idx in ch]
                body_idx = ch[-1] if child_types and child_types[-1] == "body" else None
                if body_idx is not None:
                    body = _safe_stmt_list(build(body_idx))
                    item_children = ch[:-1]
                else:
                    item_children = ch

                i = 0
                while i < len(item_children):
                    context_expr = _safe_expr(build(item_children[i]))
                    optional_vars = None
                    if i + 1 < len(item_children):
                        next_type = json_tree[item_children[i + 1]]["type"]
                        if next_type not in {"body"}:
                            optional_vars = _safe_store_expr(build(item_children[i + 1]))
                            i += 1
                    items.append(ast.withitem(context_expr=context_expr, optional_vars=optional_vars))
                    i += 1
            return ast.With(items=items or [ast.withitem(context_expr=ast.Name(id="ctx", ctx=ast.Load()), optional_vars=None)], body=body)

        if t == "Try":
            body = _safe_stmt_list(build(ch[0])) if len(ch) > 0 else [ast.Pass()]
            handlers = build(ch[1]) if len(ch) > 1 else []
            orelse = _safe_stmt_list(build(ch[2])) if len(ch) > 2 else []
            finalbody = _safe_stmt_list(build(ch[3])) if len(ch) > 3 else []
            handlers = [h for h in handlers if isinstance(h, ast.ExceptHandler)] if isinstance(handlers, list) else []
            return ast.Try(body=body, handlers=handlers, orelse=orelse, finalbody=finalbody)

        if t == "Raise":
            exc = _safe_expr(build(ch[0])) if len(ch) > 0 else None
            cause = _safe_expr(build(ch[1])) if len(ch) > 1 else None
            return ast.Raise(exc=exc, cause=cause)

        if t == "ExceptHandler":
            ex_type = None
            name = None
            body = [ast.Pass()]

            for child_idx in ch:
                child = json_tree[child_idx]
                child_type = child["type"]
                built = build(child_idx)

                if child_type == "type":
                    if isinstance(built, list) and built:
                        ex_type = _safe_expr(built[0])
                elif child_type == "name":
                    if isinstance(built, list) and built:
                        first = built[0]
                        if isinstance(first, ast.Name):
                            name = first.id
                        elif isinstance(first, str):
                            name = first
                elif child_type == "body":
                    body = _safe_stmt_list(built)

            return ast.ExceptHandler(type=ex_type, name=name, body=body)

        # defs
        if t in {"FunctionDef", "AsyncFunctionDef"}:
            name = _sanitize_identifier(_value(json_tree, idx, "generated_function"), "generated_function")
            args_obj = ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            )
            body = [ast.Pass()]
            decorators = []
            returns = None

            for child_idx in ch:
                child = json_tree[child_idx]
                child_type = child["type"]
                built = build(child_idx)

                if child_type == "arguments":
                    args_obj = built
                elif child_type == "body":
                    body = _safe_stmt_list(built)
                elif child_type == "decorator_list":
                    decorators = _safe_expr_list(built)
                elif child_type == "returns":
                    if isinstance(built, list) and built:
                        returns = _safe_expr(built[0])

            if t == "AsyncFunctionDef":
                return ast.AsyncFunctionDef(
                    name=name,
                    args=args_obj,
                    body=body,
                    decorator_list=decorators,
                    returns=returns,
                    type_comment=None,
                )
            return ast.FunctionDef(
                name=name,
                args=args_obj,
                body=body,
                decorator_list=decorators,
                returns=returns,
                type_comment=None,
            )

        if t == "ClassDef":
            name = _sanitize_identifier(_value(json_tree, idx, "GeneratedClass"), "GeneratedClass")
            bases = []
            keywords = []
            body = [ast.Pass()]
            decorators = []

            for child_idx in ch:
                child = json_tree[child_idx]
                child_type = child["type"]
                built = build(child_idx)

                if child_type == "bases":
                    bases = _safe_expr_list(built)
                elif child_type == "keywords":
                    if isinstance(built, list):
                        keywords = [x for x in built if isinstance(x, ast.keyword)]
                elif child_type == "body":
                    body = _safe_stmt_list(built)
                elif child_type == "decorator_list":
                    decorators = _safe_expr_list(built)

            return ast.ClassDef(
                name=name,
                bases=bases,
                keywords=keywords,
                body=body,
                decorator_list=decorators,
            )

        if t == "arguments":
            posonlyargs = []
            args = []
            kwonlyargs = []
            defaults = []
            kw_defaults = []
            vararg = None
            kwarg = None

            for child_idx in ch:
                child = json_tree[child_idx]
                child_type = child["type"]

                if child_type == "posonlyargs":
                    for arg_idx in _children(json_tree, child_idx):
                        arg_node = json_tree[arg_idx]
                        arg_name = _sanitize_identifier(arg_node.get("value", "x"))
                        posonlyargs.append(ast.arg(arg=arg_name, annotation=None))
                elif child_type == "args":
                    for arg_idx in _children(json_tree, child_idx):
                        arg_node = json_tree[arg_idx]
                        arg_name = _sanitize_identifier(arg_node.get("value", "x"))
                        args.append(ast.arg(arg=arg_name, annotation=None))
                elif child_type == "kwonlyargs":
                    for arg_idx in _children(json_tree, child_idx):
                        arg_node = json_tree[arg_idx]
                        arg_name = _sanitize_identifier(arg_node.get("value", "x"))
                        kwonlyargs.append(ast.arg(arg=arg_name, annotation=None))
                elif child_type == "defaults":
                    defaults = [_safe_expr(build(x)) for x in _children(json_tree, child_idx)]
                elif child_type == "kw_defaults":
                    raw = [build(x) for x in _children(json_tree, child_idx)]
                    kw_defaults = [None if x is None else _safe_expr(x) for x in raw]
                elif child_type == "vararg":
                    vararg = ast.arg(arg=_sanitize_identifier(_value(json_tree, child_idx, "args"), "args"), annotation=None)
                elif child_type == "kwarg":
                    kwarg = ast.arg(arg=_sanitize_identifier(_value(json_tree, child_idx, "kwargs"), "kwargs"), annotation=None)

            return ast.arguments(
                posonlyargs=posonlyargs,
                args=args,
                vararg=vararg,
                kwonlyargs=kwonlyargs,
                kw_defaults=kw_defaults,
                kwarg=kwarg,
                defaults=defaults,
            )

        # expressions
        if t == "Call":
            func = _safe_expr(build(ch[0])) if len(ch) > 0 else ast.Name(id="f", ctx=ast.Load())
            args = []
            keywords = []
            for child_idx in ch[1:]:
                built = build(child_idx)
                if isinstance(built, ast.keyword):
                    keywords.append(built)
                else:
                    args.append(_safe_expr(built))
            return ast.Call(func=func, args=args, keywords=keywords)

        if t == "FormattedValue":
            value = _safe_expr(build(ch[0])) if ch else ast.Constant(value="")
            return ast.FormattedValue(value=value, conversion=-1, format_spec=None)

        if t == "JoinedStr":
            values = []
            for child_idx in ch:
                built = build(child_idx)
                if isinstance(built, ast.FormattedValue):
                    values.append(built)
                elif isinstance(built, ast.Constant) and isinstance(built.value, str):
                    values.append(built)
                elif isinstance(built, str):
                    values.append(ast.Constant(value=built))
                else:
                    values.append(ast.FormattedValue(value=_safe_expr(built), conversion=-1, format_spec=None))
            return ast.JoinedStr(values=values or [ast.Constant(value="")])

        if t == "keyword":
            return ast.keyword(
                arg=_sanitize_identifier(_value(json_tree, idx, "arg"), "arg"),
                value=_safe_expr(build(ch[0])) if ch else ast.Constant(value=None),
            )

        if t.startswith("Attribute"):
            if not ch:
                return ast.Attribute(
                    value=ast.Name(id="obj", ctx=ast.Load()),
                    attr="attr",
                    ctx=_ctx_from_type(t),
                )
            attr_name = "attr"
            value_expr = _safe_expr(build(ch[0]))

            if len(ch) > 1:
                maybe_attr = build(ch[1])
                if isinstance(maybe_attr, str):
                    attr_name = _sanitize_identifier(maybe_attr, "attr")

            return ast.Attribute(
                value=value_expr,
                attr=attr_name,
                ctx=_ctx_from_type(t),
            )

        if t.startswith("BinOp"):
            left = _safe_expr(build(ch[0])) if len(ch) > 0 else ast.Constant(value=0)
            right = _safe_expr(build(ch[1])) if len(ch) > 1 else ast.Constant(value=0)
            return ast.BinOp(left=left, op=_op_from_type(t), right=right)

        if t.startswith("BoolOp"):
            values = [_safe_expr(build(c)) for c in ch] if ch else [ast.Constant(value=True), ast.Constant(value=False)]
            return ast.BoolOp(op=_boolop_from_type(t), values=values)

        if t.startswith("UnaryOp"):
            operand = _safe_expr(build(ch[0])) if ch else ast.Constant(value=0)
            return ast.UnaryOp(op=_unaryop_from_type(t), operand=operand)

        if t.startswith("Compare"):
            left = _safe_expr(build(ch[0])) if len(ch) > 0 else ast.Constant(value=0)
            comparators = [_safe_expr(build(c)) for c in ch[1:]] if len(ch) > 1 else [ast.Constant(value=0)]
            ops = _cmp_ops_from_type(t)
            while len(ops) < len(comparators):
                ops.append(ast.Eq())
            return ast.Compare(left=left, ops=ops[:len(comparators)], comparators=comparators)

        if t == "List":
            return ast.List(elts=[_safe_expr(build(c)) for c in ch], ctx=ast.Load())

        if t == "Tuple":
            return ast.Tuple(elts=[_safe_expr(build(c)) for c in ch], ctx=ast.Load())

        if t == "Set":
            return ast.Set(elts=[_safe_expr(build(c)) for c in ch])

        if t == "Dict":
            keys = []
            values = []
            for i, child_idx in enumerate(ch):
                if i % 2 == 0:
                    keys.append(_safe_expr(build(child_idx)))
                else:
                    values.append(_safe_expr(build(child_idx)))
            return ast.Dict(keys=keys, values=values)

        if t == "alias":
            name = _value(json_tree, idx, "module")
            asname = None
            if ch:
                maybe_asname = build(ch[0])
                if isinstance(maybe_asname, str):
                    asname = maybe_asname
            return ast.alias(name=name, asname=asname)

        if t == "Import":
            names = [build(c) for c in ch] if ch else [ast.alias(name="os", asname=None)]
            names = [x for x in names if isinstance(x, ast.alias)] or [ast.alias(name="os", asname=None)]
            return ast.Import(names=names)

        if t == "ImportFrom":
            module = _value(json_tree, idx, "")
            names = [build(c) for c in ch] if ch else [ast.alias(name="x", asname=None)]
            names = [x for x in names if isinstance(x, ast.alias)] or [ast.alias(name="x", asname=None)]
            return ast.ImportFrom(module=module or None, names=names, level=0)

        if t == "Subscript":
            value = _safe_expr(build(ch[0])) if len(ch) > 0 else ast.Name(id="x", ctx=ast.Load())
            slice_ = _safe_expr(build(ch[1])) if len(ch) > 1 else ast.Constant(value=0)
            return ast.Subscript(value=value, slice=slice_, ctx=ast.Load())

        if t == "Slice":
            lower = _safe_expr(build(ch[0])) if len(ch) > 0 else None
            upper = _safe_expr(build(ch[1])) if len(ch) > 1 else None
            step = _safe_expr(build(ch[2])) if len(ch) > 2 else None
            return ast.Slice(lower=lower, upper=upper, step=step)

        # fallback
        return ast.Expr(value=ast.Constant(value=f"UNSUPPORTED_NODE:{t}"))

    if not json_tree:
        return ast.Module(body=[ast.Pass()], type_ignores=[])

    module = build(0)
    if not isinstance(module, ast.Module):
        module = ast.Module(body=[_safe_stmt(module)], type_ignores=[])

    return ast.fix_missing_locations(module)


############################################################
# TOKENS -> CODE
############################################################

def ast_tokens_to_code(tokens: List[str]) -> str:
    """
    generated tokens -> json_tree -> ast -> python code
    """
    json_tree = decode_linearized_tokens_to_json_tree(tokens)
    py_ast = json_tree_to_python_ast(json_tree)
    py_ast = ast.fix_missing_locations(py_ast)

    try:
        return ast.unparse(py_ast)
    except Exception as e:
        raise RuntimeError(f"Failed to unparse generated AST: {e}") from e


def find_path_to_node(nodes: List[dict], target_idx: int) -> Optional[List[int]]:
    if target_idx < 0 or target_idx >= len(nodes) or not nodes:
        return None

    def dfs(idx: int, path: List[int], visited: set[int]) -> Optional[List[int]]:
        if idx == target_idx:
            return path
        if idx in visited:
            return None

        visited.add(idx)
        for child_pos, child_idx in enumerate(nodes[idx].get("children", [])):
            result = dfs(child_idx, path + [child_pos], visited)
            if result is not None:
                return result
        return None

    return dfs(0, [], set())


def follow_child_path(nodes: List[dict], path: List[int]) -> Optional[int]:
    if not nodes:
        return None

    idx = 0
    for child_pos in path:
        children = nodes[idx].get("children", [])
        if child_pos < 0 or child_pos >= len(children):
            return None
        idx = children[child_pos]
    return idx


def clone_json_subtree(nodes: List[dict], root_idx: int) -> List[dict]:
    if root_idx < 0 or root_idx >= len(nodes):
        return []

    cloned: List[dict] = []
    remap: Dict[int, int] = {}

    def dfs(idx: int) -> int:
        if idx in remap:
            return remap[idx]

        new_idx = len(cloned)
        remap[idx] = new_idx
        node = dict(nodes[idx])
        node.pop("children", None)
        cloned.append(node)

        children = nodes[idx].get("children", [])
        if children:
            cloned[new_idx]["children"] = [dfs(child_idx) for child_idx in children]

        return new_idx

    dfs(root_idx)
    return cloned


def json_subtree_to_code(nodes: List[dict], root_idx: int) -> str:
    node = _node(nodes, root_idx)
    if node is None:
        return ""

    if node.get("type") == "attr":
        return _sanitize_identifier(node.get("value", "attr"), "attr")

    subtree = clone_json_subtree(nodes, root_idx)
    if not subtree:
        return ""

    py_ast = json_tree_to_python_ast(subtree)
    py_ast = ast.fix_missing_locations(py_ast)
    if (
        isinstance(py_ast, ast.Module)
        and len(py_ast.body) == 1
        and isinstance(py_ast.body[0], ast.Expr)
    ):
        expr_ast = ast.Expression(body=py_ast.body[0].value)
        expr_ast = ast.fix_missing_locations(expr_ast)
        return ast.unparse(expr_ast).strip()
    return ast.unparse(py_ast).strip()


def extract_completion_from_hole_context(
    original_tree: List[dict],
    completed_tree: List[dict],
    hole_idx: int,
    hole_kind: str,
    original_code: str,
) -> str:
    """
    Extract the generated fragment by locating the same hole position inside the
    completed tree, instead of decoding `new_tokens` without context.
    """
    path = find_path_to_node(original_tree, hole_idx)
    if path is None:
        return ""

    completed_idx = follow_child_path(completed_tree, path)
    if completed_idx is None:
        return ""

    fragment = json_subtree_to_code(completed_tree, completed_idx)
    if not fragment:
        return ""

    closers = "".join(_find_unmatched_closers(original_code))

    if hole_kind in {"inline_expr", "call_args"}:
        return fragment + closers

    if hole_kind == "attr":
        return fragment

    if hole_kind == "fstring_expr":
        stripped = original_code.rstrip()
        fstring_match = re.search(r"f(['\"]).*\{\s*$", stripped)
        quote = fstring_match.group(1) if fstring_match else ""
        return fragment + "}" + quote

    return fragment


def build_full_code_from_completion(
    original_code: str,
    completion_text: str,
    full_generated_code: str,
) -> str:
    """
    Preserve the user's original prefix when we know the generated fragment that
    should be appended at the cursor position.
    """
    if completion_text:
        return original_code + completion_text
    return full_generated_code


def extract_completion_text(original_code: str, full_generated_code: str) -> str:
    if full_generated_code.startswith(original_code):
        return full_generated_code[len(original_code):]

    original_lines = original_code.splitlines()
    generated_lines = full_generated_code.splitlines()

    common_prefix_lines = 0
    for orig_line, gen_line in zip(original_lines, generated_lines):
        if orig_line == gen_line:
            common_prefix_lines += 1
        else:
            break

    if common_prefix_lines > 0:
        tail_lines = generated_lines[common_prefix_lines:]
        return "\n".join(tail_lines)

    return full_generated_code


def _find_unmatched_closers(code: str) -> List[str]:
    pair_map = {"(": ")", "[": "]", "{": "}"}
    closing_chars = {")", "]", "}"}
    stack = []
    in_single = False
    in_double = False
    escaped = False
    for ch in code:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch in pair_map:
            stack.append(pair_map[ch])
        elif ch in closing_chars and stack and ch == stack[-1]:
            stack.pop()
    closers = []
    while stack:
        closers.append(stack.pop())
    return closers


def prepare_tolerant_prefix(code: str) -> Optional[Dict[str, str]]:
    """
    Build a parseable surrogate program with an explicit hole marker that marks
    the exact completion position.
    """
    stripped = code.rstrip()
    if not stripped:
        return {
            "repaired_code": TOLERANT_STMT_HOLE,
            "hole_value": TOLERANT_STMT_HOLE,
            "hole_kind": "stmt_block",
        }

    lines = stripped.splitlines()
    last_line = lines[-1]
    base_indent = len(last_line) - len(last_line.lstrip(" "))
    child_indent = " " * (base_indent + 4)
    closers = _find_unmatched_closers(stripped)

    def expr_hole_suffix(needs_space: bool = False) -> str:
        return (" " if needs_space else "") + TOLERANT_EXPR_HOLE + "".join(closers)

    fstring_match = re.search(r"f(['\"]).*\{\s*$", stripped)
    if fstring_match:
        quote = fstring_match.group(1)
        return {
            "repaired_code": stripped + TOLERANT_EXPR_HOLE + "}" + quote,
            "hole_value": TOLERANT_EXPR_HOLE,
            "hole_kind": "fstring_expr",
        }

    if re.search(r"\bexcept\s+$", code):
        return {
            "repaired_code": stripped + " " + TOLERANT_EXPR_HOLE + ":\n" + child_indent + "pass",
            "hole_value": TOLERANT_EXPR_HOLE,
            "hole_kind": "except_clause",
        }

    keyword_expr_patterns = [
        r"\breturn\s+$",
        r"\byield\s+$",
        r"\braise\s+$",
        r"\bassert\s+$",
        r"\bif\s+$",
        r"\bin\s+$",
    ]
    if any(re.search(pattern, code) for pattern in keyword_expr_patterns):
        return {
            "repaired_code": stripped + expr_hole_suffix(needs_space=True),
            "hole_value": TOLERANT_EXPR_HOLE,
            "hole_kind": "inline_expr",
        }

    operator_expr_patterns = [
        r"=\s+$",
        r"\+\s+$",
        r"-\s+$",
        r"\*\s+$",
        r"/\s+$",
        r"%\s+$",
        r"\*\*\s+$",
        r"//\s+$",
        r"and\s+$",
        r"or\s+$",
        r",\s+$",
        r":\s+$",
    ]
    if any(re.search(pattern, code) for pattern in operator_expr_patterns):
        return {
            "repaired_code": stripped + expr_hole_suffix(needs_space=False),
            "hole_value": TOLERANT_EXPR_HOLE,
            "hole_kind": "inline_expr",
        }

    if stripped.endswith(":") and not closers:
        return {
            "repaired_code": stripped + "\n" + child_indent + TOLERANT_STMT_HOLE,
            "hole_value": TOLERANT_STMT_HOLE,
            "hole_kind": "stmt_block",
        }

    if re.search(r"\.\s*$", stripped):
        return {
            "repaired_code": stripped + TOLERANT_ATTR_HOLE,
            "hole_value": TOLERANT_ATTR_HOLE,
            "hole_kind": "attr",
        }

    if re.search(r"\(\s*$", stripped):
        return {
            "repaired_code": stripped + TOLERANT_EXPR_HOLE + "".join(closers),
            "hole_value": TOLERANT_EXPR_HOLE,
            "hole_kind": "call_args",
        }

    if closers:
        return {
            "repaired_code": stripped + expr_hole_suffix(needs_space=False),
            "hole_value": TOLERANT_EXPR_HOLE,
            "hole_kind": "inline_expr",
        }

    return None


def make_prefix_parseable(code: str) -> str:
    prepared = prepare_tolerant_prefix(code)
    if prepared is not None:
        return prepared["repaired_code"]
    return code


def find_tolerant_hole_node_idx(json_tree: List[dict], hole_value: str, hole_kind: str) -> Optional[int]:
    if hole_kind == "stmt_block":
        for idx, node in enumerate(json_tree):
            children = node.get("children", [])
            if node.get("type") == "Expr" and len(children) == 1:
                child = json_tree[children[0]]
                if child.get("value") == hole_value:
                    return idx

    for idx, node in enumerate(json_tree):
        if node.get("value") == hole_value:
            return idx

    return None


def linearize_prefix_before_node(
    nodes: List[dict],
    tokenizer,
    stop_idx: int,
) -> List[str]:
    visited = set()
    out: List[str] = []

    def dfs(idx: int) -> bool:
        if idx < 0 or idx >= len(nodes):
            return False
        if idx == stop_idx:
            return True
        if idx in visited:
            out.append("REF")
            return False

        visited.add(idx)
        node = nodes[idx]
        node_type = node.get("type", "UNKNOWN")
        children = node.get("children", [])

        out.append(f"ENTER_{node_type}")
        out.extend(tokenizer._value_tokens(node))

        if children:
            out.append(f"ARITY_{min(len(children), 16)}")
            for child_idx in children:
                if dfs(child_idx):
                    return True
        else:
            out.append("LEAF")

        out.append(f"EXIT_{node_type}")
        return False

    stopped = dfs(0)
    if not stopped:
        for idx in range(len(nodes)):
            if idx not in visited:
                out.append("EXTRA_ROOT")
                if dfs(idx):
                    break

    return out


def choose_inference_prefix_tokens(tokens: List[str], cfg, prefix_len: int) -> List[str]:
    """
    Keep a meaningful suffix for completion during inference.
    If the requested prefix would consume the whole sample, shorten it
    so the model still has something non-trivial left to predict.
    """
    if not tokens:
        return []

    total_tokens = len(tokens)
    if prefix_len <= 0:
        requested_prefix = total_tokens
    else:
        requested_prefix = min(prefix_len, total_tokens)

    if requested_prefix < total_tokens:
        return tokens[:requested_prefix]

    reserve_tokens = min(
        getattr(cfg, "infer_min_completion_tokens", 16),
        max(1, total_tokens - 1),
    )

    if total_tokens > 8:
        reserve_tokens = max(reserve_tokens, total_tokens // 3)

    effective_prefix_len = max(1, total_tokens - reserve_tokens)
    return tokens[:effective_prefix_len]


############################################################
# CODE -> CODE
############################################################

@torch.no_grad()
def continue_real_code(
    model,
    code: str,
    vocab,
    ivocab,
    tokenizer,
    cfg,
    prefix_len: int = 256,
    allow_incomplete_prefix: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
):
    """
    Main code-to-code path:
      python code -> AST tokens -> LM -> generated AST tokens -> python code
    """
    device = next(model.parameters()).device

    source_for_parse = code
    tolerant_info = prepare_tolerant_prefix(code) if allow_incomplete_prefix else None
    parsed = None
    if allow_incomplete_prefix:
        if tolerant_info is not None:
            source_for_parse = tolerant_info["repaired_code"] if tolerant_info is not None else make_prefix_parseable(code)
            parsed = tokenize_and_parse_code(source_for_parse)
        else:
            try:
                parsed = tokenize_and_parse_code(source_for_parse)
            except (SyntaxError, tokenize.TokenError):
                tolerant_info = prepare_tolerant_prefix(code)
                source_for_parse = tolerant_info["repaired_code"] if tolerant_info is not None else make_prefix_parseable(code)
                parsed = tokenize_and_parse_code(source_for_parse)
    else:
        parsed = tokenize_and_parse_code(source_for_parse)

    json_tree = parse_code_to_json_tree(source_for_parse, parsed=parsed)
    python_tokens = parsed["python_tokens"]
    normalized_source = parsed["normalized_source"]

    tokens = tokenizer.linearize(json_tree)
    hole_idx = None
    if tolerant_info is not None:
        hole_idx = find_tolerant_hole_node_idx(
            json_tree,
            hole_value=tolerant_info["hole_value"],
            hole_kind=tolerant_info["hole_kind"],
        )
        if hole_idx is not None:
            prefix_tokens = linearize_prefix_before_node(json_tree, tokenizer, hole_idx)
            if prefix_len > 0 and len(prefix_tokens) > prefix_len:
                prefix_tokens = prefix_tokens[-prefix_len:]
        else:
            prefix_tokens = choose_inference_prefix_tokens(tokens, cfg, prefix_len)
    else:
        prefix_tokens = choose_inference_prefix_tokens(tokens, cfg, prefix_len)

    start_tokens = [PROMPT_TOKEN] + prefix_tokens + [COMPLETION_TOKEN]

    generated_tokens = generate_tokens(
        model=model,
        start_tokens=start_tokens,
        vocab=vocab,
        ivocab=ivocab,
        device=device,
        cfg=cfg,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_on_ast_completion=True,
        initial_ast_depth=compute_ast_depth(prefix_tokens),
        min_ast_tokens_before_eos=getattr(cfg, "infer_min_ast_tokens", 0),
    )

    generated_body_tokens = generated_tokens[1:] if generated_tokens and generated_tokens[0] == "<BOS>" else generated_tokens
    new_tokens = [
        tok
        for tok in generated_body_tokens[len(start_tokens):]
        if tok not in {"<EOS>", "<PAD>", "<UNK>", "<BOS>", "<PROMPT>", "<COMPLETION>"}
    ]
    completed_tokens = prefix_tokens + new_tokens

    raw_generated_code = ast_tokens_to_code(completed_tokens)
    completed_tree = decode_linearized_tokens_to_json_tree(completed_tokens)
    if tolerant_info is not None:
        completion_text = ""
        if hole_idx is not None:
            completion_text = extract_completion_from_hole_context(
                original_tree=json_tree,
                completed_tree=completed_tree,
                hole_idx=hole_idx,
                hole_kind=tolerant_info["hole_kind"],
                original_code=code,
            )
        if not completion_text:
            completion_text = extract_completion_text(code, raw_generated_code)
        full_generated_code = build_full_code_from_completion(
            original_code=code,
            completion_text=completion_text,
            full_generated_code=raw_generated_code,
        )
    else:
        full_generated_code = raw_generated_code
        completion_text = extract_completion_text(code, full_generated_code)

    return {
        "input_code": code,
        "source_for_parse": source_for_parse,
        "normalized_source": normalized_source,
        "tolerant_info": tolerant_info,
        "python_tokens": python_tokens,
        "json_tree": json_tree,
        "original_tokens": tokens,
        "prefix_tokens": prefix_tokens,
        "prompt_tokens": start_tokens,
        "generated_tokens": generated_tokens,
        "new_tokens": new_tokens,
        "completed_tokens": completed_tokens,
        "completed_tree": completed_tree,
        "full_generated_code": full_generated_code,
        "generated_code": completion_text,
        "generated_completion_text": completion_text,
    }


@torch.no_grad()
def continue_real_code_safe(
    model,
    code: str,
    vocab,
    ivocab,
    tokenizer,
    cfg,
    prefix_len: int = 256,
    allow_incomplete_prefix: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    fallback_to_original: bool = False,
):
    """
    Safe wrapper: never crashes outward.
    """
    try:
        return continue_real_code(
            model=model,
            code=code,
            vocab=vocab,
            ivocab=ivocab,
            tokenizer=tokenizer,
            cfg=cfg,
            prefix_len=prefix_len,
            allow_incomplete_prefix=allow_incomplete_prefix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    except Exception as e:
        return {
            "input_code": code,
            "source_for_parse": code,
            "normalized_source": None,
            "python_tokens": None,
            "json_tree": None,
            "original_tokens": None,
            "prefix_tokens": None,
            "prompt_tokens": None,
            "generated_tokens": None,
            "new_tokens": None,
            "completed_tokens": None,
            "completed_tree": None,
            "full_generated_code": code if fallback_to_original else "",
            "generated_code": code if fallback_to_original else "",
            "generated_completion_text": code if fallback_to_original else "",
            "error": str(e),
        }


if __name__ == "__main__":
    print("parse_python.py is intended to be imported from MLCode.py")
