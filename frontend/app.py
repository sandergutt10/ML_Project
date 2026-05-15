from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import MLCode
from parse_python import continue_real_code_safe
from tests import SAMPLE_GROUPS, build_tokenizer, set_if_present


CHECKPOINT_PATH = PROJECT_DIR / "checkpoints_code_lm" / "best.pt"

STARTER_SNIPPETS = {
    "Функция и print": "def greet(name):\n    message = f'Hello, {name}'\n    print(",
    "Словарь и get": "data = {'a': 1, 'b': 2}\nvalue = data.get(",
    "Цикл и append": "def square_all(xs):\n    out = []\n    for x in xs:\n        out.append(",
    "Работа со строкой": "text = 'hello world'\nparts = text.split(",
    "Условие и return": "def is_even(x):\n    if x % 2 == 0:\n        return ",
}


def flatten_samples() -> dict[str, str]:
    options: dict[str, str] = {}
    for group_name, samples in SAMPLE_GROUPS.items():
        for sample_name, code in samples:
            options[f"{group_name} -> {sample_name}"] = code
    return options


SAMPLE_OPTIONS = flatten_samples()

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_runtime():
    device = "cuda" if MLCode.torch.cuda.is_available() else "cpu"
    model, vocab, ivocab, cfg = MLCode.load_model_for_inference(
        checkpoint_path=str(CHECKPOINT_PATH),
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
    return model, vocab, ivocab, tokenizer, cfg, device


def run_completion(code: str) -> dict:
    model, vocab, ivocab, tokenizer, cfg, _device = load_runtime()
    return continue_real_code_safe(
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


def runtime_meta() -> dict[str, object]:
    if not CHECKPOINT_PATH.exists():
        return {
            "checkpoint": CHECKPOINT_PATH.name,
            "checkpoint_exists": False,
        }

    _model, _vocab, _ivocab, _tokenizer, cfg, device = load_runtime()
    return {
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_exists": True,
        "device": device,
        "infer_prefix_len": getattr(cfg, "infer_prefix_len", 256),
        "infer_max_new_tokens": getattr(cfg, "infer_max_new_tokens", 96),
        "infer_temperature": getattr(cfg, "infer_temperature", 0.2),
        "infer_top_k": getattr(cfg, "infer_top_k", 8),
        "infer_top_p": getattr(cfg, "infer_top_p", 0.8),
        "infer_repetition_penalty": getattr(cfg, "infer_repetition_penalty", 1.02),
    }


PAGE_TEMPLATE = """
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AST Code Completion Studio</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f3efe7;
            --bg-2: #efe7da;
            --surface: #fffdfa;
            --surface-2: #fbf6ee;
            --surface-3: #f6ede1;
            --text: #1f1712;
            --text-soft: #554338;
            --muted: #6e5a4e;
            --line: #d8c8b8;
            --line-strong: #baa28f;
            --accent: #b7552b;
            --accent-strong: #94401d;
            --accent-soft: #fff0e7;
            --accent-ink: #6e2f17;
            --success: #175f52;
            --success-soft: #e7f6f2;
            --danger: #8b2f2f;
            --danger-soft: #fdeeee;
            --shadow: 0 20px 50px rgba(62, 39, 25, 0.10);
            --radius-xl: 30px;
            --radius-lg: 22px;
            --radius-md: 16px;
            --radius-sm: 12px;
            --mono: "JetBrains Mono", "Fira Code", Consolas, monospace;
            --sans: "Segoe UI", Inter, Arial, sans-serif;
        }

        * {
            box-sizing: border-box;
        }

        html, body {
            margin: 0;
            min-height: 100%;
            background:
                radial-gradient(circle at top left, rgba(183, 85, 43, 0.15), transparent 28%),
                radial-gradient(circle at 85% 10%, rgba(23, 95, 82, 0.11), transparent 20%),
                linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
            color: var(--text);
            font-family: var(--sans);
        }

        body {
            padding: 28px;
        }

        .shell {
            max-width: 1380px;
            margin: 0 auto;
        }

        .hero {
            margin-bottom: 18px;
        }

        .hero-card,
        .panel,
        .stats-card,
        .result-panel,
        .notice {
            border: 1px solid var(--line);
            background: rgba(255, 253, 250, 0.92);
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }

        .hero-card {
            border-radius: var(--radius-xl);
            padding: 30px;
            position: relative;
            overflow: hidden;
        }

        .hero-card::after {
            content: "";
            position: absolute;
            right: -70px;
            bottom: -110px;
            width: 280px;
            height: 280px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(183, 85, 43, 0.16), transparent 65%);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 9px 14px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-ink);
            border: 1px solid #efcfbe;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }

        h1 {
            margin: 16px 0 0 0;
            max-width: 820px;
            font-size: clamp(2.3rem, 4vw, 4rem);
            line-height: 0.98;
            letter-spacing: -0.045em;
        }

        .hero-text {
            margin: 18px 0 0 0;
            max-width: 760px;
            color: var(--text-soft);
            font-size: 1.03rem;
            line-height: 1.75;
        }

        .hero-side {
            border-radius: var(--radius-xl);
            padding: 24px;
        }

        .panel h2,
        .result-header h2 {
            margin: 0 0 8px 0;
            font-size: 1.2rem;
            line-height: 1.2;
        }

        .panel p,
        .result-header p {
            margin: 0;
            color: var(--text-soft);
            line-height: 1.7;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 18px;
        }

        .stats-card {
            border-radius: 24px;
            padding: 18px;
        }

        .stats-label {
            font-size: 0.77rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            font-weight: 800;
            margin-bottom: 10px;
        }

        .stats-value {
            font-size: 1.08rem;
            font-weight: 800;
            word-break: break-word;
        }

        .workspace {
            display: grid;
            grid-template-columns: 1.4fr 0.88fr;
            gap: 18px;
            align-items: start;
            margin-bottom: 18px;
        }

        .panel {
            border-radius: 28px;
            padding: 22px;
        }

        .panel-heading {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: start;
            margin-bottom: 16px;
        }

        .panel-kicker {
            font-size: 0.77rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: var(--accent-strong);
            font-weight: 800;
        }

        .field-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-bottom: 12px;
        }

        label {
            display: block;
            margin-bottom: 7px;
            font-size: 0.92rem;
            font-weight: 700;
            color: var(--text);
        }

        select,
        textarea,
        button,
        summary {
            font: inherit;
        }

        select,
        textarea {
            width: 100%;
            border-radius: var(--radius-md);
            border: 1px solid var(--line-strong);
            background: var(--surface);
            color: var(--text);
            transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
        }

        select:focus,
        textarea:focus,
        button:focus-visible,
        summary:focus-visible,
        .tab-button:focus-visible {
            outline: 3px solid rgba(183, 85, 43, 0.24);
            outline-offset: 2px;
        }

        select {
            min-height: 48px;
            padding: 12px 14px;
        }

        textarea {
            min-height: 420px;
            resize: vertical;
            padding: 18px;
            font-family: var(--mono);
            font-size: 14px;
            line-height: 1.65;
            tab-size: 4;
            white-space: pre;
            overflow-wrap: normal;
            background: #fffdfa;
        }

        textarea::placeholder {
            color: #8d7a70;
        }

        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 14px 0 0 0;
        }

        button {
            border: 0;
            border-radius: 16px;
            min-height: 48px;
            padding: 12px 18px;
            cursor: pointer;
            font-weight: 800;
            transition: transform 0.16s ease, filter 0.16s ease, opacity 0.16s ease;
        }

        button:hover {
            transform: translateY(-1px);
            filter: brightness(1.02);
        }

        button:disabled {
            cursor: wait;
            opacity: 0.75;
            transform: none;
        }

        .button-primary {
            background: linear-gradient(135deg, var(--accent), #dc8b54);
            color: #fffaf7;
            box-shadow: 0 16px 28px rgba(183, 85, 43, 0.20);
        }

        .button-secondary {
            background: var(--surface-3);
            color: var(--text);
            border: 1px solid var(--line);
        }

        .button-ghost {
            background: transparent;
            color: var(--accent-strong);
            border: 1px solid #ddc8ba;
        }

        .tip-list {
            margin: 0;
            padding-left: 18px;
            color: var(--text-soft);
            line-height: 1.65;
        }

        .tip-list li + li {
            margin-top: 8px;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 18px;
        }

        .chip {
            padding: 9px 12px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent-ink);
            border: 1px solid #efcfbe;
            font-size: 0.84rem;
            font-weight: 800;
        }

        details {
            margin-top: 18px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--surface-2);
            overflow: hidden;
        }

        summary {
            list-style: none;
            cursor: pointer;
            padding: 14px 16px;
            font-weight: 800;
        }

        summary::-webkit-details-marker {
            display: none;
        }

        .details-body {
            padding: 0 16px 16px 16px;
        }

        .json-box,
        .code-box {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: #fffdfa;
            overflow: auto;
        }

        .json-box {
            padding: 16px;
            font-family: var(--mono);
            font-size: 13px;
            line-height: 1.6;
            color: var(--text);
            white-space: pre-wrap;
        }

        .result-panel {
            border-radius: 28px;
            padding: 22px;
        }

        .result-header {
            margin-bottom: 14px;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 16px;
        }

        .status-pill {
            border-radius: 999px;
            padding: 9px 13px;
            font-size: 0.84rem;
            font-weight: 800;
            border: 1px solid transparent;
        }

        .status-neutral {
            background: var(--surface-3);
            border-color: #e2d4c7;
            color: var(--text);
        }

        .status-success {
            background: var(--success-soft);
            border-color: #c0e2d9;
            color: var(--success);
        }

        .status-error {
            background: var(--danger-soft);
            border-color: #f1c3c3;
            color: var(--danger);
        }

        .tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 14px;
        }

        .tab-button {
            background: var(--surface-2);
            color: var(--text);
            border: 1px solid var(--line);
            border-radius: 14px;
            min-height: 44px;
            padding: 10px 14px;
            font-weight: 800;
            cursor: pointer;
        }

        .tab-button.active {
            background: var(--accent-soft);
            color: var(--accent-ink);
            border-color: #ebc5b0;
        }

        .tab-pane {
            display: none;
        }

        .tab-pane.active {
            display: block;
        }

        .code-box pre {
            margin: 0;
            padding: 18px;
            font-family: var(--mono);
            font-size: 13px;
            line-height: 1.7;
            color: var(--text);
            white-space: pre-wrap;
            word-break: break-word;
        }

        .empty-state,
        .notice {
            border-radius: 24px;
            padding: 22px;
        }

        .empty-state {
            border: 1px dashed var(--line-strong);
            background: rgba(255, 251, 245, 0.82);
            color: var(--text-soft);
            line-height: 1.75;
        }

        .notice-error {
            background: var(--danger-soft);
            border: 1px solid #efc7c7;
            color: #662525;
        }

        .notice-error strong {
            color: #4f1c1c;
        }

        .result-tools {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }

        .mono-inline {
            font-family: var(--mono);
            font-size: 0.95em;
        }

        @media (max-width: 1100px) {
            body {
                padding: 18px;
            }

            .workspace {
                grid-template-columns: 1fr;
            }

            .stats {
                grid-template-columns: 1fr 1fr;
            }
        }

        @media (max-width: 720px) {
            .stats,
            .field-grid {
                grid-template-columns: 1fr;
            }

            .hero-card,
            .hero-side,
            .panel,
            .result-panel {
                padding: 18px;
            }

            textarea {
                min-height: 320px;
            }
        }
    </style>
</head>
<body>
    <div class="shell">
        <section class="hero">
            <div class="hero-card">
                <div class="eyebrow">Code Completion Studio</div>
                <h1>Интерфейс для продолжения Python-кода</h1>
                <p class="hero-text">
                    Здесь можно писать свой незавершённый фрагмент, подставлять готовые кейсы
                    и сразу смотреть, как модель продолжает код и собирает итоговый результат.
                </p>
            </div>
        </section>

        {% if not runtime.checkpoint_exists %}
        <section class="notice notice-error">
            <strong>Checkpoint не найден.</strong>
            Для работы фронтенда добавьте файл
            <span class="mono-inline">checkpoints_code_lm/best.pt</span>,
            затем перезапустите сервер.
        </section>
        {% endif %}

        <section class="stats">
            <div class="stats-card">
                <div class="stats-label">Checkpoint</div>
                <div class="stats-value">{{ runtime.checkpoint }}</div>
            </div>
            <div class="stats-card">
                <div class="stats-label">Устройство</div>
                <div class="stats-value">{{ runtime.device or "n/a" }}</div>
            </div>
            <div class="stats-card">
                <div class="stats-label">Длина префикса</div>
                <div class="stats-value">{{ runtime.infer_prefix_len or "n/a" }}</div>
            </div>
            <div class="stats-card">
                <div class="stats-label">Новых токенов максимум</div>
                <div class="stats-value">{{ runtime.infer_max_new_tokens or "n/a" }}</div>
            </div>
        </section>

        <section class="workspace">
            <div class="panel">
                <div class="panel-heading">
                    <div>
                        <div class="panel-kicker">Editor</div>
                        <h2>Рабочая зона</h2>
                        <p>Пишите свой код, подставляйте шаблоны и сразу прогоняйте модель.</p>
                    </div>
                </div>

                <div class="field-grid">
                    <div>
                        <label for="starterSelect">Быстрый старт</label>
                        <select id="starterSelect"></select>
                    </div>
                    <div>
                        <label for="sampleSelect">Пример из tests.py</label>
                        <select id="sampleSelect"></select>
                    </div>
                </div>

                <div class="actions">
                    <button class="button-secondary" id="loadStarterButton" type="button">Подставить стартовый пример</button>
                    <button class="button-secondary" id="loadSampleButton" type="button">Загрузить пример из тестов</button>
                    <button class="button-ghost" id="clearButton" type="button">Очистить поле</button>
                </div>

                <div style="margin-top: 16px;">
                    <label for="codeInput">Незавершённый Python-код</label>
                    <textarea id="codeInput" spellcheck="false" placeholder="Например:
def add(a, b):
    return "></textarea>
                </div>

                <div class="actions">
                    <button class="button-primary" id="generateButton" type="button" {% if not runtime.checkpoint_exists %}disabled{% endif %}>Сгенерировать продолжение</button>
                    <button class="button-ghost" id="copyInputButton" type="button">Скопировать ввод</button>
                </div>
            </div>

            <div class="panel">
                <div class="panel-kicker">Guide</div>
                <h2>Как пользоваться</h2>
                <p>
                    Интерфейс собран под быструю ручную проверку модели, а не под демонстрационный шум.
                </p>
                <ul class="tip-list" style="margin-top: 14px;">
                    <li>Короткие и понятные префиксы обычно дают более чистое продолжение.</li>
                    <li>Готовые кейсы можно загружать и сразу дорабатывать под свои сценарии.</li>
                    <li>Результат показывает и точное дополнение, и полный восстановленный код.</li>
                    <li>Служебная диагностика остаётся отдельно, чтобы не мешать чтению основного вывода.</li>
                </ul>

                <div class="chip-row">
                    <div class="chip">Высокий контраст</div>
                    <div class="chip">Удобный ввод</div>
                    <div class="chip">Готовые демо-примеры</div>
                </div>

                <details>
                    <summary>Параметры инференса</summary>
                    <div class="details-body">
                        <div class="json-box" id="runtimeBox"></div>
                    </div>
                </details>
            </div>
        </section>

        <section class="result-panel">
            <div class="result-header">
                <h2>Результат</h2>
                <p>После запуска здесь появятся продолжение модели, итоговая сборка кода и диагностика.</p>
            </div>

            <div class="status-row" id="statusRow">
                <div class="status-pill status-neutral">Ожидание запуска</div>
            </div>

            <div class="tabs">
                <button class="tab-button active" data-tab="combined" type="button">В контексте</button>
                <button class="tab-button" data-tab="generated" type="button">Только дополнение</button>
                <button class="tab-button" data-tab="full" type="button">Полный код</button>
                <button class="tab-button" data-tab="diagnostics" type="button">Диагностика</button>
            </div>

            <div class="tab-pane active" id="tab-combined">
                <div class="empty-state" id="combinedEmpty">
                    Запустите генерацию, чтобы здесь появился код вместе с исходным контекстом.
                </div>
                <div class="code-box" id="combinedBox" hidden><pre id="combinedCode"></pre></div>
            </div>

            <div class="tab-pane" id="tab-generated">
                <div class="empty-state" id="generatedEmpty">
                    Здесь будет только новый фрагмент, который модель добавила к вашему вводу.
                </div>
                <div class="code-box" id="generatedBox" hidden><pre id="generatedCode"></pre></div>
            </div>

            <div class="tab-pane" id="tab-full">
                <div class="empty-state" id="fullEmpty">
                    После реконструкции здесь появится полный Python-код.
                </div>
                <div class="code-box" id="fullBox" hidden><pre id="fullCode"></pre></div>
                <div class="result-tools">
                    <button class="button-secondary" id="copyFullButton" type="button">Скопировать полный код</button>
                    <button class="button-ghost" id="downloadFullButton" type="button">Скачать .py</button>
                </div>
            </div>

            <div class="tab-pane" id="tab-diagnostics">
                <div class="json-box" id="diagnosticsBox">Пока ничего нет.</div>
            </div>
        </section>
    </div>

    <script>
        const starterSnippets = {{ starter_snippets | safe }};
        const sampleOptions = {{ sample_options | safe }};
        const runtime = {{ runtime_json | safe }};

        const starterSelect = document.getElementById("starterSelect");
        const sampleSelect = document.getElementById("sampleSelect");
        const codeInput = document.getElementById("codeInput");
        const generateButton = document.getElementById("generateButton");
        const loadStarterButton = document.getElementById("loadStarterButton");
        const loadSampleButton = document.getElementById("loadSampleButton");
        const clearButton = document.getElementById("clearButton");
        const copyInputButton = document.getElementById("copyInputButton");
        const copyFullButton = document.getElementById("copyFullButton");
        const downloadFullButton = document.getElementById("downloadFullButton");
        const runtimeBox = document.getElementById("runtimeBox");
        const diagnosticsBox = document.getElementById("diagnosticsBox");
        const combinedEmpty = document.getElementById("combinedEmpty");
        const generatedEmpty = document.getElementById("generatedEmpty");
        const fullEmpty = document.getElementById("fullEmpty");
        const combinedBox = document.getElementById("combinedBox");
        const generatedBox = document.getElementById("generatedBox");
        const fullBox = document.getElementById("fullBox");
        const combinedCode = document.getElementById("combinedCode");
        const generatedCode = document.getElementById("generatedCode");
        const fullCode = document.getElementById("fullCode");
        const statusRow = document.getElementById("statusRow");

        let latestFullCode = "";

        function populateSelect(select, options) {
            Object.keys(options).forEach((key) => {
                const option = document.createElement("option");
                option.value = key;
                option.textContent = key;
                select.appendChild(option);
            });
        }

        function setStatus(items) {
            statusRow.innerHTML = "";
            items.forEach((item) => {
                const pill = document.createElement("div");
                pill.className = `status-pill ${item.className}`;
                pill.textContent = item.label;
                statusRow.appendChild(pill);
            });
        }

        function setCode(preNode, boxNode, emptyNode, value, emptyText) {
            const content = value && value.length ? value : "";
            if (content) {
                preNode.textContent = content;
                boxNode.hidden = false;
                emptyNode.hidden = true;
            } else {
                preNode.textContent = "";
                boxNode.hidden = true;
                emptyNode.hidden = false;
                emptyNode.textContent = emptyText;
            }
        }

        function setDiagnostics(data) {
            diagnosticsBox.textContent = JSON.stringify(data, null, 2);
        }

        async function copyText(text, fallbackLabel) {
            if (!text) {
                return;
            }
            try {
                await navigator.clipboard.writeText(text);
                setStatus([
                    { label: fallbackLabel, className: "status-success" }
                ]);
            } catch (_error) {
                setStatus([
                    { label: "Не удалось скопировать", className: "status-error" }
                ]);
            }
        }

        function downloadFullCode() {
            if (!latestFullCode) {
                return;
            }
            const blob = new Blob([latestFullCode], { type: "text/x-python;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "generated_completion.py";
            document.body.appendChild(link);
            link.click();
            link.remove();
            URL.revokeObjectURL(url);
        }

        function activateTab(tabName) {
            document.querySelectorAll(".tab-button").forEach((button) => {
                button.classList.toggle("active", button.dataset.tab === tabName);
            });
            document.querySelectorAll(".tab-pane").forEach((pane) => {
                pane.classList.toggle("active", pane.id === `tab-${tabName}`);
            });
        }

        function insertAtSelection(textarea, text) {
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            textarea.setRangeText(text, start, end, "end");
        }

        function indentSelection(textarea) {
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            const value = textarea.value;
            const lineStart = value.lastIndexOf("\\n", start - 1) + 1;
            const selected = value.slice(lineStart, end);
            const updated = selected.replace(/^/gm, "    ");
            textarea.setRangeText(updated, lineStart, end, "preserve");
            textarea.selectionStart = start + 4;
            textarea.selectionEnd = end + 4 * (updated.match(/^    /gm) || []).length;
        }

        function outdentSelection(textarea) {
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            const value = textarea.value;
            const lineStart = value.lastIndexOf("\\n", start - 1) + 1;
            const selected = value.slice(lineStart, end);
            const matches = selected.match(/^ {1,4}/gm) || [];
            const updated = selected.replace(/^ {1,4}/gm, "");
            textarea.setRangeText(updated, lineStart, end, "preserve");
            const removed = matches.reduce((sum, match) => sum + match.length, 0);
            textarea.selectionStart = Math.max(lineStart, start - Math.min(4, start - lineStart));
            textarea.selectionEnd = Math.max(textarea.selectionStart, end - removed);
        }

        function handleEditorKeys(event) {
            if (event.key === "Tab") {
                event.preventDefault();
                if (event.shiftKey) {
                    outdentSelection(event.target);
                } else if (event.target.selectionStart !== event.target.selectionEnd || event.target.value.slice(0, event.target.selectionStart).includes("\\n")) {
                    indentSelection(event.target);
                } else {
                    insertAtSelection(event.target, "    ");
                }
                return;
            }

            if (event.key === "Enter") {
                event.preventDefault();
                const textarea = event.target;
                const start = textarea.selectionStart;
                const value = textarea.value;
                const lineStart = value.lastIndexOf("\\n", start - 1) + 1;
                const currentLine = value.slice(lineStart, start);
                const indentMatch = currentLine.match(/^\\s*/);
                const baseIndent = indentMatch ? indentMatch[0] : "";
                const extraIndent = /:\\s*$/.test(currentLine) ? "    " : "";
                insertAtSelection(textarea, "\\n" + baseIndent + extraIndent);
            }
        }

        async function generateCompletion() {
            const code = codeInput.value;
            generateButton.disabled = true;
            generateButton.textContent = "Генерация...";
            setStatus([
                { label: "Модель работает", className: "status-neutral" }
            ]);

            try {
                const response = await fetch("/api/complete", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ code })
                });

                const payload = await response.json();
                if (!response.ok) {
                    throw new Error(payload.error || "Ошибка сервера");
                }

                const generated = payload.generated_completion_text || "";
                const full = payload.full_generated_code || "";
                const combined = (payload.input_code || "") + generated;
                latestFullCode = full;

                setCode(
                    combinedCode,
                    combinedBox,
                    combinedEmpty,
                    combined,
                    "Пустой результат."
                );
                setCode(
                    generatedCode,
                    generatedBox,
                    generatedEmpty,
                    generated || "<EMPTY>",
                    "Пустой результат."
                );
                setCode(
                    fullCode,
                    fullBox,
                    fullEmpty,
                    full || "<EMPTY>",
                    "Пустой результат."
                );

                const hasError = Boolean(payload.error);
                setStatus([
                    { label: `AST-токенов: ${(payload.new_tokens || []).length}`, className: "status-neutral" },
                    { label: hasError ? "Есть ошибка реконструкции" : "Успешно", className: hasError ? "status-error" : "status-success" }
                ]);

                setDiagnostics({
                    token_count: (payload.new_tokens || []).length,
                    has_error: hasError,
                    error: payload.error || null
                });
                activateTab("combined");
            } catch (error) {
                setStatus([
                    { label: "Ошибка генерации", className: "status-error" }
                ]);
                setDiagnostics({
                    token_count: 0,
                    has_error: true,
                    error: error.message
                });
            } finally {
                generateButton.disabled = !runtime.checkpoint_exists;
                generateButton.textContent = "Сгенерировать продолжение";
            }
        }

        populateSelect(starterSelect, starterSnippets);
        populateSelect(sampleSelect, sampleOptions);

        runtimeBox.textContent = JSON.stringify(runtime, null, 2);
        codeInput.value = starterSnippets[Object.keys(starterSnippets)[0]] || "";

        loadStarterButton.addEventListener("click", () => {
            codeInput.value = starterSnippets[starterSelect.value] || "";
            codeInput.focus();
        });

        loadSampleButton.addEventListener("click", () => {
            codeInput.value = sampleOptions[sampleSelect.value] || "";
            codeInput.focus();
        });

        clearButton.addEventListener("click", () => {
            codeInput.value = "";
            codeInput.focus();
        });

        copyInputButton.addEventListener("click", () => copyText(codeInput.value, "Ввод скопирован"));
        copyFullButton.addEventListener("click", () => copyText(latestFullCode, "Полный код скопирован"));
        downloadFullButton.addEventListener("click", downloadFullCode);
        generateButton.addEventListener("click", generateCompletion);
        codeInput.addEventListener("keydown", handleEditorKeys);

        document.querySelectorAll(".tab-button").forEach((button) => {
            button.addEventListener("click", () => activateTab(button.dataset.tab));
        });
    </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(
        PAGE_TEMPLATE,
        starter_snippets=json.dumps(STARTER_SNIPPETS, ensure_ascii=False),
        sample_options=json.dumps(SAMPLE_OPTIONS, ensure_ascii=False),
        runtime=runtime_meta(),
        runtime_json=json.dumps(runtime_meta(), ensure_ascii=False),
    )


@app.post("/api/complete")
def api_complete():
    if not CHECKPOINT_PATH.exists():
        return (
            jsonify(
                {
                    "error": "Checkpoint not found. Add checkpoints_code_lm/best.pt and restart the frontend."
                }
            ),
            400,
        )

    payload = request.get_json(silent=True) or {}
    code = payload.get("code", "")
    if not isinstance(code, str):
        return jsonify({"error": "Field 'code' must be a string."}), 400

    result = run_completion(code)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
