from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st

import MLCode
from parse_python import continue_real_code_safe
from tests import SAMPLE_GROUPS, build_tokenizer, set_if_present


CHECKPOINT_PATH = PROJECT_DIR / "checkpoints_code_lm" / "best.pt"

st.set_page_config(
    page_title="AST-дополнение кода",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            --bg: #0b1020;
            --panel: rgba(15, 23, 42, 0.78);
            --panel-strong: #111827;
            --panel-soft: rgba(30, 41, 59, 0.72);
            --text: #e5eefc;
            --muted: #94a3b8;
            --line: rgba(148, 163, 184, 0.18);
            --accent: #60a5fa;
            --accent-2: #22c55e;
            --accent-3: #a78bfa;
            --danger: #f97316;
            --shadow: 0 20px 60px rgba(2, 6, 23, 0.35);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(96, 165, 250, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(167, 139, 250, 0.14), transparent 24%),
                linear-gradient(180deg, #020617 0%, #0b1020 45%, #0f172a 100%);
        }

        .block-container {
            max-width: 1320px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        section[data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.78);
            border-right: 1px solid var(--line);
            backdrop-filter: blur(16px);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }

        .sidebar-shell {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.72));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 16px;
            box-shadow: var(--shadow);
            margin-bottom: 0.9rem;
        }

        .sidebar-kicker {
            color: #93c5fd;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .sidebar-title {
            color: white;
            font-size: 1.08rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .sidebar-text {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .sidebar-stat {
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            background: rgba(2, 6, 23, 0.42);
            padding: 11px 12px;
            margin-top: 0.65rem;
        }

        .sidebar-stat-label {
            color: #93c5fd;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .sidebar-stat-value {
            color: white;
            font-size: 0.98rem;
            font-weight: 700;
            word-break: break-word;
        }

        .sidebar-list {
            margin: 0.8rem 0 0 0;
            padding-left: 1rem;
            color: #dbeafe;
        }

        .sidebar-list li {
            margin-bottom: 0.4rem;
            line-height: 1.45;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 30px;
            border-radius: 28px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(15, 23, 42, 0.72)),
                linear-gradient(120deg, rgba(96, 165, 250, 0.10), rgba(167, 139, 250, 0.08));
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 300px;
            height: 300px;
            right: -80px;
            top: -80px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(96,165,250,0.28), transparent 65%);
            pointer-events: none;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.75rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #bfdbfe;
            background: rgba(96, 165, 250, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.22);
            border-radius: 999px;
            padding: 8px 12px;
            margin-bottom: 1rem;
            font-weight: 700;
        }

        .hero h1 {
            margin: 0;
            font-size: 3rem;
            line-height: 1;
            letter-spacing: -0.04em;
            max-width: 760px;
        }

        .hero p {
            margin: 1rem 0 0 0;
            color: #cbd5e1;
            font-size: 1.02rem;
            line-height: 1.7;
            max-width: 780px;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.55fr 0.95fr;
            gap: 20px;
            align-items: stretch;
        }

        .hero-side {
            position: relative;
            z-index: 1;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.78), rgba(15, 23, 42, 0.95));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 24px;
            padding: 20px;
        }

        .hero-side-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .hero-side p {
            margin: 0;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin: 1rem 0 1.25rem 0;
        }

        .kpi {
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 18px;
            box-shadow: var(--shadow);
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }

        .kpi-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text);
            word-break: break-word;
        }

        .surface {
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 22px;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .surface-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .surface-subtitle {
            color: var(--muted);
            line-height: 1.65;
            margin-bottom: 1rem;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 1rem;
        }

        .mini-card {
            border-radius: 20px;
            border: 1px solid var(--line);
            background: rgba(30, 41, 59, 0.42);
            padding: 16px;
            color: #dbeafe;
        }

        .mini-card strong {
            display: block;
            color: white;
            margin-bottom: 0.25rem;
        }

        div[data-baseweb="tab-list"] {
            gap: 10px;
            margin: 0.25rem 0 1rem 0;
        }

        button[data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.72);
            color: var(--text);
            border: 1px solid var(--line);
            border-radius: 14px;
            min-height: 48px;
            padding: 10px 16px;
            font-weight: 700;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, rgba(96,165,250,0.22), rgba(167,139,250,0.22));
            border-color: rgba(96, 165, 250, 0.38);
        }

        .stTextArea textarea {
            font-family: 'JetBrains Mono', monospace !important;
            border-radius: 18px !important;
            background: rgba(2, 6, 23, 0.72) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            padding: 16px !important;
            line-height: 1.6 !important;
        }

        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.85) !important;
            border: 1px solid var(--line) !important;
            border-radius: 14px !important;
            min-height: 48px;
        }

        .stButton > button {
            border: 0;
            border-radius: 16px;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            font-weight: 700;
            min-height: 50px;
            box-shadow: 0 14px 28px rgba(59, 130, 246, 0.22);
        }

        .stButton > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }

        .result-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }

        .result-card {
            background: rgba(2, 6, 23, 0.74);
            border: 1px solid rgba(96, 165, 250, 0.14);
            border-radius: 22px;
            padding: 16px;
            height: 100%;
        }

        .result-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #93c5fd;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 0.9rem;
        }

        .status-pill {
            border-radius: 999px;
            padding: 8px 12px;
            background: rgba(96, 165, 250, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.18);
            color: #dbeafe;
            font-size: 0.84rem;
        }

        .empty {
            padding: 20px;
            border-radius: 20px;
            background: rgba(15, 23, 42, 0.5);
            border: 1px dashed rgba(148, 163, 184, 0.22);
            color: var(--muted);
            line-height: 1.65;
        }

        .footer-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }

        .stCodeBlock pre, .stCode pre {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
        }

        @media (max-width: 1100px) {
            .hero-grid, .kpi-grid, .mini-grid, .result-grid {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 2.3rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Загрузка чекпоинта и токенизатора...")
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


def render_result(result: dict) -> None:
    generated = result.get("generated_completion_text") or "<EMPTY>"
    full_code = result.get("full_generated_code") or "<EMPTY>"
    input_code = result.get("input_code") or ""
    token_count = len(result.get("new_tokens") or [])
    has_error = bool(result.get("error"))
    inline_preview = input_code + generated if input_code else generated

    st.markdown('<div class="surface"><div class="surface-title">Результат</div><div class="surface-subtitle">Сравните точный сгенерированный фрагмент и полный восстановленный Python-код. Оба представления остаются рядом и легко читаются.</div>', unsafe_allow_html=True)

    st.markdown('<div class="result-card"><div class="result-label">Продолжение в контексте исходного ввода</div>', unsafe_allow_html=True)
    st.code(inline_preview or "<EMPTY>", language="python")
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="result-card"><div class="result-label">Сгенерированное дополнение</div>', unsafe_allow_html=True)
        st.code(generated, language="python")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="result-card"><div class="result-label">Полный восстановленный код</div>', unsafe_allow_html=True)
        st.code(full_code, language="python")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'''<div class="status-row">
            <div class="status-pill">Сгенерировано AST-токенов: {token_count}</div>
            <div class="status-pill">Статус: {'Ошибка' if has_error else 'Успешно'}</div>
        </div></div>''',
        unsafe_allow_html=True,
    )

    with st.expander("Диагностика", expanded=False):
        st.json(
            {
                "token_count": token_count,
                "has_error": has_error,
                "error": result.get("error"),
            }
        )

    if has_error:
        st.error(result["error"])



def render_custom_input_tab() -> None:
    st.markdown(
        '''
        <div class="surface">
            <div class="surface-title">Рабочая область</div>
            <div class="surface-subtitle">Вставьте неполный Python-префикс и запустите тот же inference-пайплайн, который уже используется в проекте. Интерфейс сосредоточен на одном главном действии: ввод сверху, результат снизу, диагностика отдельно.</div>
            <div class="mini-grid">
                <div class="mini-card"><strong>Подходит для</strong>Быстрой проверки промптов, ручного тестирования и демонстрации модели без борьбы с интерфейсом.</div>
                <div class="mini-card"><strong>Как использовать</strong>Оставьте курсор в точке, где должно начаться дополнение, и запустите генерацию. Приложение покажет и точный фрагмент, и восстановленный код.</div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    default_code = "def greet(name):\n    message = f'Hello, {name}'\n    print("
    code = st.text_area(
        "Ввод Python-кода",
        value=default_code,
        height=320,
        placeholder="Введите сюда неполный Python-фрагмент...",
        help="Модель работает лучше, когда префикс ясен, а точка продолжения очевидна.",
    )

    c1, c2 = st.columns([1, 1])
    run_clicked = c1.button("Сгенерировать продолжение", use_container_width=True)
    c2.caption("Совет: короткие и чёткие префиксы удобнее для быстрой визуальной проверки.")

    if run_clicked:
        with st.spinner("Генерируется продолжение..."):
            result = run_completion(code)
        render_result(result)
    else:
        st.markdown(
            '<div class="empty">Введите или вставьте Python-префикс, затем нажмите <strong>Сгенерировать продолжение</strong>. Ниже появятся точный фрагмент, который добавила модель, и полный восстановленный код.</div>',
            unsafe_allow_html=True,
        )



def render_examples_tab() -> None:
    st.markdown(
        '''
        <div class="surface">
            <div class="surface-title">Примеры</div>
            <div class="surface-subtitle">Запускайте подготовленные промпты из <code>tests.py</code> и смотрите результаты в том же двухпанельном формате.</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    group_name = st.selectbox("Группа примеров", list(SAMPLE_GROUPS.keys()))
    sample_names = {sample_name: code for sample_name, code in SAMPLE_GROUPS[group_name]}
    sample_name = st.selectbox("Пример", list(sample_names.keys()))
    code = sample_names[sample_name]

    st.code(code, language="python")

    if st.button("Запустить пример", key="run_example", use_container_width=True):
        with st.spinner("Выполняется пример..."):
            result = run_completion(code)
        render_result(result)
    else:
        st.markdown(
            '<div class="empty">Выберите пример и запустите его, чтобы проверить поведение модели на подготовленных сценариях.</div>',
            unsafe_allow_html=True,
        )



def render_sidebar(cfg, device: str) -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-shell">
                <div class="sidebar-kicker">Runtime</div>
                <div class="sidebar-title">Параметры запуска</div>
                <div class="sidebar-text">Здесь оставлена только полезная информация о текущем состоянии модели и конфигурации инференса.</div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Чекпоинт</div>
                    <div class="sidebar-stat-value">{CHECKPOINT_PATH.name}</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Устройство</div>
                    <div class="sidebar-stat-value">{device}</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Макс. новых токенов</div>
                    <div class="sidebar-stat-value">{getattr(cfg, 'infer_max_new_tokens', 96)}</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Temperature</div>
                    <div class="sidebar-stat-value">{getattr(cfg, 'infer_temperature', 0.2)}</div>
                </div>
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Top-k / Top-p</div>
                    <div class="sidebar-stat-value">{getattr(cfg, 'infer_top_k', 8)} / {getattr(cfg, 'infer_top_p', 0.8)}</div>
                </div>
            </div>
            <div class="sidebar-shell">
                <div class="sidebar-kicker">Справка</div>
                <div class="sidebar-title">Как читать результат</div>
                <ul class="sidebar-list">
                    <li>слева показывается только добавленный моделью фрагмент</li>
                    <li>справа выводится полный код после восстановления</li>
                    <li>в разделе «Диагностика» доступны служебные детали запуска</li>
                    <li>вкладка «Примеры» использует кейсы из <code>tests.py</code></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )



def main() -> None:
    inject_styles()
    _model, _vocab, _ivocab, _tokenizer, cfg, device = load_runtime()
    render_sidebar(cfg, device)

    st.markdown(
        f'''
        <section class="hero">
            <div class="hero-grid">
                <div>
                    <div class="eyebrow">AST-дополнение кода</div>
                    <h1>Интерактивная площадка для тестирования дополнений модели</h1>
                    <p>
                        Здесь можно запускать дополнение кода на собственных префиксах, смотреть точный фрагмент,
                        который сгенерировала модель, и сравнивать его с полным восстановленным Python-кодом.
                    </p>
                </div>
                <div class="hero-side">
                    <div class="hero-side-title">Для чего подходит страница</div>
                    <p>Для ручной проверки модели, демонстрации примеров и быстрого сравнения результатов на пользовательском вводе и готовых сценариях.</p>
                </div>
            </div>
        </section>
        ''',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'''
        <div class="kpi-grid">
            <div class="kpi"><div class="kpi-label">Чекпоинт</div><div class="kpi-value">{CHECKPOINT_PATH.name}</div></div>
            <div class="kpi"><div class="kpi-label">Устройство</div><div class="kpi-value">{device}</div></div>
            <div class="kpi"><div class="kpi-label">Длина префикса</div><div class="kpi-value">{getattr(cfg, "infer_prefix_len", 256)}</div></div>
            <div class="kpi"><div class="kpi-label">Макс. новых токенов</div><div class="kpi-value">{getattr(cfg, "infer_max_new_tokens", 96)}</div></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    workspace_tab, examples_tab = st.tabs(["Рабочая область", "Примеры"])

    with workspace_tab:
        render_custom_input_tab()

    with examples_tab:
        render_examples_tab()

    st.markdown('<div class="footer-note">Страница использует тот же inference-пайплайн проекта и позволяет быстро проверять результаты на своих примерах и на кейсах из тестового набора.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
