"""
indic-eval Streamlit Dashboard
================================
Run with:
    streamlit run dashboard/app.py

Features:
- Run evaluations directly from the browser
- Load and compare saved result JSONs
- View per-sample predictions vs references
- Download results as JSON
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path so indic_eval imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="indic-eval",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

    /* Score cards */
    .score-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #f78166;
    }
    .score-card-high  { border-left-color: #3fb950; }
    .score-card-mid   { border-left-color: #d29922; }
    .score-card-low   { border-left-color: #f78166; }

    .score-label { font-size: 0.78rem; color: #8b949e; margin-bottom: 0.3rem; letter-spacing: 0.05em; text-transform: uppercase; }
    .score-value-high { font-size: 2rem; font-weight: 700; color: #3fb950; line-height: 1; }
    .score-value-mid  { font-size: 2rem; font-weight: 700; color: #d29922; line-height: 1; }
    .score-value-low  { font-size: 2rem; font-weight: 700; color: #f78166; line-height: 1; }
    .score-meta { font-size: 0.75rem; color: #484f58; margin-top: 0.3rem; }

    /* Section headers */
    .section-header {
        font-size: 0.7rem;
        font-weight: 600;
        color: #8b949e;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #21262d;
    }

    /* Overall banner */
    .overall-banner {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 2rem;
    }

    /* Sample prediction boxes */
    .pred-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-family: monospace;
        font-size: 0.9rem;
        color: #e6edf3;
        margin-top: 0.25rem;
    }
    .pred-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }

    /* Comparison table */
    .compare-table { width: 100%; border-collapse: collapse; }
    .compare-table th { background: #161b22; color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #30363d; }
    .compare-table td { padding: 0.6rem 1rem; border-bottom: 1px solid #21262d; color: #e6edf3; font-size: 0.9rem; }
    .compare-table tr:last-child td { border-bottom: none; font-weight: 600; }
    .best-score { color: #3fb950; font-weight: 700; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────

TASK_META = {
    "hindi_reading_comprehension": {
        "icon": "📖", "label": "Hindi Reading Comprehension",
        "metric": "exact_match", "lang": "Hindi"
    },
    "en_hi_translation": {
        "icon": "🔄", "label": "EN → HI Translation",
        "metric": "bleu", "lang": "EN → HI"
    },
    "hinglish_sentiment": {
        "icon": "💬", "label": "Hinglish Sentiment",
        "metric": "exact_match", "lang": "Hinglish"
    },
    "indian_cultural_reasoning": {
        "icon": "🏛️", "label": "Indian Cultural Reasoning",
        "metric": "mcq_accuracy", "lang": "Hindi/EN"
    },
    "hindi_summarisation": {
        "icon": "📝", "label": "Hindi Summarisation",
        "metric": "rouge_l", "lang": "Hindi"
    },
}


# ── Utility functions ──────────────────────────────────────────────────────────

def score_class(score: float) -> str:
    if score >= 0.7: return "high"
    if score >= 0.4: return "mid"
    return "low"


def render_score_card(task_name: str, score: float, metric_name: str,
                      n_samples: int = None, latency_ms: float = None):
    meta = TASK_META.get(task_name, {"icon": "⚙️", "label": task_name})
    cls  = score_class(score)
    meta_parts = []
    if metric_name:   meta_parts.append(metric_name)
    if n_samples:     meta_parts.append(f"{n_samples} samples")
    if latency_ms:    meta_parts.append(f"{latency_ms:.0f}ms avg")
    meta_str = "  ·  ".join(meta_parts)

    st.markdown(f"""
    <div class="score-card score-card-{cls}">
        <div class="score-label">{meta['icon']} {meta.get('label', task_name)}</div>
        <div class="score-value-{cls}">{score * 100:.1f}%</div>
        <div class="score-meta">{meta_str}</div>
    </div>
    """, unsafe_allow_html=True)


def render_overall_banner(model_name: str, overall: float, n_tasks: int, time_s: float = None):
    cls = score_class(overall)
    time_str = f"  ·  {time_s:.1f}s" if time_s else ""
    st.markdown(f"""
    <div class="overall-banner">
        <div>
            <div class="score-label">Overall Score · {n_tasks} tasks{time_str}</div>
            <div class="score-value-{cls}" style="font-size:2.8rem">{overall * 100:.1f}%</div>
        </div>
        <div style="border-left:1px solid #30363d; padding-left:2rem">
            <div class="score-label">Model</div>
            <div style="color:#e6edf3; font-size:1.1rem; font-weight:600; margin-top:0.2rem">{model_name}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sample_predictions(task_result):
    """Show prediction vs reference for the first few samples."""
    preds = task_result.get("predictions", [])
    refs  = task_result.get("references",  [])
    if not preds:
        st.info("No sample predictions available in this result file.")
        return

    for i, (pred, ref) in enumerate(zip(preds[:5], refs[:5])):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="pred-label">Prediction</div><div class="pred-box">{pred}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="pred-label">Reference</div><div class="pred-box">{ref}</div>', unsafe_allow_html=True)
        if i < min(len(preds), 5) - 1:
            st.markdown("<hr style='border-color:#21262d; margin:0.75rem 0'>", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🇮🇳 indic-eval")
    st.markdown("<div style='color:#8b949e; font-size:0.85rem'>Evaluation harness for LLMs on Indic language tasks</div>", unsafe_allow_html=True)
    st.divider()

    mode = st.radio(
        "Mode",
        ["▶  Run Evaluation", "📂  Load Results", "⚖️  Compare Models"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("<div class='section-header'>Tasks</div>", unsafe_allow_html=True)
    for key, meta in TASK_META.items():
        st.markdown(
            f"<div style='font-size:0.82rem; color:#8b949e; padding:0.15rem 0'>"
            f"{meta['icon']} <span style='color:#e6edf3'>{meta['label']}</span>"
            f" <span style='color:#484f58'>· {meta['lang']}</span></div>",
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        "<div style='font-size:0.75rem; color:#484f58'>"
        "<a href='https://github.com/shivamkesarwani/indic-eval' style='color:#58a6ff'>GitHub</a>"
        " · "
        "<a href='https://www.sarvam.ai/blogs/evaluating-indian-language-asr' style='color:#58a6ff'>Sarvam Blog</a>"
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Run Evaluation
# ══════════════════════════════════════════════════════════════════════════════

if "Run" in mode:
    st.markdown("## Run Evaluation")
    st.markdown("<div style='color:#8b949e; margin-bottom:1.5rem'>Configure your model and tasks, then click Run.</div>", unsafe_allow_html=True)

    col_model, col_tasks = st.columns([1, 1], gap="large")

    with col_model:
        st.markdown("<div class='section-header'>Model</div>", unsafe_allow_html=True)

        model_type = st.selectbox("Backend", ["API (OpenAI-compatible)", "HuggingFace"], label_visibility="collapsed")

        if "API" in model_type:
            preset = st.selectbox(
                "Quick select",
                ["Custom", "Sarvam AI", "OpenAI", "Groq (free)", "Ollama (local)"],
            )
            preset_urls = {
                "Sarvam AI":    ("sarvam-2b",       "https://api.sarvam.ai/v1"),
                "OpenAI":       ("gpt-4o-mini",      "https://api.openai.com/v1"),
                "Groq (free)":  ("llama3-8b-8192",   "https://api.groq.com/openai/v1"),
                "Ollama (local)": ("llama3",          "http://localhost:11434/v1"),
                "Custom":       ("",                  "https://api.openai.com/v1"),
            }
            default_model, default_url = preset_urls.get(preset, ("", ""))
            model_name = st.text_input("Model name", value=default_model, placeholder="e.g. sarvam-2b")
            base_url   = st.text_input("Base URL",   value=default_url)
            api_key    = st.text_input("API Key", type="password",
                                       value=os.environ.get("OPENAI_API_KEY", ""),
                                       help="Leave blank for Ollama or models that don't need a key")
        else:
            model_name = st.text_input("HuggingFace model ID",
                                       value="sarvamai/sarvam-2b-v0.5",
                                       placeholder="e.g. sarvamai/sarvam-2b-v0.5")
            device     = st.selectbox("Device", ["cpu", "cuda", "mps"])
            load_8bit  = st.checkbox("8-bit quantization (requires bitsandbytes)",
                                     help="Reduces memory usage for large models like Sarvam 30B")

    with col_tasks:
        st.markdown("<div class='section-header'>Tasks & Samples</div>", unsafe_allow_html=True)

        all_task_keys = list(TASK_META.keys())
        selected_tasks = st.multiselect(
            "Select tasks",
            options=all_task_keys,
            default=all_task_keys,
            format_func=lambda k: f"{TASK_META[k]['icon']} {TASK_META[k]['label']}",
            label_visibility="collapsed",
        )

        n_samples = st.slider("Samples per task", min_value=3, max_value=100, value=10,
                              help="More samples = more accurate results but slower")

        st.markdown(
            f"<div style='color:#8b949e; font-size:0.82rem; margin-top:0.5rem'>"
            f"Estimated: ~{len(selected_tasks) * n_samples} total model calls</div>",
            unsafe_allow_html=True
        )

    st.markdown("")
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("▶  Run Evaluation", type="primary", use_container_width=True)

    if run_btn:
        if not selected_tasks:
            st.error("Please select at least one task.")
        elif not model_name:
            st.error("Please enter a model name.")
        else:
            try:
                from indic_eval.models import load_model
                from indic_eval.tasks import get_task
                from indic_eval.evaluator import EvalReport

                if "API" in model_type:
                    model_config = {"type": "api", "model": model_name,
                                    "base_url": base_url, "api_key": api_key or None}
                else:
                    model_config = {"type": "hf", "model": model_name,
                                    "device": device, "load_in_8bit": load_8bit}

                model = load_model(model_config)

                progress_bar = st.progress(0, text="Starting...")
                status_box   = st.empty()
                results      = []
                t0           = time.time()

                for i, task_name in enumerate(selected_tasks):
                    meta = TASK_META.get(task_name, {})
                    progress_bar.progress(
                        i / len(selected_tasks),
                        text=f"{meta.get('icon','⚙️')} Running: {meta.get('label', task_name)}"
                    )
                    task   = get_task(task_name)
                    result = task.evaluate(model, n=n_samples, verbose=False)
                    results.append(result)
                    m = result.metrics[0]
                    status_box.markdown(
                        f"✓ **{task_name}** — {m.name}: **{m.score*100:.1f}%**"
                    )

                progress_bar.progress(1.0, text="Complete!")
                status_box.empty()

                report = EvalReport(
                    model_name=model_name,
                    tasks=results,
                    total_time_s=time.time() - t0,
                )
                st.session_state["report"]      = report
                st.session_state["report_dict"] = report.summary()

            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                with st.expander("Full error"):
                    st.exception(e)

    # ── Render results if available ────────────────────────────────────────────
    if "report" in st.session_state:
        report = st.session_state["report"]
        st.divider()

        render_overall_banner(
            model_name=report.model_name,
            overall=report.overall_score(),
            n_tasks=len(report.tasks),
            time_s=report.total_time_s,
        )

        st.markdown("<div class='section-header'>Per-Task Results</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(report.tasks), 3))
        for i, t in enumerate(report.tasks):
            m   = t.metrics[0]
            lat = next((x for x in t.metrics if x.name == "latency_ms"), None)
            with cols[i % len(cols)]:
                render_score_card(
                    task_name=t.task_name,
                    score=m.score,
                    metric_name=m.name,
                    n_samples=t.n_samples,
                    latency_ms=lat.score if lat else None,
                )

        with st.expander("🔍 Sample Predictions"):
            task_idx = st.selectbox(
                "Task",
                range(len(report.tasks)),
                format_func=lambda i: report.tasks[i].task_name,
            )
            t = report.tasks[task_idx]
            # Build dict for render function
            render_sample_predictions({
                "predictions": t.predictions,
                "references":  t.references,
            })

        dl_col, _ = st.columns([1, 3])
        with dl_col:
            summary = st.session_state.get("report_dict", {})
            st.download_button(
                "⬇  Download Results JSON",
                data=json.dumps(summary, ensure_ascii=False, indent=2),
                file_name=f"indic_eval_{report.model_name.replace('/', '_')}.json",
                mime="application/json",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Load Results
# ══════════════════════════════════════════════════════════════════════════════

elif "Load" in mode:
    st.markdown("## Load Results")
    st.markdown("<div style='color:#8b949e; margin-bottom:1.5rem'>Upload a saved results JSON to explore it.</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload results JSON", type=["json"], label_visibility="collapsed")

    if uploaded:
        try:
            data = json.load(uploaded)
        except Exception as e:
            st.error(f"Could not parse JSON: {e}")
            st.stop()

        model_name   = data.get("model", "unknown")
        overall      = data.get("overall_score", 0.0)
        tasks_data   = data.get("tasks", [])
        timestamp    = data.get("timestamp", "")
        total_time_s = data.get("total_time_s")

        render_overall_banner(
            model_name=model_name,
            overall=overall,
            n_tasks=len(tasks_data),
            time_s=total_time_s,
        )

        if timestamp:
            st.markdown(f"<div style='color:#484f58; font-size:0.8rem; margin-bottom:1rem'>Evaluated: {timestamp}</div>", unsafe_allow_html=True)

        if not tasks_data:
            st.warning("No task results found in this file.")
        else:
            st.markdown("<div class='section-header'>Per-Task Results</div>", unsafe_allow_html=True)
            cols = st.columns(min(len(tasks_data), 3))
            for i, t in enumerate(tasks_data):
                metrics = t.get("metrics", {})
                if not metrics:
                    continue
                primary_metric, primary_score = list(metrics.items())[0]
                lat = t.get("latency", {}).get("mean_ms")
                with cols[i % len(cols)]:
                    render_score_card(
                        task_name=t["task"],
                        score=primary_score,
                        metric_name=primary_metric,
                        n_samples=t.get("n_samples"),
                        latency_ms=lat,
                    )

            # All metrics table
            with st.expander("📊 All Metrics"):
                for t in tasks_data:
                    st.markdown(f"**{t['task']}**")
                    for metric_name, score in t.get("metrics", {}).items():
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; "
                            f"color:#8b949e; font-size:0.85rem; padding:0.15rem 0'>"
                            f"<span>{metric_name}</span>"
                            f"<span style='color:#e6edf3'>{score*100:.2f}%</span></div>",
                            unsafe_allow_html=True
                        )
                    st.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)

            # Raw JSON
            with st.expander("🗂 Raw JSON"):
                st.json(data)

            # Download
            dl_col, _ = st.columns([1, 3])
            with dl_col:
                st.download_button(
                    "⬇  Download JSON",
                    data=json.dumps(data, ensure_ascii=False, indent=2),
                    file_name=f"indic_eval_{model_name.replace('/', '_')}.json",
                    mime="application/json",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — Compare Models
# ══════════════════════════════════════════════════════════════════════════════

elif "Compare" in mode:
    st.markdown("## Compare Models")
    st.markdown("<div style='color:#8b949e; margin-bottom:1.5rem'>Upload two or more result JSONs to compare them side by side.</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload result JSONs (select multiple)",
        type=["json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and len(uploaded_files) >= 2:
        reports = []
        for f in uploaded_files:
            try:
                reports.append(json.load(f))
            except Exception as e:
                st.error(f"Could not parse {f.name}: {e}")

        if len(reports) < 2:
            st.warning("Need at least 2 valid JSON files to compare.")
        else:
            # Collect all task names across all reports
            all_tasks = []
            for r in reports:
                for t in r.get("tasks", []):
                    if t["task"] not in all_tasks:
                        all_tasks.append(t["task"])

            def get_primary_score(report, task_name):
                for t in report.get("tasks", []):
                    if t["task"] == task_name:
                        scores = list(t.get("metrics", {}).values())
                        return scores[0] if scores else None
                return None

            # Overall scores row
            st.markdown("<div class='section-header'>Overall Score</div>", unsafe_allow_html=True)
            cols = st.columns(len(reports))
            overalls = [r.get("overall_score", 0.0) for r in reports]
            best_overall = max(overalls)
            for i, (r, overall) in enumerate(zip(reports, overalls)):
                with cols[i]:
                    is_best = overall == best_overall
                    cls = score_class(overall)
                    badge = " 🏆" if is_best else ""
                    st.markdown(f"""
                    <div class="score-card score-card-{cls}">
                        <div class="score-label">{r.get('model','unknown')}{badge}</div>
                        <div class="score-value-{cls}">{overall*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Per-task comparison table
            st.markdown("<div class='section-header'>Per-Task Comparison</div>", unsafe_allow_html=True)

            # Build HTML table
            model_names = [r.get("model", f"Model {i+1}") for i, r in enumerate(reports)]
            header_cells = "".join(f"<th>{m}</th>" for m in model_names)
            rows = ""

            for task_name in all_tasks:
                meta = TASK_META.get(task_name, {"icon": "⚙️", "label": task_name})
                scores = [get_primary_score(r, task_name) for r in reports]
                valid  = [s for s in scores if s is not None]
                best   = max(valid) if valid else None

                cells = ""
                for s in scores:
                    if s is None:
                        cells += "<td style='color:#484f58'>N/A</td>"
                    elif s == best:
                        cells += f"<td class='best-score'>{s*100:.1f}%</td>"
                    else:
                        cells += f"<td>{s*100:.1f}%</td>"

                rows += f"<tr><td>{meta['icon']} {meta['label']}</td>{cells}</tr>"

            # Overall row
            best_o = max(overalls)
            overall_cells = ""
            for o in overalls:
                if o == best_o:
                    overall_cells += f"<td class='best-score'>{o*100:.1f}%</td>"
                else:
                    overall_cells += f"<td>{o*100:.1f}%</td>"
            rows += f"<tr><td><strong>Overall</strong></td>{overall_cells}</tr>"

            st.markdown(f"""
            <table class="compare-table">
                <thead><tr><th>Task</th>{header_cells}</tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)

            # Winner summary
            winner_idx = overalls.index(best_overall)
            winner = reports[winner_idx].get("model", f"Model {winner_idx+1}")
            margin = best_overall - min(overalls)
            st.markdown(
                f"<div style='color:#8b949e; font-size:0.85rem; margin-top:1rem'>"
                f"🏆 <strong style='color:#e6edf3'>{winner}</strong> leads by "
                f"<strong style='color:#3fb950'>{margin*100:.1f}%</strong> overall</div>",
                unsafe_allow_html=True
            )

    elif uploaded_files and len(uploaded_files) == 1:
        st.info("Upload at least one more file to compare.")
    else:
        st.markdown("""
        <div style='background:#161b22; border:1px dashed #30363d; border-radius:12px;
                    padding:2rem; text-align:center; color:#484f58; margin-top:1rem'>
            Upload 2 or more result JSON files to see a side-by-side comparison
        </div>
        """, unsafe_allow_html=True)
