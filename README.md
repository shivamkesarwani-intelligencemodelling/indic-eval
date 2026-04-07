<div align="center">

<br/>

```
  ██╗███╗   ██╗██████╗ ██╗ ██████╗    ███████╗██╗   ██╗ █████╗ ██╗
  ██║████╗  ██║██╔══██╗██║██╔════╝    ██╔════╝██║   ██║██╔══██╗██║
  ██║██╔██╗ ██║██║  ██║██║██║         █████╗  ██║   ██║███████║██║
  ██║██║╚██╗██║██║  ██║██║██║         ██╔══╝  ╚██╗ ██╔╝██╔══██║██║
  ██║██║ ╚████║██████╔╝██║╚██████╗    ███████╗ ╚████╔╝ ██║  ██║███████╗
  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
```

**An open evaluation harness for LLMs on Indic language tasks.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tasks](https://img.shields.io/badge/tasks-5-orange?style=flat-square)](#tasks)
[![Models](https://img.shields.io/badge/backends-API%20%2B%20HuggingFace-purple?style=flat-square)](#supported-models)

<br/>

*Most LLM benchmarks are English-first.*
*India has 22 scheduled languages, 1.4 billion people, and a generation of AI models built for them.*
*This project builds the missing evaluation layer.*

<br/>

</div>

---

## Why indic-eval?

Frameworks like `lm-evaluation-harness` and HELM are built around English-language tasks. When applied to Indic languages, they miss three things that actually matter:

- **Code-switching** — real Indian text is not purely Hindi or purely English. It's Hinglish, and no standard benchmark evaluates it.
- **Cultural grounding** — a model can translate "Onam" correctly and still have no idea what it means. Cultural reasoning is a separate, testable capability.
- **Metric appropriateness** — BLEU was designed for European languages. chrF is demonstrably better for morphologically rich languages like Hindi. Most harnesses don't distinguish.

`indic-eval` is built around these realities. It's lightweight, extensible, and designed to plug into any model — whether you're evaluating GPT-4o, a Sarvam model, or something running locally on Ollama.

---

## Tasks

| # | Task | Language | Primary Metric | Data Source |
|---|------|----------|----------------|-------------|
| 1 | `hindi_reading_comprehension` | Hindi | Exact Match + Token F1 | [IndicQA](https://arxiv.org/abs/2212.05409) |
| 2 | `en_hi_translation` | EN → HI | BLEU + chrF | [FLORES-200](https://arxiv.org/abs/2207.04672) |
| 3 | `hinglish_sentiment` | Hinglish (code-switched) | Accuracy | Custom |
| 4 | `indian_cultural_reasoning` | Hindi / EN | MCQ Accuracy | Custom |
| 5 | `hindi_summarisation` | Hindi | ROUGE-L + chrF | Custom |

> All tasks include bundled fallback samples so the harness runs fully offline without any dataset downloads.

### What makes tasks 3 and 4 novel

**Hinglish Sentiment** is built on real code-switched social text — the kind of language that appears on Twitter, WhatsApp, and product reviews across India. A sentence like *"Yaar ye movie bilkul bakwaas thi"* requires a model to understand romanised Hindi embedded in informal English syntax. No standard benchmark covers this.

**Indian Cultural Reasoning** tests whether a model genuinely understands India — its classical arts, festivals, geography, history, and social context — not just whether it can transliterate Hindi words. A model that scores 90% on translation but 40% here is a model that knows the language but not the country.

---

## Supported Models

Any model, any provider. Two backends:

### OpenAI-compatible API
Works with **Sarvam AI**, OpenAI, Anthropic (via proxy), Groq, Together AI, Ollama, and any provider that implements the `/v1/chat/completions` endpoint.

```python
from indic_eval.models import APIModel

model = APIModel(
    model_name="sarvam-2b",
    base_url="https://api.sarvam.ai/v1",
    api_key="your_key",
)
```

### HuggingFace Transformers
Any causal language model on the Hub — including quantized variants via `bitsandbytes`.

```python
from indic_eval.models import HuggingFaceModel

model = HuggingFaceModel(
    model_name="sarvamai/sarvam-2b-v0.5",
    device="cuda",           # or "cpu", "mps"
    load_in_8bit=True,       # optional: 8-bit quantization
)
```

---

## Quickstart

```bash
git clone https://github.com/shivamkesarwani-intelligencemodelling/indic-eval
cd indic-eval
pip install -e .
```

**Run with no API key (mock model — verifies the pipeline works):**
```bash
python examples/quickstart.py
```

**Run via CLI:**
```bash
# All tasks, save results
indic-eval --model gpt-4o-mini --all-tasks --output results/gpt4o.json

# Specific tasks only
indic-eval --model gpt-4o-mini --tasks hinglish_sentiment indian_cultural_reasoning

# HuggingFace model
indic-eval --model sarvamai/sarvam-2b-v0.5 --model-type hf --device cpu

# List all tasks
indic-eval --list-tasks
```

**Launch the dashboard:**
```bash
streamlit run dashboard/app.py
```

---

## Python API

```python
from indic_eval.models import APIModel
from indic_eval.evaluator import Evaluator

model = APIModel(model_name="gpt-4o-mini", api_key="...")

evaluator = Evaluator(
    model=model,
    tasks=["hindi_reading_comprehension", "hinglish_sentiment", "indian_cultural_reasoning"],
    n_samples=50,
)

report = evaluator.run()
report.print_table()
report.save("results/gpt4o_mini.json")
```

**Output:**
```
======================================================================
  indic-eval  |  Model: gpt-4o-mini
======================================================================
  Task                                 Metric               Score
----------------------------------------------------------------------
  hindi_reading_comprehension          exact_match          71.2%  (184ms)
  hinglish_sentiment                   exact_match          83.3%  (165ms)
  indian_cultural_reasoning            mcq_accuracy         90.0%  (178ms)
----------------------------------------------------------------------
  Overall Score                        (mean)               81.5%
======================================================================
  Done in 42.3s  |  2026-04-05T10:22:11
```

---

## Metrics

| Metric | Used For | Notes |
|--------|----------|-------|
| `exact_match` | QA, classification | Normalised, case-insensitive |
| `token_f1` | Reading comprehension | Standard SQuAD-style F1 |
| `bleu` | Translation | via sacrebleu, corpus-level |
| `chrf` | Translation, summarisation | Better than BLEU for Indic morphology |
| `rouge_l` | Summarisation | LCS-based |
| `mcq_accuracy` | Cultural reasoning | Extracts A/B/C/D from free-form output |
| `latency_ms` | All tasks | mean, p50, p95 tracked per task |

All metrics degrade gracefully — if `sacrebleu` or `rouge_score` are not installed, fallback implementations are used automatically.

---

## Adding a Custom Task

```python
from indic_eval.tasks import BaseTask, TASK_REGISTRY
from indic_eval.metrics import exact_match, latency_stats

class BengaliQA(BaseTask):
    name        = "bengali_qa"
    description = "Reading comprehension in Bengali"
    language    = "bn"

    PROMPT = "নিচের অনুচ্ছেদটি পড়ুন এবং প্রশ্নের উত্তর দিন।\n\nঅনুচ্ছেদ: {context}\n\nপ্রশ্ন: {question}\n\nউত্তর:"

    def load_samples(self, n=50):
        return [
            {"prompt": self.PROMPT.format(context="...", question="..."),
             "reference": "..."},
        ] * n

    def parse_output(self, response):
        return response.text.strip().split("\n")[0]

    def compute_metrics(self, predictions, references, latencies):
        return [exact_match(predictions, references), latency_stats(latencies)]

# Register it
TASK_REGISTRY["bengali_qa"] = BengaliQA
```

That's it. It immediately works with the CLI, the evaluator, and the dashboard.

---

## Dashboard

A Streamlit dashboard for running evaluations and exploring results without touching the terminal.

```bash
streamlit run dashboard/app.py
```

Features:
- Configure and run evaluations directly in the browser
- Load and compare saved result JSONs across models
- View per-sample predictions vs. references
- Download results as JSON

---

## Project Structure

```
indic-eval/
│
├── indic_eval/
│   ├── models/         ← APIModel, HuggingFaceModel, load_model()
│   ├── tasks/          ← All task definitions + registry
│   ├── metrics/        ← BLEU, chrF, ROUGE-L, EM, F1, latency
│   ├── evaluator.py    ← Evaluator + EvalReport
│   └── cli.py          ← indic-eval CLI entrypoint
│
├── dashboard/
│   └── app.py          ← Streamlit dashboard
│
├── examples/
│   └── quickstart.py   ← Zero-config demo (no API key needed)
│
├── results/            ← Saved benchmark JSONs (gitignored)
├── data/samples/       ← Bundled fallback samples
└── tests/              ← Unit tests
```

---

## Installation Options

```bash
# Core (API models only)
pip install -e .

# With HuggingFace support
pip install -e ".[hf]"

# With 8-bit quantization
pip install -e ".[quantized]"

# Development
pip install -e ".[dev]"
```

---

## Roadmap

- [ ] Bengali, Tamil, Telugu task variants
- [ ] Side-by-side multi-model comparison in dashboard
- [ ] Automatic leaderboard generation from results JSONs
- [ ] IndicGLUE task integration
- [ ] LangChain / LlamaIndex model adapter

Contributions welcome — especially new tasks for underrepresented Indian languages.

---

## Background

Built at the intersection of production ML engineering and a genuine curiosity about India's linguistic diversity. The `hinglish_sentiment` and `indian_cultural_reasoning` tasks emerged from a simple observation: the hardest part of building AI for India isn't the language — it's the context. A model that speaks Hindi but doesn't know what Diwali is, or can't parse *"yaar ye bahut unfair hai"*, is not ready for real Indian users.

The evaluation framework is deliberately minimal — no heavy dependencies, no complex config files, graceful fallbacks throughout. The goal is that anyone with a model and a Python environment can run a meaningful Indic evaluation in under five minutes.

---

## Citation

If you use `indic-eval` in research, please cite:

```bibtex
@software{kesarwani2026indiceval,
  author    = {Kesarwani, Shivam},
  title     = {indic-eval: Evaluation Harness for LLMs on Indic Language Tasks},
  year      = {2026},
  url       = {https://github.com/shivamkesarwani-intelligencemodelling/indic-eval},
}
```

---

## Author

**Shivam Kesarwani**
IIT Delhi, Engineering Physics · Plaksha University, AI/ML

*Published in Physical Review C (2018) and European Physical Journal A (2018)*

[shivamkesarwani@alumni.iitd.ac.in](mailto:shivamkesarwani@alumni.iitd.ac.in)

---

## License

MIT — use it, fork it, build on it.

---

<div align="center">
<br/>
<sub>Built for India's AI moment. Contributions welcome.</sub>
<br/><br/>
</div>
