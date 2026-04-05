"""
indic-eval CLI
==============
A command-line interface for evaluating LLMs on Indic language tasks.

Usage examples:
---------------

  # List all available tasks
  indic-eval --list-tasks

  # Run all tasks with an OpenAI-compatible API
  indic-eval --model gpt-4o-mini --api-key sk-... --all-tasks

  # Run with Sarvam AI
  indic-eval --model sarvam-2b --base-url https://api.sarvam.ai/v1 --api-key ... --all-tasks

  # Run with Ollama (local, no API key needed)
  indic-eval --model llama3 --base-url http://localhost:11434/v1 --all-tasks

  # Run with a HuggingFace model
  indic-eval --model sarvamai/sarvam-2b-v0.5 --model-type hf --device cpu

  # Run specific tasks only
  indic-eval --model gpt-4o-mini --tasks hinglish_sentiment indian_cultural_reasoning

  # Save results to a file
  indic-eval --model gpt-4o-mini --all-tasks --output results/gpt4o.json

  # Run quietly (no progress output)
  indic-eval --model gpt-4o-mini --all-tasks --quiet

  # Compare two models
  indic-eval --model gpt-4o-mini --all-tasks --output results/gpt4o.json
  indic-eval --model gpt-3.5-turbo --all-tasks --output results/gpt35.json
  indic-eval --compare results/gpt4o.json results/gpt35.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ── Colours for terminal output ────────────────────────────────────────────────

class Colour:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    WHITE  = "\033[97m"

def c(text, colour):
    """Wrap text in a colour code — auto-disabled if not a TTY."""
    if not sys.stdout.isatty():
        return text
    return f"{colour}{text}{Colour.RESET}"

def score_colour(score: float) -> str:
    if score >= 0.7:  return Colour.GREEN
    if score >= 0.4:  return Colour.YELLOW
    return Colour.RED


# ── Helpers ────────────────────────────────────────────────────────────────────

def print_banner():
    banner = r"""
  ██╗███╗   ██╗██████╗ ██╗ ██████╗    ███████╗██╗   ██╗ █████╗ ██╗
  ██║████╗  ██║██╔══██╗██║██╔════╝    ██╔════╝██║   ██║██╔══██╗██║
  ██║██╔██╗ ██║██║  ██║██║██║         █████╗  ██║   ██║███████║██║
  ██║██║╚██╗██║██║  ██║██║██║         ██╔══╝  ╚██╗ ██╔╝██╔══██║██║
  ██║██║ ╚████║██████╔╝██║╚██████╗    ███████╗ ╚████╔╝ ██║  ██║███████╗
  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
    """
    print(c(banner, Colour.CYAN))
    print(c("  Evaluation harness for LLMs on Indic language tasks\n", Colour.DIM))


def print_task_list():
    try:
        from indic_eval.tasks import list_tasks
    except ImportError as e:
        print(c(f"Error loading tasks: {e}", Colour.RED))
        sys.exit(1)

    tasks = list_tasks()
    print(c("\n  Available Tasks\n", Colour.BOLD))
    print(f"  {'#':<4} {'Name':<38} {'Language':<12} Description")
    print(c("  " + "─" * 78, Colour.DIM))

    lang_colours = {"hi": Colour.YELLOW, "hi-en": Colour.CYAN, "en": Colour.BLUE}

    for i, t in enumerate(tasks, 1):
        lang = t["language"]
        lang_col = lang_colours.get(lang, Colour.WHITE)
        print(
            f"  {c(str(i), Colour.DIM):<4} "
            f"{c(t['name'], Colour.WHITE):<38} "
            f"{c(lang, lang_col):<12} "
            f"{c(t['description'], Colour.DIM)}"
        )

    print(c("\n  " + "─" * 78, Colour.DIM))
    print(f"  {c(str(len(tasks)) + ' tasks total', Colour.DIM)}\n")


def print_compare(paths: list):
    reports = []
    for p in paths:
        if not Path(p).exists():
            print(c(f"File not found: {p}", Colour.RED))
            sys.exit(1)
        with open(p, encoding="utf-8") as f:
            reports.append(json.load(f))

    all_tasks = []
    for r in reports:
        for t in r.get("tasks", []):
            if t["task"] not in all_tasks:
                all_tasks.append(t["task"])

    def get_score(report, task_name):
        for t in report.get("tasks", []):
            if t["task"] == task_name:
                scores = list(t["metrics"].values())
                return scores[0] if scores else None
        return None

    col_w = 10
    name_w = 36

    print(c("\n  Model Comparison\n", Colour.BOLD))

    header = f"  {'Task':<{name_w}}"
    for r in reports:
        model = r.get("model", "unknown")[:col_w - 1]
        header += f"  {c(model, Colour.CYAN):>{col_w}}"
    print(header)
    print(c("  " + "─" * (name_w + (col_w + 2) * len(reports)), Colour.DIM))

    for task_name in all_tasks:
        row = f"  {task_name:<{name_w}}"
        scores = [get_score(r, task_name) for r in reports]
        best = max((s for s in scores if s is not None), default=0)
        for s in scores:
            if s is None:
                row += f"  {'N/A':>{col_w}}"
            else:
                pct = f"{s*100:.1f}%"
                col = Colour.GREEN if s == best else Colour.WHITE
                row += f"  {c(pct, col):>{col_w}}"
        print(row)

    print(c("  " + "─" * (name_w + (col_w + 2) * len(reports)), Colour.DIM))

    overall_row = f"  {'Overall Score':<{name_w}}"
    overalls = [r.get("overall_score") for r in reports]
    best_overall = max((s for s in overalls if s is not None), default=0)
    for s in overalls:
        if s is None:
            overall_row += f"  {'N/A':>{col_w}}"
        else:
            pct = f"{s*100:.1f}%"
            col = Colour.GREEN if s == best_overall else Colour.WHITE
            overall_row += f"  {c(pct, col):>{col_w}}"
    print(c(overall_row, Colour.BOLD))
    print()


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indic-eval",
        description="Evaluation harness for LLMs on Indic language tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=False,
    )

    misc = parser.add_argument_group("General")
    misc.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
    misc.add_argument("--version", action="store_true", help="Show version and exit")
    misc.add_argument("--list-tasks", action="store_true", help="List all available tasks and exit")
    misc.add_argument("--compare", nargs="+", metavar="RESULTS_JSON",
                      help="Compare two or more saved result JSON files")

    model_g = parser.add_argument_group("Model")
    model_g.add_argument("--model", "-m", metavar="MODEL_NAME", help="Model name or HuggingFace model ID")
    model_g.add_argument("--model-type", choices=["api", "hf"], default="api", metavar="TYPE",
                         help="Backend: 'api' (OpenAI-compatible) or 'hf' (HuggingFace). Default: api")
    model_g.add_argument("--base-url", default="https://api.openai.com/v1", metavar="URL",
                         help="API base URL. Change for Sarvam, Ollama, Groq, etc.")
    model_g.add_argument("--api-key", default=None, metavar="KEY",
                         help="API key (falls back to OPENAI_API_KEY env var)")
    model_g.add_argument("--device", default="cpu", metavar="DEVICE",
                         help="Device for HF models: cpu, cuda, mps. Default: cpu")
    model_g.add_argument("--load-in-8bit", action="store_true",
                         help="Load HF model in 8-bit quantization (requires bitsandbytes)")
    model_g.add_argument("--temperature", type=float, default=0.0, metavar="FLOAT",
                         help="Sampling temperature. Default: 0.0 (greedy)")

    task_g = parser.add_argument_group("Tasks")
    task_g.add_argument("--tasks", "-t", nargs="+", metavar="TASK",
                        help="One or more task names to evaluate on")
    task_g.add_argument("--all-tasks", action="store_true", help="Run all available tasks")
    task_g.add_argument("--n-samples", "-n", type=int, default=20, metavar="N",
                        help="Samples per task. Default: 20")

    out_g = parser.add_argument_group("Output")
    out_g.add_argument("--output", "-o", default=None, metavar="PATH",
                       help="Save results to a JSON file")
    out_g.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress output (still prints final table)")
    out_g.add_argument("--no-table", action="store_true", help="Suppress the final results table")
    out_g.add_argument("--no-colour", action="store_true", help="Disable coloured output")

    return parser


def validate_tasks(task_names: list) -> list:
    try:
        from indic_eval.tasks import ALL_TASKS
    except ImportError as e:
        print(c(f"\nError importing tasks: {e}\n", Colour.RED))
        sys.exit(1)

    invalid = [t for t in task_names if t not in ALL_TASKS]
    if invalid:
        print(c(f"\nUnknown task(s): {', '.join(invalid)}", Colour.RED))
        print(c(f"Available: {', '.join(ALL_TASKS)}\n", Colour.DIM))
        print(f"Run {c('indic-eval --list-tasks', Colour.CYAN)} to see all tasks.")
        sys.exit(1)
    return task_names


def _print_coloured_table(report):
    W = 36
    print(c("  " + "═" * 68, Colour.DIM))
    print(f"  {c('indic-eval', Colour.CYAN)}  {c('│', Colour.DIM)}  "
          f"{c('Model:', Colour.DIM)} {c(report.model_name, Colour.WHITE)}")
    print(c("  " + "═" * 68, Colour.DIM))
    print(f"  {c('Task', Colour.DIM):<{W+9}} {c('Metric', Colour.DIM):<28} {c('Score', Colour.DIM):>8}")
    print(c("  " + "─" * 68, Colour.DIM))

    for t in report.tasks:
        m = t.metrics[0]
        lat = next((x for x in t.metrics if x.name == "latency_ms"), None)
        lat_str = c(f"  ({lat.score:.0f}ms)", Colour.DIM) if lat else ""
        score_str = c(f"{m.score*100:.1f}%", score_colour(m.score))
        print(f"  {t.task_name:<{W}} {c(m.name, Colour.DIM):<20} {score_str:>8}{lat_str}")

    print(c("  " + "─" * 68, Colour.DIM))
    overall = report.overall_score()
    overall_str = c(f"{overall*100:.1f}%", score_colour(overall))
    print(f"  {c('Overall Score', Colour.BOLD):<{W}} {c('(mean)', Colour.DIM):<20} {overall_str:>8}")
    print(c("  " + "═" * 68, Colour.DIM))
    print(f"  {c(f'Completed in {report.total_time_s:.1f}s', Colour.DIM)}  "
          f"{c('│', Colour.DIM)}  {c(report.timestamp, Colour.DIM)}\n")


def run_evaluation(args):
    try:
        from indic_eval.models import load_model
        from indic_eval.tasks import get_task, ALL_TASKS
        from indic_eval.evaluator import EvalReport
    except ImportError as e:
        print(c(f"\nImport error: {e}", Colour.RED))
        print(c("Make sure you've run: pip install -e .\n", Colour.DIM))
        sys.exit(1)

    if args.all_tasks:
        task_names = ALL_TASKS
    elif args.tasks:
        task_names = validate_tasks(args.tasks)
    else:
        task_names = ALL_TASKS

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    if args.model_type == "api" and not api_key:
        if "localhost" not in args.base_url and "127.0.0.1" not in args.base_url:
            print(c("\nWarning: No API key set. Use --api-key or export OPENAI_API_KEY.\n", Colour.YELLOW))

    model_config = {
        "type":         args.model_type,
        "model":        args.model,
        "base_url":     args.base_url,
        "api_key":      api_key,
        "temperature":  args.temperature,
        "device":       args.device,
        "load_in_8bit": args.load_in_8bit,
    }

    if not args.quiet:
        backend_str = (
            f"API  →  {args.base_url}" if args.model_type == "api"
            else f"HuggingFace  →  {args.device}"
        )
        print(f"  {c('Model:', Colour.DIM)}    {c(args.model, Colour.WHITE)}")
        print(f"  {c('Backend:', Colour.DIM)}  {c(backend_str, Colour.DIM)}")
        print(f"  {c('Tasks:', Colour.DIM)}    {c(str(len(task_names)), Colour.WHITE)}  "
              f"{c('(' + ', '.join(task_names) + ')', Colour.DIM)}")
        print(f"  {c('Samples:', Colour.DIM)}  {c(str(args.n_samples) + ' per task', Colour.DIM)}\n")

    try:
        model = load_model(model_config)
    except Exception as e:
        print(c(f"\nFailed to load model: {e}\n", Colour.RED))
        sys.exit(1)

    results = []
    total_start = time.time()

    for i, task_name in enumerate(task_names):
        task = get_task(task_name)

        if not args.quiet:
            print(f"  {c(f'[{i+1}/{len(task_names)}]', Colour.DIM)} {c('▶', Colour.CYAN)} {task_name}")

        try:
            task_result = task.evaluate(model, n=args.n_samples, verbose=False)
            results.append(task_result)

            if not args.quiet:
                m = task_result.metrics[0]
                lat = next((x for x in task_result.metrics if x.name == "latency_ms"), None)
                lat_str = c(f"  ({lat.score:.0f}ms avg)", Colour.DIM) if lat else ""
                score_str = c(f"{m.score*100:.1f}%", score_colour(m.score))
                print(f"       {c('✓', Colour.GREEN)} {m.name}: {score_str}{lat_str}")

        except KeyboardInterrupt:
            print(c("\n\nInterrupted. Saving partial results...", Colour.YELLOW))
            break
        except Exception as e:
            print(c(f"       {c('✗', Colour.RED)} Failed: {e}", Colour.RED))
            continue

    if not results:
        print(c("\nNo tasks completed successfully.\n", Colour.RED))
        sys.exit(1)

    report = EvalReport(model_name=args.model, tasks=results, total_time_s=time.time() - total_start)

    if not args.no_table:
        print()
        _print_coloured_table(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        report.save(args.output)
        print(f"  {c('💾  Saved →', Colour.GREEN)} {c(args.output, Colour.WHITE)}\n")

    return report


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.no_colour:
        for attr in vars(Colour):
            if not attr.startswith("_"):
                setattr(Colour, attr, "")

    if args.help:
        print_banner()
        parser.print_help()
        sys.exit(0)

    if args.version:
        try:
            from indic_eval import __version__
            print(f"indic-eval {__version__}")
        except ImportError:
            print("indic-eval (version unknown)")
        sys.exit(0)

    if args.list_tasks:
        print_banner()
        print_task_list()
        sys.exit(0)

    if args.compare:
        print_banner()
        print_compare(args.compare)
        sys.exit(0)

    if not args.model:
        print_banner()
        print(c("  Error: --model is required.\n", Colour.RED))
        print("  Examples:")
        print(f"    {c('indic-eval --model gpt-4o-mini --all-tasks', Colour.CYAN)}")
        print(f"    {c('indic-eval --model sarvam-2b --base-url https://api.sarvam.ai/v1 --api-key ...', Colour.CYAN)}")
        print(f"    {c('indic-eval --model sarvamai/sarvam-2b-v0.5 --model-type hf', Colour.CYAN)}")
        print(f"\n  Run {c('indic-eval --list-tasks', Colour.CYAN)} to see all tasks.\n")
        sys.exit(1)

    if not args.quiet:
        print_banner()

    run_evaluation(args)


if __name__ == "__main__":
    main()
