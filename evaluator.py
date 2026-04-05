import json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from models import BaseModel
from tasks import get_task, ALL_TASKS, TaskResult

@dataclass
class EvalReport:
    model_name: str
    tasks: List[TaskResult]
    total_time_s: float
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def overall_score(self):
        return sum(t.primary_score() for t in self.tasks) / len(self.tasks) if self.tasks else 0.0

    def summary(self):
        return {"model": self.model_name, "timestamp": self.timestamp,
                "total_time_s": round(self.total_time_s, 2),
                "tasks": [t.to_dict() for t in self.tasks],
                "overall_score": round(self.overall_score(), 4)}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to {path}")

    def print_table(self):
        W = 36
        print(); print("="*70)
        print(f"  indic-eval  |  Model: {self.model_name}")
        print("="*70)
        print(f"  {'Task':<{W}} {'Metric':<20} {'Score':>8}")
        print("-"*70)
        for t in self.tasks:
            m = t.metrics[0]
            lat = next((x for x in t.metrics if x.name=="latency_ms"), None)
            lat_str = f"  ({lat.score:.0f}ms)" if lat else ""
            print(f"  {t.task_name:<{W}} {m.name:<20} {m.score*100:>7.1f}%{lat_str}")
        print("-"*70)
        print(f"  {'Overall Score':<{W}} {'(mean)':<20} {self.overall_score()*100:>7.1f}%")
        print("="*70)
        print(f"  Done in {self.total_time_s:.1f}s  |  {self.timestamp}"); print()

class Evaluator:
    def __init__(self, model, tasks=None, n_samples=50, verbose=True):
        self.model = model
        self.task_names = tasks or ALL_TASKS
        self.n_samples = n_samples
        self.verbose = verbose

    def run(self):
        results = []
        t0 = time.time()
        for name in self.task_names:
            task = get_task(name)
            if self.verbose: print(f"\n▶  {name}  ({self.n_samples} samples)")
            result = task.evaluate(self.model, n=self.n_samples, verbose=self.verbose)
            results.append(result)
            if self.verbose:
                m = result.metrics[0]
                print(f"   ✓ {m.name}: {m.score*100:.1f}%")
        report = EvalReport(self.model.model_name, results, time.time()-t0)
        if self.verbose: report.print_table()
        return report
