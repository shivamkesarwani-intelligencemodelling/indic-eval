from __future__ import annotations
import re, time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
from indic-eval.metrics import (exact_match, token_f1, bleu, chrf,
                                 rouge_l, mcq_accuracy, latency_stats, MetricResult)
from indic-eval.models import BaseModel, ModelResponse

@dataclass
class TaskResult:
    task_name: str
    metrics: List[MetricResult]
    predictions: List[str]
    references: List[str]
    latencies_ms: List[float]
    n_samples: int

    def primary_score(self):
        return self.metrics[0].score if self.metrics else 0.0

    def to_dict(self):
        return {"task": self.task_name, "n_samples": self.n_samples,
                "metrics": {m.name: round(m.score, 4) for m in self.metrics},
                "latency": {"mean_ms": round(sum(self.latencies_ms)/max(len(self.latencies_ms),1),1)}}

class BaseTask(ABC):
    name = "base"; description = ""; language = "hi"

    @abstractmethod
    def load_samples(self, n=50): pass

    @abstractmethod
    def parse_output(self, response): pass

    @abstractmethod
    def compute_metrics(self, predictions, references, latencies): pass

    def evaluate(self, model, n=50, verbose=False):
        samples = self.load_samples(n)
        predictions, references, latencies = [], [], []
        for i, s in enumerate(samples):
            if verbose: print(f"  [{i+1}/{len(samples)}] {self.name}", end="\r")
            resp = model.generate(s["prompt"], max_tokens=256)
            predictions.append(self.parse_output(resp))
            references.append(s["reference"])
            latencies.append(resp.latency_ms)
        metrics = self.compute_metrics(predictions, references, latencies)
        return TaskResult(self.name, metrics, predictions, references, latencies, len(samples))


# ── Task 1: Hindi Reading Comprehension ───────────────────────────
class HindiReadingComprehension(BaseTask):
    name = "hindi_reading_comprehension"
    description = "Reading comprehension in Hindi (IndicQA style)"
    language = "hi"
    PROMPT = """Read the following Hindi passage and answer the question.
Give only the answer, nothing else.

Passage: {context}

Question: {question}

Answer:"""

    def load_samples(self, n=50):
        try:
            from datasets import load_dataset
            ds = load_dataset("ai4bharat/IndicQA","IndicQA.hi",split="test",trust_remote_code=True)
            out = []
            for row in ds.select(range(min(n, len(ds)))):
                if row["answers"]["text"]:
                    out.append({"prompt": self.PROMPT.format(context=row["context"],
                                                              question=row["question"]),
                                "reference": row["answers"]["text"][0]})
            return out[:n]
        except Exception:
            return self._fallback(n)

    def _fallback(self, n):
        s = [
            {"prompt": self.PROMPT.format(
                context="महात्मा गांधी का जन्म 2 अक्टूबर 1869 को पोरबंदर, गुजरात में हुआ था।",
                question="महात्मा गांधी का जन्म कब हुआ था?"),
             "reference": "2 अक्टूबर 1869"},
            {"prompt": self.PROMPT.format(
                context="ताजमहल आगरा में स्थित है। इसे शाहजहाँ ने बनवाया था।",
                question="ताजमहल का निर्माण किसने करवाया?"),
             "reference": "शाहजहाँ"},
            {"prompt": self.PROMPT.format(
                context="भारत का राष्ट्रीय पक्षी मोर है। इसे 1963 में घोषित किया गया।",
                question="मोर को राष्ट्रीय पक्षी कब घोषित किया गया?"),
             "reference": "1963"},
        ]
        return (s * ((n // len(s)) + 1))[:n]

    def parse_output(self, r): return r.text.strip().split("\n")[0]
    def compute_metrics(self, p, r, l):
        return [exact_match(p, r), token_f1(p, r), latency_stats(l)]

# ── Task 2: English → Hindi Translation ───────────────────────────
class EnglishToHindiTranslation(BaseTask):
    name = "en_hi_translation"
    description = "English → Hindi translation (FLORES-200)"
    language = "hi"
    PROMPT = """Translate the following English sentence to Hindi.
Give only the Hindi translation, nothing else.

English: {source}

Hindi:"""

    def load_samples(self, n=50):
        try:
            from datasets import load_dataset
            ds = load_dataset("facebook/flores","eng_Latn-hin_Deva",split="devtest",trust_remote_code=True)
            return [{"prompt": self.PROMPT.format(source=row["sentence_eng_Latn"]),
                     "reference": row["sentence_hin_Deva"]}
                    for row in ds.select(range(min(n, len(ds))))]
        except Exception:
            return self._fallback(n)

    def _fallback(self, n):
        s = [
            {"prompt": self.PROMPT.format(source="India is a diverse country with many languages."),
             "reference": "भारत एक विविध देश है जिसमें कई भाषाएँ हैं।"},
            {"prompt": self.PROMPT.format(source="The train arrives at eight in the morning."),
             "reference": "ट्रेन सुबह आठ बजे पहुँचती है।"},
            {"prompt": self.PROMPT.format(source="She is studying artificial intelligence."),
             "reference": "वह कृत्रिम बुद्धिमत्ता का अध्ययन कर रही है।"},
        ]
        return (s * ((n // len(s)) + 1))[:n]

    def parse_output(self, r): return r.text.strip().split("\n")[0]
    def compute_metrics(self, p, r, l): return [bleu(p, r), chrf(p, r), latency_stats(l)]


# ── Task 3: Hinglish Sentiment ─────────────────────────────────────
class HinglishSentiment(BaseTask):
    name = "hinglish_sentiment"
    description = "Sentiment classification on Hinglish (code-switched) text"
    language = "hi-en"
    PROMPT = """Classify the sentiment of this Hinglish text as Positive, Negative, or Neutral.
Give only one word: Positive, Negative, or Neutral.

Text: {text}

Sentiment:"""
    SAMPLES = [
        ("Yaar ye movie bilkul bakwaas thi, time waste ho gaya.", "Negative"),
        ("Aaj ka din bahut achha tha! Sab kuch perfect raha.", "Positive"),
        ("Theek hai, kuch khaas nahi tha.", "Neutral"),
        ("Itna bura khana maine kabhi nahi khaya!", "Negative"),
        ("Bhai tera code bahut clean hai, mast kaam!", "Positive"),
        ("Train late thi as usual, koi surprise nahi.", "Negative"),
        ("Dekha nahi abhi, baad mein dekhenge.", "Neutral"),
        ("Wow yaar, ye jagah ekdum fantastic hai!", "Positive"),
        ("Service theek thi, na zyada acchi na buri.", "Neutral"),
        ("Phir se bijli gayi! Kab sudhrega ye system?", "Negative"),
    ]

    def load_samples(self, n=50):
        s = [{"prompt": self.PROMPT.format(text=t), "reference": lb} for t, lb in self.SAMPLES]
        return (s * ((n // len(s)) + 1))[:n]

    def parse_output(self, r):
        text = r.text.strip().split("\n")[0].capitalize()
        for lb in ["Positive", "Negative", "Neutral"]:
            if lb.lower() in text.lower(): return lb
        return text

    def compute_metrics(self, p, r, l): return [exact_match(p, r), latency_stats(l)]


# ── Task 4: Indian Cultural Reasoning ─────────────────────────────
class IndianCulturalReasoning(BaseTask):
    name = "indian_cultural_reasoning"
    description = "MCQ on Indian culture, history, geography, society"
    language = "hi"
    PROMPT = """Answer this multiple choice question about India. Give only the letter (A, B, C, or D).

Question: {question}
A) {a}
B) {b}
C) {c}
D) {d}

Answer:"""
    SAMPLES = [
        ("Which classical dance form originated in Kerala?",
         "Bharatanatyam","Kathakali","Odissi","Manipuri","B"),
        ("The festival of Onam is celebrated in which state?",
         "Tamil Nadu","Karnataka","Kerala","Andhra Pradesh","C"),
        ("Who wrote Jana Gana Mana?",
         "Bankim Chandra","Rabindranath Tagore","Sarojini Naidu","Subramania Bharati","B"),
        ("Which river is holiest in Hinduism?",
         "Yamuna","Saraswati","Ganga","Godavari","C"),
        ("Ajanta Caves are located in which state?",
         "Rajasthan","Madhya Pradesh","Maharashtra","Gujarat","C"),
        ("Pongal is the harvest festival of which community?",
         "Punjabis","Tamils","Bengalis","Gujaratis","B"),
        ("Largest state in India by area?",
         "Maharashtra","Uttar Pradesh","Madhya Pradesh","Rajasthan","D"),
        ("Indian Constitution was adopted on?",
         "15 Aug 1947","26 Jan 1950","26 Nov 1949","2 Oct 1950","C"),
        ("Which instrument is Ravi Shankar known for?",
         "Tabla","Sarod","Sitar","Veena","C"),
        ("Which city is famous for Biryani in Mughal history?",
         "Delhi","Agra","Hyderabad","Lucknow","C"),
    ]

    def load_samples(self, n=50):
        s = [{"prompt": self.PROMPT.format(question=q,a=a,b=b,c=c,d=d),
              "reference": ans} for q,a,b,c,d,ans in self.SAMPLES]
        return (s * ((n // len(s)) + 1))[:n]

    def parse_output(self, r): return r.text.strip()
    def compute_metrics(self, p, r, l): return [mcq_accuracy(p, r), latency_stats(l)]


# ── Task 5: Hindi Summarisation ────────────────────────────────────
class HindiSummarisation(BaseTask):
    name = "hindi_summarisation"
    description = "Abstractive summarisation of Hindi news text"
    language = "hi"
    PROMPT = """Summarise the following Hindi news article in 1-2 sentences.
Give only the summary in Hindi.

Article: {article}

Summary:"""
    SAMPLES = [
        ("नई दिल्ली: इसरो ने आज सफलतापूर्वक एक नया उपग्रह लॉन्च किया। यह उपग्रह संचार सेवाओं को बेहतर बनाएगा।",
         "इसरो ने एक नया संचार उपग्रह सफलतापूर्वक लॉन्च किया।"),
        ("मुंबई: वैश्विक कमजोरी के कारण सेंसेक्स में आज 500 अंकों की गिरावट आई। विशेषज्ञों का कहना है कि अनिश्चितता जारी रहेगी।",
         "वैश्विक कमजोरी से सेंसेक्स 500 अंक गिरा।"),
    ]

    def load_samples(self, n=50):
        s = [{"prompt": self.PROMPT.format(article=a), "reference": r} for a,r in self.SAMPLES]
        return (s * ((n // len(s)) + 1))[:n]

    def parse_output(self, r): return r.text.strip().split("\n")[0]
    def compute_metrics(self, p, r, l): return [rouge_l(p, r), chrf(p, r), latency_stats(l)]


# ── Registry ────────────────────────────────────────────────────────
TASK_REGISTRY = {
    "hindi_reading_comprehension": HindiReadingComprehension,
    "en_hi_translation":           EnglishToHindiTranslation,
    "hinglish_sentiment":          HinglishSentiment,
    "indian_cultural_reasoning":   IndianCulturalReasoning,
    "hindi_summarisation":         HindiSummarisation,
}
ALL_TASKS = list(TASK_REGISTRY.keys())

def get_task(name):
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: '{name}'. Available: {ALL_TASKS}")
    return TASK_REGISTRY[name]()

def list_tasks():
    return [{"name": cls().name, "description": cls().description, "language": cls().language}
            for cls in TASK_REGISTRY.values()]
