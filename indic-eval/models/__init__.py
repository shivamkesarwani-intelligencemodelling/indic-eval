import os, time
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelResponse:
    text: str
    latency_ms: float
    tokens_used: Optional[int] = None

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt, max_tokens=256): pass

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"

class APIModel(BaseModel):
    """OpenAI-compatible API — works with Sarvam AI, GPT, Ollama, etc."""
    def __init__(self, model_name, base_url="https://api.openai.com/v1",
                 api_key=None, temperature=0.0):
        super().__init__(model_name)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature

    def generate(self, prompt, max_tokens=256):
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        t0 = time.time()
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=self.temperature,
        )
        latency = (time.time() - t0) * 1000
        text = resp.choices[0].message.content or ""
        return ModelResponse(text=text.strip(), latency_ms=latency,
                             tokens_used=getattr(resp.usage, "total_tokens", None))

class HuggingFaceModel(BaseModel):
    """HuggingFace transformers — load any causal LM from the Hub."""
    def __init__(self, model_name, device="cpu", torch_dtype="auto", load_in_8bit=False):
        super().__init__(model_name)
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self._pipeline = None

    def _load(self):
        if self._pipeline: return
        from transformers import pipeline
        self._pipeline = pipeline(
            "text-generation", model=self.model_name,
            device=self.device if not self.load_in_8bit else None,
            torch_dtype=self.torch_dtype, load_in_8bit=self.load_in_8bit,
        )

    def generate(self, prompt, max_tokens=256):
        self._load()
        t0 = time.time()
        out = self._pipeline(prompt, max_new_tokens=max_tokens,
                             do_sample=False, return_full_text=False)
        return ModelResponse(text=out[0]["generated_text"].strip(),
                             latency_ms=(time.time()-t0)*1000)

def load_model(config):
    t = config.get("type", "api").lower()
    if t == "api":
        return APIModel(config["model"], config.get("base_url","https://api.openai.com/v1"),
                        config.get("api_key"), config.get("temperature", 0.0))
    elif t in ("hf", "huggingface"):
        return HuggingFaceModel(config["model"], config.get("device","cpu"),
                                config.get("torch_dtype","auto"), config.get("load_in_8bit",False))
    raise ValueError(f"Unknown type: {t}")
