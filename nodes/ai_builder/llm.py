# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
Minimal chat clients. Only `requests` is used, no SDKs, so every request is easy to inspect.

Providers:
  ollama            local or Ollama Cloud (bearer key optional)  -> POST {url}/api/chat
  anthropic         Claude models                               -> POST {url}/v1/messages
  openai_compatible OpenAI, LM Studio, vLLM, llama.cpp server... -> POST {url}/v1/chat/completions
"""

import json
import os
import re
import time

import requests

from . import config


class LLMConfig(dict):
    """A dict with defaults. Passed between nodes as the GAP_AI_LLM type."""

    DEFAULTS = {
        "provider": config.DEFAULT_PROVIDER,
        "url": config.DEFAULT_OLLAMA_URL,
        "model": config.DEFAULT_CODE_MODEL,
        "vision_model": config.DEFAULT_VISION_MODEL,
        "embed_model": config.DEFAULT_EMBED_MODEL,
        "api_key": "",
        "temperature": config.DEFAULT_TEMPERATURE,
        "num_ctx": config.DEFAULT_NUM_CTX,
        "max_tokens": config.DEFAULT_MAX_TOKENS,
        "keep_alive": config.DEFAULT_KEEP_ALIVE,
        "timeout": config.DEFAULT_TIMEOUT,
    }

    def __init__(self, **kwargs):
        super().__init__(self.DEFAULTS)
        self.update({k: v for k, v in kwargs.items() if v is not None})

    def describe(self):
        return f"{self['provider']} | model={self['model']} | vision={self['vision_model']} | url={self['url']}"


def resolve_api_key(provider, explicit_key=""):
    """Explicit node input > OS keyring (API Key Manager) > environment variable."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    names = {
        "anthropic": (["Anthropic", "Claude"], "ANTHROPIC_API_KEY"),
        "openai_compatible": (["OpenAI"], "OPENAI_API_KEY"),
        "ollama": (["Ollama Cloud", "Ollama"], "OLLAMA_API_KEY"),
    }.get(provider, ([], ""))
    try:
        from ..geekatplay_key_manager import get_key  # type: ignore
        for name in names[0]:
            value = get_key(name)
            if value:
                return value
    except Exception:
        pass
    return os.environ.get(names[1], "") if names[1] else ""


def strip_thinking(text):
    """Remove <think>...</think> blocks emitted by reasoning models (qwen3, deepseek-r1...)."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


def extract_code(text):
    """Return Python code from a model reply. Prefers ```python fences; falls back to the whole text."""
    text = strip_thinking(text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return "\n\n".join(b.strip("\n") for b in blocks).strip() + "\n"
    if "import bpy" in text or "bpy." in text:
        return text.strip() + "\n"
    return ""


def extract_json(text):
    """Parse the first JSON object/array in a reply. Returns None when nothing parses."""
    text = strip_thinking(text)
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    for cand in candidates:
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = cand.find(opener), cand.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(cand[start:end + 1])
                except json.JSONDecodeError:
                    continue
    return None


class LLMClient:
    def __init__(self, cfg):
        self.cfg = cfg if isinstance(cfg, LLMConfig) else LLMConfig(**(cfg or {}))
        self.api_key = resolve_api_key(self.cfg["provider"], self.cfg.get("api_key", ""))
        self.last_request = None
        self.last_raw = None

    # ------------------------------------------------------------------ public
    # Transient failures worth retrying. A local Ollama routinely drops the connection while it
    # unloads a 27B vision model and loads a 32B coder on the same GPU; that is not a reason to
    # throw away a fifteen-minute build.
    RETRYABLE = (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError,
                 requests.exceptions.ReadTimeout)
    RETRY_DELAYS = (5, 15, 40)

    def chat(self, messages, images=None, json_mode=False, model=None, temperature=None, max_tokens=None):
        """
        messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
        images:   list of base64 PNG strings attached to the LAST user message.
        Returns the assistant text (thinking tags stripped).
        """
        provider = self.cfg["provider"]
        model = model or self.cfg["model"]
        temperature = self.cfg["temperature"] if temperature is None else temperature
        max_tokens = max_tokens or self.cfg["max_tokens"]
        send = {"ollama": self._chat_ollama, "anthropic": self._chat_anthropic,
                "openai_compatible": self._chat_openai}.get(provider)
        if send is None:
            raise ValueError(f"Unknown provider '{provider}'")
        last_error = None
        for attempt, delay in enumerate((0,) + self.RETRY_DELAYS):
            if delay:
                print(f"[AI Scene Builder] {provider} connection dropped ({type(last_error).__name__}); "
                      f"waiting {delay}s for the server to settle, retry {attempt}/{len(self.RETRY_DELAYS)}",
                      flush=True)
                time.sleep(delay)
                if provider == "ollama" and not self._ollama_up():
                    continue
            try:
                return strip_thinking(send(messages, images, json_mode, model, temperature, max_tokens))
            except requests.exceptions.HTTPError:
                raise                       # a real 4xx/5xx answer; retrying will not change it
            except self.RETRYABLE as e:
                last_error = e
        raise RuntimeError(
            f"{provider} kept dropping the connection ({type(last_error).__name__}: {last_error}). "
            "For Ollama this usually means the model does not fit in VRAM alongside the previous one - "
            "check 'ollama ps', free memory, or pick a smaller model in AI Model Config.") from last_error

    def _ollama_up(self):
        try:
            requests.get(f"{self.cfg['url'].rstrip('/')}/api/tags", headers=self._headers(), timeout=(3, 10)).raise_for_status()
            return True
        except Exception:
            return False

    def embed(self, texts):
        """Embeddings via Ollama only (other providers return None -> BM25 fallback)."""
        if self.cfg["provider"] != "ollama" or not self.cfg.get("embed_model"):
            return None
        try:
            r = requests.post(
                f"{self.cfg['url'].rstrip('/')}/api/embed",
                json={"model": self.cfg["embed_model"], "input": list(texts)},
                headers=self._headers(),
                timeout=self.cfg["timeout"],
            )
            r.raise_for_status()
            return r.json().get("embeddings")
        except Exception as e:
            print(f"[AI Scene Builder] Embeddings unavailable ({e}); using keyword retrieval.")
            return None

    def list_ollama_models(self):
        try:
            r = requests.get(f"{self.cfg['url'].rstrip('/')}/api/tags", headers=self._headers(), timeout=(5, 30))
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def pick_best_models(self, details=None):
        """Choose the strongest installed Ollama model for each job.

        Model size is the biggest quality factor in this pipeline, and the best model is rarely the
        one a workflow file happens to name - a tag like "qwen3.8:latest" hides a 27B model behind
        an unremarkable name. Ranking by what Ollama actually reports beats hard-coding names.

        Returns {"code": name|None, "vision": name|None, "embed": name|None, "reasons": {...}}.
        """
        details = details if details is not None else self.ollama_model_details()
        if not details:
            return {"code": None, "vision": None, "embed": None, "reasons": {}}

        def size(name):
            return details[name].get("params_b") or 0.0

        def caps(name):
            return set(details[name].get("capabilities") or [])

        # Vision: must actually be able to see. Bigger reads the reference far more carefully.
        vision = [n for n in details if "vision" in caps(n)]
        best_vision = max(vision, key=size) if vision else None

        # Code: prefer a purpose-built coder, then general models, biggest first. Base/instruct-less
        # variants ("-base") only autocomplete and cannot follow instructions, so they are excluded.
        usable = [n for n in details if "embedding" not in caps(n) and "-base" not in n.lower()]
        coders = [n for n in usable if "coder" in n.lower() or "code" in n.lower()]
        best_code = max(coders, key=size) if coders else (max(usable, key=size) if usable else None)

        embed = [n for n in details if "embedding" in caps(n)]
        best_embed = max(embed, key=size) if embed else None

        reasons = {}
        if best_vision:
            reasons["vision"] = f"{best_vision} ({size(best_vision):g}B, largest model with vision)"
        if best_code:
            kind = "purpose-built coder" if best_code in coders else "largest general model"
            reasons["code"] = f"{best_code} ({size(best_code):g}B, {kind})"
        if best_embed:
            reasons["embed"] = f"{best_embed}"
        return {"code": best_code, "vision": best_vision, "embed": best_embed, "reasons": reasons}

    def model_has_vision(self, model=None):
        """Can this model look at an image? Anthropic/OpenAI chat models can; for Ollama we ask the
        server (capabilities list) and cache the answer. Unknown -> False, so a blind coder is never
        sent pictures it would silently ignore."""
        model = model or self.cfg["model"]
        if self.cfg["provider"] in ("anthropic", "openai_compatible"):
            return True
        cache = getattr(self, "_vision_cache", None)
        if cache is None:
            cache = self._vision_cache = {}
        if model not in cache:
            details = self.ollama_model_details()
            info = details.get(model) or details.get(model + ":latest") or {}
            cache[model] = "vision" in set(info.get("capabilities") or [])
        return cache[model]

    def ollama_model_details(self):
        """{name: {"params_b": float|None, "families": [...], "capabilities": [...]}}.

        Reads the real parameter count reported by Ollama, because a tag like "qwen3.8:latest" hides
        a 27B model behind a name that looks like nothing.
        """
        out = {}
        try:
            r = requests.get(f"{self.cfg['url'].rstrip('/')}/api/tags", headers=self._headers(), timeout=(5, 30))
            r.raise_for_status()
            for m in r.json().get("models", []):
                details = m.get("details") or {}
                size = str(details.get("parameter_size") or "")
                match = re.search(r"(\d+(?:\.\d+)?)\s*B", size, re.IGNORECASE)
                out[m["name"]] = {
                    "params_b": float(match.group(1)) if match else None,
                    "families": details.get("families") or [],
                    "capabilities": m.get("capabilities") or [],
                }
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------ providers
    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _chat_ollama(self, messages, images, json_mode, model, temperature, max_tokens):
        msgs = [dict(m) for m in messages]
        if images:
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["images"] = list(images)
                    break
        payload = {
            "model": model,
            "messages": msgs,
            "stream": False,
            "keep_alive": self.cfg["keep_alive"],
            "options": {"temperature": temperature, "num_ctx": int(self.cfg["num_ctx"]), "num_predict": int(max_tokens)},
        }
        if json_mode:
            payload["format"] = "json"
        self.last_request = payload
        r = requests.post(f"{self.cfg['url'].rstrip('/')}/api/chat", json=payload, headers=self._headers(), timeout=self.cfg["timeout"])
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama model '{model}' not found. Run:  ollama pull {model}   (or pick another model in the AI Model Config node)."
            )
        r.raise_for_status()
        data = r.json()
        self.last_raw = data
        return data.get("message", {}).get("content", "")

    def _chat_anthropic(self, messages, images, json_mode, model, temperature, max_tokens):
        if not self.api_key:
            raise RuntimeError("Anthropic API key missing. Save it as 'Anthropic' in the API Key Manager or set ANTHROPIC_API_KEY.")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if json_mode:
            system += "\n\nRespond with valid JSON only."
        converted = []
        for m in messages:
            if m["role"] == "system":
                continue
            content = [{"type": "text", "text": m["content"]}]
            converted.append({"role": m["role"], "content": content})
        if images and converted:
            for m in reversed(converted):
                if m["role"] == "user":
                    for b64 in images:
                        m["content"].insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
                    break
        payload = {"model": model, "max_tokens": int(max_tokens), "temperature": temperature, "messages": converted}
        if system:
            payload["system"] = system
        self.last_request = payload
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        r = requests.post(f"{self.cfg['url'].rstrip('/')}/v1/messages", json=payload, headers=headers, timeout=self.cfg["timeout"])
        r.raise_for_status()
        data = r.json()
        self.last_raw = data
        return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")

    def _chat_openai(self, messages, images, json_mode, model, temperature, max_tokens):
        converted = []
        for m in messages:
            converted.append({"role": m["role"], "content": m["content"]})
        if images:
            for m in reversed(converted):
                if m["role"] == "user":
                    parts = [{"type": "text", "text": m["content"]}]
                    for b64 in images:
                        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                    m["content"] = parts
                    break
        payload = {"model": model, "messages": converted, "temperature": temperature, "max_tokens": int(max_tokens)}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        self.last_request = payload
        r = requests.post(f"{self.cfg['url'].rstrip('/')}/v1/chat/completions", json=payload, headers=self._headers(), timeout=self.cfg["timeout"])
        r.raise_for_status()
        data = r.json()
        self.last_raw = data
        return data["choices"][0]["message"].get("content", "") or ""
