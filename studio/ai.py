"""Talking to the AI service. You bring your own key, and choose the service and model in the dashboard
(Settings, AI) or in .env. What is saved from the dashboard is kept in ai.json in the data folder and wins.

    PEAKCUT_AI_PROVIDER   gemini | openai | xai | groq | openrouter | anthropic | ollama | custom
    PEAKCUT_AI_MODELS     one or more model ids, comma separated, tried in this order
    PEAKCUT_AI_KEY        your key (or PEAKCUT_AI_KEY_FILE, a file holding it)
    PEAKCUT_AI_BASE_URL   only for "custom" (any service that speaks the OpenAI chat format), or to override

Every service is asked for the same thing: JSON that follows a small schema. Gemini takes the schema directly.
The others are told the schema in the prompt and asked for a JSON object, which works with any chat model."""
import base64
import json
import os
import re
import threading
import time

import httpx

from . import config

# name -> (label shown on the pages, address, how it is spoken to, where to get a key, an example model id)
PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "gemini", "https://aistudio.google.com/apikey", "gemini-2.5-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "openai", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "anthropic", "https://console.anthropic.com", ""),
    "ollama": ("Ollama (on this machine)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", "llama3.1"),
    "custom": ("Custom (OpenAI-compatible)", "", "openai", "", ""),
}
ALIASES = {"google": "gemini", "grok": "xai", "claude": "anthropic", "local": "ollama"}
NO_KEY = {"ollama", "custom"}       # a model running on your own machine needs no key

# ---------------------------------------------------------------- the settings in use
_lock = threading.Lock()
_saved = {"at": None, "data": {}}


def _file():
    return config.DATA_DIR / "ai.json"


def _from_dashboard() -> dict:
    """What was saved from the dashboard, read again only when the file changes."""
    f = _file()
    try:
        at = f.stat().st_mtime_ns
    except OSError:
        at = 0
    if _saved["at"] != at:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if at else {}
        except (OSError, ValueError):
            data = {}
        _saved.update(at=at, data=data if isinstance(data, dict) else {})
    return _saved["data"]


def current() -> dict:
    """Provider, models, address and key: what the dashboard saved, or else what .env says."""
    d = _from_dashboard()
    p = str(d.get("provider") or config.AI_PROVIDER or "").strip().lower()
    final = d.get("final_models") if d.get("provider") else [m.strip() for m in os.environ.get("PEAKCUT_AI_FINAL_MODELS", "").split(",")]
    video = d.get("video_models") if d.get("provider") else [m.strip() for m in os.environ.get("PEAKCUT_AI_VIDEO_MODELS", "").split(",")]
    return {"provider": ALIASES.get(p, p),
            "models": list(d["models"]) if d.get("provider") else list(config.AI_MODELS),
            "final_models": [str(m) for m in (final or []) if str(m).strip()],
            "video_models": [str(m) for m in (video or []) if str(m).strip()],
            "base_url": (str(d.get("base_url") or "") if d.get("provider") else config.AI_BASE_URL).rstrip("/"),
            "key": str(d["key"]) if d.get("provider") and "key" in d else config.ai_key()}


def provider() -> str:
    return current()["provider"]


def models(level="scan") -> list:
    """The models to try, in order. level "final" is for the last decisions (choosing the best clips, checking each
    story is whole); if no separate final models were chosen, it is the same list as the scan models."""
    c = current()
    if level == "video":               # the model that looks at pictures of the video; none chosen means none used
        return c["video_models"]
    return c["final_models"] if level == "final" and c["final_models"] else c["models"]


def key() -> str:
    return current()["key"]


def label() -> str:
    return PROVIDERS.get(provider(), ("",))[0]


def base_url() -> str:
    c = current()
    return c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]


def ready() -> bool:
    c = current()
    return bool(c["provider"] in PROVIDERS and c["models"] and base_url() and (c["key"] or c["provider"] in NO_KEY))


def public() -> dict:
    """What the settings page may see. The key itself never leaves the server: only whether there is one, and
    its last four characters so the owner can tell which key it is."""
    c = current()
    return {"provider": c["provider"], "models": c["models"], "final_models": c["final_models"], "video_models": c["video_models"], "base_url": c["base_url"], "has_key": bool(c["key"]),
            "key_hint": c["key"][-4:] if len(c["key"]) >= 12 else "", "ready": ready(), "label": label(),
            "saved_here": bool(_from_dashboard().get("provider")),
            "providers": [{"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3], "example": v[4]}
                          for k, v in PROVIDERS.items()]}


def _model_list(ids) -> list:
    if isinstance(ids, str):
        ids = ids.split(",")
    ms = [str(m).strip()[:120] for m in (ids or []) if str(m).strip()][:5]
    if any(not re.fullmatch(r"[\w.\-:/@]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / only.")
    return ms


def save(provider_id: str, model_ids, address: str = "", new_key=None, final_ids=None, video_ids=None) -> dict:
    """Save the choice made in the dashboard. `new_key` None keeps the key that is already there (the page never
    has it to send back); an empty string removes it. `final_ids` are the models for the last decisions; None keeps
    what was saved, an empty list means "use the same models as for everything else". `video_ids` are the models
    that look at pictures of the video; None keeps what was saved, an empty list means pictures are not used."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    ms = _model_list(model_ids)
    if not ms:
        raise ValueError("Enter the model to use.")
    fm = current()["final_models"] if final_ids is None else _model_list(final_ids)
    vm = current()["video_models"] if video_ids is None else _model_list(video_ids)
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    old = current()
    k = old["key"] if new_key is None and old["provider"] == p else str(new_key or "").strip()[:400]
    if not k and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    with _lock:
        f = _file()
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({"provider": p, "models": ms, "final_models": fm, "video_models": vm, "base_url": address, "key": k}), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)        # the key is only for whoever runs the server
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


def list_models(provider_id=None, address=None, with_key=None) -> list:
    """The model ids a service offers, so the settings page can show a list to pick from."""
    c = current()
    p = ALIASES.get(str(provider_id or c["provider"]).lower(), str(provider_id or c["provider"]).lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    k = str(with_key or "").strip() or (c["key"] if c["provider"] == p else "")
    base = str(address or "").strip().rstrip("/") or (c["base_url"] if c["provider"] == p else "") or PROVIDERS[p][1]
    if not base:
        raise ValueError("Enter the service's address first.")
    how = PROVIDERS[p][2]
    if how == "gemini":
        r = httpx.get(base, params={"pageSize": 200}, headers={"x-goog-api-key": k}, timeout=20)
    elif how == "anthropic":
        r = httpx.get(f"{base}/models", params={"limit": 200}, headers={"x-api-key": k, "anthropic-version": "2023-06-01"}, timeout=20)
    else:
        r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {k}"} if k else {}, timeout=20)
    if r.status_code != 200:
        raise AIError(f"HTTP {r.status_code} {r.text[:200].replace(k or chr(0), '<key>')}")
    data = r.json()
    if how == "gemini":
        found = [m.get("name", "").split("/")[-1] for m in data.get("models", [])
                 if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    else:
        found = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    return sorted({m for m in found if m})[:400]


def test() -> dict:
    """One tiny request with the settings in use, to prove the key, the address and the model all work."""
    ms = models()
    if not ready():
        raise ValueError("Save a service, a model and a key first.")
    ask = 'Answer with a JSON array that holds the single string "ok".'
    schema = {"type": "ARRAY", "items": {"type": "STRING"}}
    out, sent, wrote = call(ms[0], ask, schema)
    res = {"ok": True, "model": ms[0], "tokens": sent + wrote, "answer": str(out)[:80]}
    fin = models("final")
    if fin and fin[0] != ms[0]:                 # a separate final model must work too
        _o, s2, w2 = call(fin[0], ask, schema)
        res.update(final_model=fin[0], tokens=res["tokens"] + s2 + w2)
    vid = models("video")
    if vid and vid[0] not in (ms[0], res.get("final_model")):    # so does the model that looks at pictures
        _o, s3, w3 = call(vid[0], ask, schema)
        res.update(video_model=vid[0], tokens=res["tokens"] + s3 + w3)
    return res


class AIError(RuntimeError):
    pass


# ---------------------------------------------------------------- what a call cost
_seen = threading.local()      # the cost the service reported for the last call made on this thread
_prices = {"at": 0.0, "by_model": {}}


def last_cost():
    """The cost (US dollars) the service itself reported for the last call on this thread, or None."""
    return getattr(_seen, "cost", None)


def estimate_cost(model: str, tokens_in: int, tokens_out: int):
    """A guess from the service's published prices, only for OpenRouter (the only one that lists them). None if unknown."""
    if provider() != "openrouter":
        return None
    try:
        if time.time() - _prices["at"] > 6 * 3600:
            r = httpx.get(f"{base_url()}/models", timeout=8)
            if r.status_code != 200:
                return None
            _prices.update(at=time.time(), by_model={
                m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"]))
                for m in r.json().get("data", []) if isinstance(m, dict) and isinstance(m.get("pricing"), dict)
                and m["pricing"].get("prompt") is not None and m["pricing"].get("completion") is not None})
        p = _prices["by_model"].get(model)
        return None if p is None else tokens_in * p[0] + tokens_out * p[1]
    except Exception:  # noqa: BLE001
        return None


def _fail(r, model):
    text = r.text[:300].replace(key() or "\0", "<key>")
    raise AIError(f"HTTP {r.status_code} {text}")


# ---------------------------------------------------------------- the schema, said in words for chat models
def _json_schema(s):
    """Gemini's schema spelling (ARRAY, OBJECT...) as ordinary JSON Schema."""
    if isinstance(s, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)) for k, v in s.items()}
    if isinstance(s, list):
        return [_json_schema(x) for x in s]
    return s


def _wrapped(prompt, schema):
    return (prompt + "\n\nAnswer with JSON only, no other text and no code fences: one object with a single key "
            "\"items\", whose value follows this JSON Schema:\n" + json.dumps(_json_schema(schema)))


def _parse(text):
    """The JSON in an answer, even if the model wrapped it in a code fence or a sentence."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        out = json.loads(text)
    except ValueError:
        m = re.search(r"[\[{]", text)
        if not m:
            raise AIError("The AI did not answer with JSON.")
        try:
            out, _end = json.JSONDecoder().raw_decode(text[m.start():])    # the first JSON value, even with text after it
        except ValueError:
            raise AIError("The AI did not answer with JSON.")
    if isinstance(out, dict) and "items" in out:
        out = out["items"]
    return out


# ---------------------------------------------------------------- one call
def call(model: str, prompt: str, schema: dict, images=None):
    """Ask one model. Returns (the parsed JSON answer, tokens sent, tokens written back).
    `images` is a list of (label, jpeg bytes) shown to the model after the prompt; only services that speak the
    OpenAI chat format (OpenRouter and others) take them."""
    how = PROVIDERS[provider()][2]
    _seen.cost = None
    if images:
        if how != "openai":
            raise AIError("Looking at pictures of the video works with OpenRouter or another OpenAI-style service.")
        return _openai(model, prompt, schema, images)
    return {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[how](model, prompt, schema)


def _gemini(model, prompt, schema):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json", "responseSchema": schema}}
    r = httpx.post(f"{base_url()}/{model}:generateContent", json=body, headers={"x-goog-api-key": key()}, timeout=120)
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    u = data.get("usageMetadata", {})
    sent = int(u.get("promptTokenCount") or 0)
    return json.loads(text), sent, max(0, int(u.get("totalTokenCount") or 0) - sent)   # the answer, and any thinking billed with it


def _openai(model, prompt, schema, images=None):
    headers = {"Authorization": f"Bearer {key()}"} if key() else {}
    content = _wrapped(prompt, schema)
    if images:                  # each picture comes after a short text label (its time), as one message
        content = [{"type": "text", "text": content}]
        for label, jpeg in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    # A limit on the answer. Without one, OpenRouter holds back the money for the model's longest possible answer
    # (about $0.65 for a big model), and a call is refused when the balance is below that, however short the answer is.
    body = {"model": model, "max_tokens": 16000 if images else 8000,
            "messages": [{"role": "user", "content": content}]}
    if provider() == "openrouter":
        body["usage"] = {"include": True}          # OpenRouter then says what the call really cost
    # Most services accept these two extras; a few models refuse one of them, so they are dropped on a refusal.
    extras = [{"temperature": 0.4, "response_format": {"type": "json_object"}}, {"response_format": {"type": "json_object"}}, {}]
    r = None
    for extra in extras:
        r = httpx.post(f"{base_url()}/chat/completions", json={**body, **extra}, headers=headers, timeout=360 if images else 180)
        if r.status_code not in (400, 422):
            break
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    u = data.get("usage") or {}
    if isinstance(u.get("cost"), (int, float)):
        _seen.cost = float(u["cost"])
    return _parse(data["choices"][0]["message"].get("content")), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def _anthropic(model, prompt, schema):
    body = {"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
    r = httpx.post(f"{base_url()}/messages", json=body, timeout=180,
                   headers={"x-api-key": key(), "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        _fail(r, model)
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage") or {}
    return _parse(text), int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
