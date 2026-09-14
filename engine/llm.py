"""
LLM interface for FinSight AI.

The financial intelligence engine remains deterministic and grounded in
structured financial analytics. This module adds optional live LLM
providers — Google Gemini, OpenAI, Anthropic Claude, and xAI Grok — used
ONLY to explain evidence already computed by FinSight. The LLM never
replaces the numbers.

Configuration (environment variables, `.env` file, or Streamlit secrets):
    GEMINI_API_KEY     (fallback: GOOGLE_API_KEY) - Google AI Studio key
    OPENAI_API_KEY                               - OpenAI platform key
    ANTHROPIC_API_KEY                            - Anthropic console key
    GROK_API_KEY       (fallback: XAI_API_KEY)    - xAI console key
    GEMINI_MODEL       (default: gemini-2.5-flash)
    OPENAI_MODEL       (default: gpt-4o-mini)
    ANTHROPIC_MODEL    (default: claude-sonnet-4-5)
    GROK_MODEL         (default: grok-3-mini)
    LLM_PROVIDER       (default: auto) - auto | gemini | openai |
                       anthropic | grok | demo

Get keys:
    Gemini:    https://aistudio.google.com/apikey
    OpenAI:    https://platform.openai.com/api-keys
    Anthropic: https://console.anthropic.com/
    Grok:      https://console.x.ai/
"""

import json
import os

GEMINI_MODELS_DEFAULT = "gemini-3.6-flash"
OPENAI_MODEL_DEFAULT = "gpt-4o-mini"
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-5"
GROK_MODELS_DEFAULT = "grok-3-mini"
GROK_BASE_URL = "https://api.x.ai/v1"

GROUNDING_RULES = (
    "You are FinSight AI, a financial-statement review assistant. "
    "You are given pre-computed financial evidence (metrics, year-over-year "
    "changes, peer comparisons, risk/anomaly flags). Rules:\n"
    "1. Only use numbers that appear in the provided evidence. Never invent, "
    "estimate, or round beyond what is shown.\n"
    "2. If the evidence says a value is unavailable / not provided, say so "
    "explicitly instead of guessing.\n"
    "3. Distinguish facts (from evidence) from interpretation (your analysis).\n"
    "4. Keep the answer concise and structured with short sections.\n"
    "5. All monetary values are in USD millions unless the evidence says otherwise."
)

_session_keys = {"gemini": None, "openai": None, "anthropic": None, "grok": None}


def set_api_keys(gemini=None, openai=None, anthropic=None, grok=None):
    """Inject API keys at runtime (e.g. from the Streamlit UI).

    Session keys take precedence over environment variables but are never
    written to disk.
    """
    if gemini is not None:
        _session_keys["gemini"] = gemini.strip() or None
    if openai is not None:
        _session_keys["openai"] = openai.strip() or None
    if anthropic is not None:
        _session_keys["anthropic"] = anthropic.strip() or None
    if grok is not None:
        _session_keys["grok"] = grok.strip() or None


def reset_session_keys():
    """Clear all runtime-injected keys (fresh Test uses only typed input)."""
    for slot in _session_keys:
        _session_keys[slot] = None


def resolve_provider(requested="auto"):
    """Public wrapper: which provider would serve `requested` (or 'demo')."""
    return _resolve_provider(requested)


def _load_dotenv():
    """Best-effort `.env` loading without a hard dependency."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        return
    except Exception:
        pass

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        for candidate in (
            os.path.join(os.path.dirname(here), ".env"),
            os.path.join(os.getcwd(), ".env"),
        ):
            if not os.path.isfile(candidate):
                continue
            with open(candidate, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass


def _secret(name):
    """Read Streamlit secrets when running inside Streamlit."""
    try:
        import streamlit as st  # type: ignore

        value = st.secrets.get(name)
        return str(value).strip() if value else None
    except Exception:
        return None


def _get_key(*names):
    _load_dotenv()

    session_map = {"GEMINI_API_KEY": "gemini", "GOOGLE_API_KEY": "gemini",
                   "OPENAI_API_KEY": "openai",
                   "ANTHROPIC_API_KEY": "anthropic",
                   "GROK_API_KEY": "grok", "XAI_API_KEY": "grok"}
    for name in names:
        slot = session_map.get(name)
        if slot and _session_keys.get(slot):
            return _session_keys[slot]

    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()

    for name in names:
        value = _secret(name)
        if value:
            return value

    return None


def get_gemini_key():
    return _get_key("GEMINI_API_KEY", "GOOGLE_API_KEY")


def get_openai_key():
    return _get_key("OPENAI_API_KEY")


def get_anthropic_key():
    return _get_key("ANTHROPIC_API_KEY")


def get_grok_key():
    return _get_key("GROK_API_KEY", "XAI_API_KEY")


def _sdk_available(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def get_llm_status():
    """Return provider availability for the UI."""
    gemini_key = bool(get_gemini_key())
    openai_key = bool(get_openai_key())
    anthropic_key = bool(get_anthropic_key())
    grok_key = bool(get_grok_key())

    providers = {
        "gemini": {
            "configured": gemini_key,
            "sdk_installed": _sdk_available("google.genai")
            or _sdk_available("google.generativeai"),
            "model": os.environ.get("GEMINI_MODEL", GEMINI_MODELS_DEFAULT),
        },
        "openai": {
            "configured": openai_key,
            "sdk_installed": _sdk_available("openai"),
            "model": os.environ.get("OPENAI_MODEL", OPENAI_MODEL_DEFAULT),
        },
        "anthropic": {
            "configured": anthropic_key,
            "sdk_installed": _sdk_available("anthropic"),
            "model": os.environ.get("ANTHROPIC_MODEL", ANTHROPIC_MODEL_DEFAULT),
        },
        "grok": {
            "configured": grok_key,
            "sdk_installed": _sdk_available("openai"),
            "model": os.environ.get("GROK_MODEL", GROK_MODELS_DEFAULT),
        },
    }

    # Auto preference order: gemini -> openai -> anthropic -> grok.
    active = next(
        (name for name in ("gemini", "openai", "anthropic", "grok")
         if providers[name]["configured"]),
        None,
    )

    if active:
        mode, provider = "LIVE", active
    else:
        mode, provider = "DEMO", "Local FinSight AI"

    return {
        "mode": mode,  # LIVE when any provider key is present, else DEMO
        "provider": provider,
        "api_connected": bool(active),
        "ready_for_provider_integration": True,
        "providers": providers,
    }


def _resolve_provider(requested="auto"):
    requested = (requested or "auto").strip().lower()
    forced = (os.environ.get("LLM_PROVIDER") or "").strip().lower()

    if requested == "auto" and forced in (
        "gemini", "openai", "anthropic", "grok", "demo",
    ):
        requested = forced

    key_getters = {
        "gemini": get_gemini_key,
        "openai": get_openai_key,
        "anthropic": get_anthropic_key,
        "grok": get_grok_key,
    }

    if requested in key_getters:
        return requested if key_getters[requested]() else "demo"
    if requested == "demo":
        return "demo"

    # auto: gemini -> openai -> anthropic -> grok, else demo
    for name, getter in key_getters.items():
        if getter():
            return name
    return "demo"


def _evidence_to_text(evidence, limit=12000):
    """Serialize grounding evidence compactly (dicts, DataFrames, text).

    Financial tables are small (years x metrics), so the full frame is
    included rather than a truncated head — the model must see every
    engine-calculated value it is asked about.
    """
    try:
        import pandas as pd  # type: ignore

        if isinstance(evidence, pd.DataFrame):
            text = evidence.head(100).to_string(index=False)
            return text[:limit]
    except Exception:
        pass

    if isinstance(evidence, dict):
        try:
            return json.dumps(evidence, indent=1, default=str)[:limit]
        except Exception:
            return str(evidence)[:limit]

    return str(evidence or "")[:limit]


def _call_gemini(prompt, system, model, timeout):
    """Call Gemini via the new google-genai SDK, else the legacy SDK."""
    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=get_gemini_key())
        # http_options timeout is milliseconds; without it a stalled
        # network hangs the Streamlit spinner indefinitely.
        config = {"http_options": {"timeout": int(timeout * 1000)}}
        if system:
            config["system_instruction"] = system
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return (getattr(response, "text", "") or "").strip()
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    try:
        import google.generativeai as genai_legacy  # type: ignore

        genai_legacy.configure(api_key=get_gemini_key())
        model_obj = genai_legacy.GenerativeModel(
            model, system_instruction=system or GROUNDING_RULES
        )
        response = model_obj.generate_content(
            prompt, request_options={"timeout": timeout}
        )
        return (getattr(response, "text", "") or "").strip()
    except ImportError as exc:
        raise RuntimeError(
            "Gemini SDK not installed. Run: pip install google-genai"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc


def _call_openai(prompt, system, model, timeout):
    """Call OpenAI Chat Completions."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from exc

    try:
        client = OpenAI(api_key=get_openai_key())
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            timeout=timeout,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc


def _call_anthropic(prompt, system, model, timeout):
    """Call Anthropic Claude Messages API."""
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc

    try:
        client = anthropic.Anthropic(
            api_key=get_anthropic_key(), timeout=timeout
        )
        kwargs = {
            "model": model,
            "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1024")),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        parts = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", "") == "text"
        ]
        return "".join(parts).strip()
    except Exception as exc:
        raise RuntimeError(f"Anthropic API error: {exc}") from exc


def _call_grok(prompt, system, model, timeout):
    """Call xAI Grok through its OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from exc

    try:
        client = OpenAI(api_key=get_grok_key(), base_url=GROK_BASE_URL)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            timeout=timeout,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError(f"Grok API error: {exc}") from exc


def _call_with_fallback(call_fn, prompt, model, model_default, timeout):
    """Call a provider, retrying once with its default model.

    Returns (text, model_used). A misconfigured *_MODEL env override
    (unknown model id) is the most common setup failure — instead of
    failing outright, fall back to the known-good default once.
    """
    import re

    try:
        return call_fn(prompt, GROUNDING_RULES, model, timeout), model
    except RuntimeError as exc:
        message = str(exc)
        looks_like_bad_model = (
            model != model_default
            and re.search(
                r"not.?found|404|invalid.?model|does not exist|unknown model",
                message,
                re.IGNORECASE,
            )
        )
        if not looks_like_bad_model:
            raise
        text = call_fn(prompt, GROUNDING_RULES, model_default, timeout)
        return text, model_default


def test_provider_connection(provider):
    """Make a minimal live call to check a provider end-to-end.

    Returns a dict: {provider, model_requested, model_used, ok, message}.
    Costs a few dozen tokens at most.
    """
    provider = (provider or "").strip().lower()

    key_getters = {
        "gemini": (get_gemini_key, "GEMINI_MODEL", GEMINI_MODELS_DEFAULT, _call_gemini),
        "openai": (get_openai_key, "OPENAI_MODEL", OPENAI_MODEL_DEFAULT, _call_openai),
        "anthropic": (get_anthropic_key, "ANTHROPIC_MODEL", ANTHROPIC_MODEL_DEFAULT, _call_anthropic),
        "grok": (get_grok_key, "GROK_MODEL", GROK_MODELS_DEFAULT, _call_grok),
    }

    if provider not in key_getters:
        return {
            "provider": provider,
            "model_requested": None,
            "model_used": None,
            "ok": False,
            "message": f"Unknown provider '{provider}'.",
        }

    get_key, model_env, model_default, call_fn = key_getters[provider]

    if not get_key():
        return {
            "provider": provider,
            "model_requested": None,
            "model_used": None,
            "ok": False,
            "message": "No API key configured.",
        }

    model_requested = os.environ.get(model_env, model_default)
    timeout = int(os.environ.get("LLM_TIMEOUT", "60"))

    try:
        text, model_used = _call_with_fallback(
            call_fn,
            "Reply with exactly: OK",
            model_requested,
            model_default,
            timeout,
        )
    except RuntimeError as exc:
        return {
            "provider": provider,
            "model_requested": model_requested,
            "model_used": None,
            "ok": False,
            "message": str(exc)[:300],
        }

    return {
        "provider": provider,
        "model_requested": model_requested,
        "model_used": model_used,
        "ok": True,
        "message": f"Connected (model: {model_used}). Reply: {(text or '')[:100]}",
    }


# Curated fallback when a provider offers no list-models API.
KNOWN_ANTHROPIC_MODELS = [
    "claude-sonnet-4-5",
    "claude-opus-4-1",
    "claude-sonnet-4-0",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest",
]

# Model ids that are never useful for chat answers.
_NON_CHAT_MARKERS = (
    "embed", "whisper", "tts", "dall-e", "dall·e", "moderation",
    "babbage", "davinci", "transcribe", "image", "audio",
)


def _is_chat_model(model_id):
    name = str(model_id).lower()
    return not any(marker in name for marker in _NON_CHAT_MARKERS)


def list_provider_models(provider):
    """List usable chat models for a configured provider.

    Returns {provider, ok, models, source, message} where source is
    'live' (fetched from the provider) or 'curated' (Anthropic exposes
    no list-models endpoint, so a known-good list is returned).
    Requires a key; never raises.
    """
    provider = (provider or "").strip().lower()
    timeout = int(os.environ.get("LLM_TIMEOUT", "60"))

    if provider == "gemini":
        if not get_gemini_key():
            return {"provider": provider, "ok": False, "models": [],
                    "source": "none", "message": "No API key configured."}
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=get_gemini_key())
            models = sorted({
                str(m.name).split("/", )[-1]
                for m in client.models.list()
                if m.name and _is_chat_model(str(m.name).split("/")[-1])
            })
            if not models:
                raise RuntimeError("provider returned an empty model list")
            return {"provider": provider, "ok": True, "models": models,
                    "source": "live",
                    "message": f"{len(models)} models available."}
        except Exception as exc:
            return {"provider": provider, "ok": False, "models": [],
                    "source": "none",
                    "message": f"Gemini model list failed: {exc}"[:300]}

    if provider in ("openai", "grok"):
        get_key = get_openai_key if provider == "openai" else get_grok_key
        if not get_key():
            return {"provider": provider, "ok": False, "models": [],
                    "source": "none", "message": "No API key configured."}
        try:
            from openai import OpenAI  # type: ignore

            kwargs = {"api_key": get_key(), "timeout": timeout}
            if provider == "grok":
                kwargs["base_url"] = GROK_BASE_URL
            client = OpenAI(**kwargs)
            models = sorted({
                item.id for item in client.models.list().data
                if item.id and _is_chat_model(item.id)
            })
            if not models:
                raise RuntimeError("provider returned an empty model list")
            return {"provider": provider, "ok": True, "models": models,
                    "source": "live",
                    "message": f"{len(models)} models available."}
        except Exception as exc:
            label = "OpenAI" if provider == "openai" else "Grok"
            return {"provider": provider, "ok": False, "models": [],
                    "source": "none",
                    "message": f"{label} model list failed: {exc}"[:300]}

    if provider == "anthropic":
        if not get_anthropic_key():
            return {"provider": provider, "ok": False, "models": [],
                    "source": "none", "message": "No API key configured."}
        return {"provider": provider, "ok": True,
                "models": list(KNOWN_ANTHROPIC_MODELS), "source": "curated",
                "message": "Anthropic exposes no list-models API; "
                           "showing known Claude models."}

    return {"provider": provider, "ok": False, "models": [],
            "source": "none", "message": f"Unknown provider '{provider}'."}


def generate_ai_response(prompt, context=None, provider="auto", model=None):
    """Generate an AI interpretation of grounded financial evidence.

    Parameters
    ----------
    prompt : str
        Grounded prompt / question, ideally with evidence attached.
    context : dict, optional
        Structured financial context (serialized as grounding evidence).
    provider : str
        'auto' | 'gemini' | 'openai' | 'anthropic' | 'grok' | 'demo'.
    model : str, optional
        Model override (defaults from GEMINI_MODEL / OPENAI_MODEL /
        ANTHROPIC_MODEL / GROK_MODEL).

    Returns
    -------
    dict with keys: status (SUCCESS | DEMO | ERROR), provider, model,
    response, context_available.
    """
    context = context or {}

    if not prompt or not str(prompt).strip():
        return {
            "status": "ERROR",
            "provider": "none",
            "model": None,
            "response": "No AI prompt was provided.",
            "context_available": bool(context),
        }

    active = _resolve_provider(provider)

    if active == "demo":
        hint = (
            "FinSight AI is currently running in demo mode. "
            "The financial intelligence engine has prepared grounded "
            "financial evidence, risk signals, anomalies, and peer "
            "benchmarks for AI interpretation. "
        )
        if (provider or "auto").lower() != "demo":
            hint += (
                "To enable live AI explanations, set GEMINI_API_KEY, "
                "OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROK_API_KEY "
                "(FinSight AI Chat key panel, .env file, or "
                "environment variable)."
            )
        return {
            "status": "DEMO",
            "provider": "Local FinSight AI",
            "model": None,
            "response": hint,
            "context_available": bool(context),
            "prompt_ready": True,
        }

    evidence_text = _evidence_to_text(context)
    full_prompt = str(prompt).strip()
    if evidence_text:
        full_prompt = (
            f"{full_prompt}\n\n--- GROUNDING EVIDENCE (use only these numbers) ---\n"
            f"{evidence_text}"
        )

    timeout = int(os.environ.get("LLM_TIMEOUT", "60"))

    provider_calls = {
        "gemini": (_call_gemini, "GEMINI_MODEL", GEMINI_MODELS_DEFAULT),
        "openai": (_call_openai, "OPENAI_MODEL", OPENAI_MODEL_DEFAULT),
        "anthropic": (_call_anthropic, "ANTHROPIC_MODEL", ANTHROPIC_MODEL_DEFAULT),
        "grok": (_call_grok, "GROK_MODEL", GROK_MODELS_DEFAULT),
    }

    try:
        call_fn, model_env, model_default = provider_calls[active]
        active_model = model or os.environ.get(model_env, model_default)
        text, active_model = _call_with_fallback(
            call_fn, full_prompt, active_model, model_default, timeout
        )
    except RuntimeError as exc:
        return {
            "status": "ERROR",
            "provider": active,
            "model": None,
            "response": str(exc),
            "context_available": bool(context),
        }

    if not text:
        return {
            "status": "ERROR",
            "provider": active,
            "model": active_model,
            "response": "The AI provider returned an empty response. Please retry.",
            "context_available": bool(context),
        }

    return {
        "status": "SUCCESS",
        "provider": active,
        "model": active_model,
        "response": text,
        "context_available": bool(context),
    }


def build_chat_evidence(result, company, data_source="default dataset",
                        snapshot=None, trend_records=None, selected_year=None):
    """Build the grounding-evidence dict for a chat question.

    Shared by the Streamlit UI and tests so both exercise the same logic.

    - Deterministic SUCCESS  -> full engine answer + summary + full table.
    - Anything else          -> company snapshot + recent trend + engine note,
      so health / risk / summary questions are still answered from
      Python-calculated values.
    """
    result = result or {}

    if result.get("status") == "SUCCESS":
        data_frame = result.get("data")
        try:
            table = (
                data_frame.to_dict(orient="records")
                if data_frame is not None and not data_frame.empty
                else []
            )
        except Exception:
            table = []

        return {
            "data_source": data_source,
            "company": company,
            "engine_answer": result.get("answer", ""),
            "engine_summary": result.get("summary"),
            "engine_table": table,
        }

    if snapshot is None:
        snapshot = {}
    if trend_records is None:
        trend_records = []

    return {
        "data_source": data_source,
        "company": company,
        "selected_year": selected_year,
        "engine_note": result.get("answer", ""),
        "year_snapshot": snapshot,
        "recent_trend": trend_records,
    }


def generate_financial_answer(question, evidence, provider="auto", model=None):
    """Answer a dataset question with a live LLM grounded in engine values.

    `evidence` carries the company, the data source (uploaded file vs
    default dataset), and the Python engine's calculated numbers. The
    LLM explains those values; it must not change or invent figures.
    """
    inner = evidence.get("evidence", {}) \
        if isinstance(evidence, dict) else {}
    state = inner.get("current_state", {}) \
        if isinstance(inner, dict) else {}

    company = state.get("company") or evidence.get("company", "") \
        if isinstance(evidence, dict) else ""
    source = (state.get("filename")
              or evidence.get("data_source", "")) \
        if isinstance(evidence, dict) else ""
    if isinstance(source, dict):
        source = ""

    prompt = (
        f"User question about the financial dataset: {question}\n"
        f"{('Company: ' + str(company) + '. ') if company else ''}"
        f"{('Data source: ' + str(source) + '. ') if source else ''}\n"
        "FinSight's Python engine calculated the grounded values below from "
        "that dataset. Answer the user's question using ONLY these values: "
        "quote the relevant figures with their periods, explain what "
        "happened, why it matters, and what to look at next. Use FinSight's "
        "pre-computed changes and ratios as given — never recompute them "
        "differently. If a requested value is absent from the evidence, "
        "say it is not available instead of estimating it. Do not claim "
        "causes the data does not support; use 'may indicate' or 'could "
        "warrant investigation'. Structure longer answers as: Finding, "
        "Evidence (figures with periods), Interpretation."
    )
    return generate_ai_response(prompt, context={"evidence": evidence},
                                provider=provider, model=model)
