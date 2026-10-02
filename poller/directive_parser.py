import json
import logging
import os
import re
import threading

log = logging.getLogger("print-poller.directives")

LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
LLAMA_CONTEXT = int(os.environ.get("LLAMA_CONTEXT", "512"))
_DIRECTIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "page-ranges": {"type": "string", "pattern": "^[0-9,-]+$"},
        "page-set": {"type": "string", "enum": ["even", "odd"]},
        "print-color-mode": {"type": "string", "enum": ["monochrome", "color"]},
        "sides": {
            "type": "string",
            "enum": ["one-sided", "two-sided-long-edge", "two-sided-short-edge"],
        },
    },
    "additionalProperties": False,
}
_LLAMA = None
_LLAMA_LOCK = threading.Lock()


def _canonicalize_sides(value):
    if not value:
        return None
    v = value.strip().lower()
    replacements = {
        "one sided": "one-sided",
        "one-sided": "one-sided",
        "two sided long edge": "two-sided-long-edge",
        "two-sided-long-edge": "two-sided-long-edge",
        "two sided short edge": "two-sided-short-edge",
        "two-sided-short-edge": "two-sided-short-edge",
        "simplex": "one-sided",
        "duplex": "two-sided-long-edge",
    }
    return replacements.get(v, v)


def _llama_json(payload, page_count=None):
    if not os.path.exists(LLAMA_MODEL_PATH):
        return {}

    page_ctx = ""
    if page_count:
        last = str(page_count)
        penultimate = str(max(1, page_count - 1))
        last_two = f"{max(1, page_count - 1)}-{page_count}"
        page_ctx = (
            f"Total pages in document: {page_count}.\n"
            f"- 'last page' -> '{last}'\n"
            f"- 'second last page' or 'penultimate page' -> '{penultimate}'\n"
            f"- 'last two pages' or 'last 2 pages' -> '{last_two}'\n"
        )

    prompt = (
        "<|system|>\n"
        "You extract CUPS print settings from emails into JSON.\n"
        "Supported keys: \"page-ranges\", \"page-set\", \"print-color-mode\", \"sides\".\n"
        f"{page_ctx}"
        "Examples:\n"
        "- 'first two pages' -> {\"page-ranges\": \"1-2\"}\n"
        "- 'first page' -> {\"page-ranges\": \"1\"}\n"
        "- 'first two pages, page 5, & page 10-12' -> {\"page-ranges\": \"1-2,5,10-12\"}\n"
        "- 'pages 3 to 5' -> {\"page-ranges\": \"3-5\"}\n"
        "- 'even numbered pages' -> {\"page-set\": \"even\"}\n"
        "- 'odd numbered pages' or 'every other page' -> {\"page-set\": \"odd\"}\n"
        "- 'print duplex in bw' -> {\"print-color-mode\": \"monochrome\", \"sides\": \"two-sided-long-edge\"}\n"
        "Omit keys not requested. If no print settings requested, output {}.\n"
        "<|user|>\n"
        f"Subject: {payload.get('subject', '')}\n"
        f"Body: {payload.get('body', '')}\n"
        "<|assistant|>\n"
    )

    try:
        from llama_cpp import Llama, LlamaGrammar

        global _LLAMA
        with _LLAMA_LOCK:
            if _LLAMA is None:
                _LLAMA = Llama(
                    model_path=LLAMA_MODEL_PATH,
                    n_ctx=LLAMA_CONTEXT,
                    verbose=False,
                )
            result = _LLAMA.create_completion(
                prompt=prompt,
                max_tokens=128,
                temperature=0.0,
                grammar=LlamaGrammar.from_json_schema(json.dumps(_DIRECTIVE_SCHEMA), verbose=False),
            )
    except Exception as e:
        log.warning("local LLM unavailable for print directive parsing: %s", e)
        return {}

    choices = result.get("choices", [])
    text = choices[0].get("text", "").strip() if choices else ""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    normalized = {}
    email_text = f"{payload.get('subject', '')} {payload.get('body', '')}".lower()
    if isinstance(data, dict):
        color_mode = str(data.get("print-color-mode") or data.get("print_color_mode") or "").strip().lower()
        monochrome_terms = r"\b(?:bw|b\s*&\s*w|mono(?:chrome)?|grayscale|black\s*(?:and|&)\s*white)\b"
        if color_mode == "monochrome" and re.search(monochrome_terms, email_text):
            normalized["print-color-mode"] = color_mode
        elif color_mode == "color" and re.search(r"\bcolou?r\b", email_text) and not re.search(monochrome_terms, email_text):
            normalized["print-color-mode"] = color_mode

        if "sides" in data and data["sides"] and re.search(r"\b(?:duplex|two\s*[- ]?sided|both\s+sides|one\s*[- ]?sided|simplex)\b", email_text):
            normalized["sides"] = _canonicalize_sides(str(data["sides"]))

        page_set = str(data.get("page-set") or "").strip().lower()
        if re.search(r"\b(?:even\s+numbered|even\s+pages?|page-set:\s*even)\b", email_text):
            normalized["page-set"] = "even"
        elif re.search(r"\b(?:odd\s+numbered|odd\s+pages?|every\s+other\s+page|page-set:\s*odd)\b", email_text):
            normalized["page-set"] = "odd"
        elif page_set in {"even", "odd"} and re.search(r"\b(?:even|odd|every\s+other)\b", email_text):
            normalized["page-set"] = page_set

        if "page-set" not in normalized:
            if page_count:
                if re.search(r"\b(?:second\s+(?:to\s+)?last|penultimate)\s+pages?\b", email_text):
                    normalized["page-ranges"] = str(max(1, page_count - 1))
                elif re.search(r"\blast\s+(?:2|two)\s+pages?\b", email_text):
                    normalized["page-ranges"] = f"{max(1, page_count - 1)}-{page_count}"
                elif re.search(r"\blast\s+page\b", email_text):
                    normalized["page-ranges"] = str(page_count)

            if "page-ranges" not in normalized:
                page_val = data.get("page-ranges") or data.get("page_ranges")
                if page_val and re.search(r"\b(?:pages?|first|last|[0-9])\b", email_text):
                    cleaned = str(page_val).strip(" ,-")
                    if cleaned:
                        normalized["page-ranges"] = cleaned

    return normalized


def parse_print_directives(subject, body, page_count=None):
    llm_map = _llama_json({"subject": subject, "body": body}, page_count=page_count)
    return llm_map or {}


def parse_email_message(msg, page_count=None):
    subject = msg.get("Subject", "") or ""
    body_parts = []
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        if part.get_filename():
            continue
        ct = part.get_content_type()
        if ct not in {"text/plain", "text/html"}:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        try:
            body = payload.decode("utf-8", errors="replace")
        except Exception:
            body = str(payload)
        if body:
            body_parts.append(body)
    return parse_print_directives(subject, "\n".join(body_parts), page_count=page_count)
