"""Research agent.

Loop: plan sub-questions -> web-search each (in parallel) -> synthesise structured notes
-> attach optional image candidates. Every fact in the notes must trace back to a URL
that the search tool actually returned.
"""
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from . import llm
from .images import find_images
from .schemas import (
    Brief,
    DataPoint,
    Image,
    Note,
    PlanOutput,
    ResearchResponse,
    Source,
    SynthesisOutput,
)

log = logging.getLogger("research")

PLANNER_SYSTEM = """You are the Research Planner of DeckPilot. Given a presentation brief, produce the web research
questions that, answered together, give a presenter everything needed for a professional, evidence-based deck.

Rules:
- Produce exactly {n} distinct questions, each covering a different angle. Where it fits the topic, cover:
  background and definitions; the current state WITH hard numbers and trends over time (so a chart can be drawn);
  the main players, options or approaches and how they compare; real examples or case studies; benefits and
  opportunities; challenges, risks and counter-arguments; recent developments; outlook, and practical recommendations.
- Order them so the answers could tell a story: context, evidence, analysis, implications.
- Each question must be self-contained (it is searched on its own), specific, and answerable from public web sources.
  Name the geography, industry or time period from the brief; use words like "latest" or the current year where it matters.
- Honour the objective, audience, scope and key points of the brief. If the user insisted on specific points, make sure
  at least one question targets each.
- Respond with ONE JSON object and nothing else, matching this JSON schema:
{schema}
"""

SEARCH_PROMPT = """Research question: {question}

Context: this feeds a professional presentation about "{topic}" for {audience}. Objective: {objective}
Scope: {scope}

Search the web and write a fact-dense answer of roughly 250-400 words:
- Lead with the direct answer, then supporting detail.
- Include concrete numbers with units, dates and named organisations. When a metric has a history, give several
  years or periods so a trend can be charted.
- Where options or players are compared, give the comparison on the same measures.
- Mention notable examples or case studies and any credible disagreement or caveat.
- Prefer recent, authoritative sources (official statistics, regulators, major research firms, reputable press) and
  cite them. If something could not be verified, say so rather than guessing."""

SYNTH_SYSTEM = """You are the Research Synthesiser of DeckPilot. You receive raw web-search answers for several
research questions, each with the source URLs that back it. Turn them into structured notes for a slide builder.

Rules:
- Use ONLY information contained in the provided answers. Never add facts from your own knowledge.
- "overview": 3-4 sentences giving the big picture of the whole topic for the stated audience.
- Produce one note per question (merge near-duplicates). Each note has a short "heading" (max 8 words, states the
  theme) and 4-6 "findings": complete, specific statements (max 30 words each) that carry numbers, dates or names and,
  where possible, say why it matters.
- "data_points": every number the answers state explicitly and that could appear on a chart or stat card.
  For each: "label" (what is measured), "value" (exactly as written, e.g. "$4.2 billion"), "numeric" (the same number
  as a plain float, in ONE consistent unit across data points of the same label, or null if not numeric),
  "unit", "period" (year/period) and "source_url". Include time series as several data points with the same label and
  different periods. Never estimate, extrapolate or compute new numbers. Skip the field entirely if there are none.
- "source_urls" must be copied exactly from the URLs provided for that question; never invent URLs.
- Reflect the brief's objective and key points; keep the tone {tone}.
- Respond with ONE JSON object and nothing else, matching this JSON schema:
{schema}
"""


# ----------------------------------------------------------------------------- mock
MOCK_HEADINGS = [
    "Background and relevance",
    "Current state in numbers",
    "Challenges and risks",
    "Outlook and next steps",
    "Key players and examples",
    "Recent developments",
    "Benefits and opportunities",
    "Practical recommendations",
    "Case studies",
    "Comparison of approaches",
]


def _mock_plan(brief: Brief, n: int) -> list[str]:
    t = brief.topic
    base = [
        f"What is {t} and why does it matter today?",
        f"What is the current state of {t}, with key figures?",
        f"What are the main challenges and risks around {t}?",
        f"What is the outlook and what are the next steps for {t}?",
        f"Who are the key players or examples in {t}?",
        f"What recent developments have shaped {t}?",
    ]
    return (base * 2)[:n]


def _mock_data_points(notes: list[Note]) -> None:
    """Placeholder numbers so the offline demo can show stat cards and a chart (clearly marked as mock)."""
    if len(notes) > 1:
        url = notes[1].source_urls[0] if notes[1].source_urls else ""
        notes[1].data_points = [
            DataPoint(label="Mock metric", value=f"{v} units", numeric=float(v), unit="units", period=str(y), source_url=url)
            for y, v in zip((2021, 2022, 2023, 2024), (12, 18, 27, 41))
        ]
    if notes:
        url = notes[0].source_urls[0] if notes[0].source_urls else ""
        notes[0].data_points = [
            DataPoint(label="Mock share", value="42%", numeric=42.0, unit="%", period="2024", source_url=url),
            DataPoint(label="Mock growth", value="3.1x", numeric=3.1, unit="x", period="2024", source_url=url),
            DataPoint(label="Mock market size", value="$7 billion", numeric=7.0, unit="USD bn", period="2024", source_url=url),
        ]


def _mock_search(question: str) -> tuple[str, list[dict]]:
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:40]
    text = (
        f"(mock answer) {question} Key point one: adoption is growing steadily. "
        "Key point two: costs and regulation vary by region. "
        "Key point three: experts expect continued change over the next few years."
    )
    return text, [
        {"title": f"Mock source A - {slug}", "url": f"https://example.com/mock/{slug}/a"},
        {"title": f"Mock source B - {slug}", "url": f"https://example.org/mock/{slug}/b"},
    ]


# ----------------------------------------------------------------------------- steps
def plan_questions(brief: Brief, n: int) -> list[str]:
    if llm.provider() == "mock":
        return _mock_plan(brief, n)
    system = PLANNER_SYSTEM.replace("{n}", str(n))
    out = llm.ask_json(system, f"BRIEF:\n{brief.model_dump_json(indent=2)}", PlanOutput, role="RESEARCH")
    questions = [q.strip() for q in out.questions if q.strip()]
    return questions[:n] or [f"Give an overview of {brief.topic}."]


def search(question: str, brief: Brief) -> tuple[str, list[dict]]:
    if llm.provider() == "mock":
        return _mock_search(question)
    prompt = SEARCH_PROMPT.format(
        question=question,
        topic=brief.topic,
        audience=brief.audience,
        objective=brief.objective or "inform the audience",
        scope=brief.scope or "general",
    )
    return llm.web_search(prompt, role="SEARCH")


def _fallback_notes(findings: list[dict]) -> list[Note]:
    notes = []
    for f in findings:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", f["text"]) if len(s.strip()) > 20]
        notes.append(
            Note(
                heading=f["question"].rstrip("?")[:70],
                findings=sentences[:4] or [f["text"][:200]],
                source_urls=[s["url"] for s in f["sources"]],
            )
        )
    return notes


def synthesise(brief: Brief, findings: list[dict]) -> SynthesisOutput:
    if llm.provider() == "mock":
        notes = _fallback_notes(findings)
        for note, heading in zip(notes, MOCK_HEADINGS):
            note.heading = heading
        _mock_data_points(notes)
        return SynthesisOutput(
            overview=f"(mock overview) {brief.topic}: a short, offline stand-in for the research summary.", notes=notes
        )
    payload = [
        {"question": f["question"], "answer": f["text"][:6000], "sources": f["sources"]}
        for f in findings
    ]
    system = SYNTH_SYSTEM.replace("{tone}", brief.tone)
    user = f"BRIEF:\n{brief.model_dump_json(indent=2)}\n\nRESEARCH RESULTS:\n{json.dumps(payload, indent=2)}"
    try:
        out = llm.ask_json(system, user, SynthesisOutput, role="RESEARCH")
        return out
    except llm.LLMError as exc:
        log.warning("synthesis failed, using fallback notes: %s", exc)
        return SynthesisOutput(notes=_fallback_notes(findings))


# ----------------------------------------------------------------------------- entry
def run(brief: Brief) -> ResearchResponse:
    max_queries = max(1, min(10, int(os.getenv("MAX_QUERIES", "6") or "6")))
    questions = plan_questions(brief, max_queries)

    def one(q: str):
        try:
            text, sources = search(q, brief)
            return {"question": q, "text": text, "sources": sources}
        except llm.LLMError as exc:
            log.warning("search failed for %r: %s", q, exc)
            return None

    workers = max(1, min(int(os.getenv("SEARCH_WORKERS", "4") or "4"), len(questions)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        findings = [f for f in pool.map(one, questions) if f]
    if not findings:
        raise llm.LLMError("every web search failed; check the LLM/search configuration")

    synthesis = synthesise(brief, findings)
    notes = synthesis.notes or _fallback_notes(findings)

    # keep only URLs that the search tool really returned
    known = {s["url"]: s["title"] for f in findings for s in f["sources"]}
    for note in notes:
        note.source_urls = [u for u in note.source_urls if u in known]
        for dp in note.data_points:
            if dp.source_url not in known:
                dp.source_url = note.source_urls[0] if note.source_urls else ""
    used = {u for note in notes for u in note.source_urls} or set(known)
    sources = [Source(title=known[u], url=u) for u in known if u in used][:24]

    images: list[Image] = []
    if os.getenv("ENABLE_IMAGES", "false").lower() == "true":
        images = find_images(brief.topic)

    return ResearchResponse(queries=questions, overview=synthesis.overview, notes=notes, sources=sources, images=images)
