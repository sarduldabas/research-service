"""Contracts for the research-service."""
from pydantic import BaseModel, Field


class Brief(BaseModel):
    topic: str
    objective: str = ""
    audience: str = "General business audience"
    scope: str = ""
    key_points: list[str] = Field(default_factory=list)
    tone: str = "professional"
    slide_count: int = Field(9, ge=3, le=15)


class Source(BaseModel):
    title: str = ""
    url: str


class DataPoint(BaseModel):
    """A number that is explicitly stated in the research, kept apart so charts and stat cards can be trusted."""

    label: str                       # what is measured, e.g. "India installed solar capacity"
    value: str                       # exactly as written in the source, e.g. "$4.2 billion" or "63%"
    numeric: float | None = None     # the same value as a plain number in one consistent unit (for charts)
    unit: str = ""                   # e.g. "GW", "%", "USD bn"
    period: str = ""                 # year or period the value refers to, e.g. "2024"
    source_url: str = ""


class Note(BaseModel):
    heading: str
    findings: list[str]
    data_points: list[DataPoint] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)


class Image(BaseModel):
    url: str
    caption: str = ""
    source: str = ""


class ResearchRequest(BaseModel):
    brief: Brief


class ResearchResponse(BaseModel):
    queries: list[str]
    overview: str = ""
    notes: list[Note]
    sources: list[Source]
    images: list[Image] = Field(default_factory=list)


# ---- shapes the LLM must return -------------------------------------------------
class PlanOutput(BaseModel):
    questions: list[str]


class SynthesisOutput(BaseModel):
    overview: str = ""
    notes: list[Note]
