import logging

from fastapi import FastAPI, HTTPException

from . import agent
from .llm import LLMError, provider
from .schemas import ResearchRequest, ResearchResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="DeckPilot research-service",
    version="1.0.0",
    description="Business capability: research and evidence gathering. Plans research questions, "
    "searches the web (Foundry web_search tool), and returns structured, cited notes.",
)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "service": "research-service", "llm_provider": provider()}


@app.post("/research", response_model=ResearchResponse, tags=["research"])
def research(req: ResearchRequest):
    """Run the research agent for a normalised brief."""
    try:
        return agent.run(req.brief)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
