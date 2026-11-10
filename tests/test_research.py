import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["ENABLE_IMAGES"] = "false"

from fastapi.testclient import TestClient  # noqa: E402

from app import agent  # noqa: E402
from app.llm import extract_json  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)
BRIEF = {"topic": "Solar power in India", "audience": "Students", "slide_count": 6}


def test_health():
    assert client.get("/health").json()["service"] == "research-service"


def test_research_returns_notes_sources_and_queries():
    r = client.post("/research", json={"brief": BRIEF})
    assert r.status_code == 200
    body = r.json()
    assert len(body["queries"]) == 6
    assert body["overview"]
    assert any(n["data_points"] for n in body["notes"])
    assert len(body["notes"]) >= 1
    urls = {s["url"] for s in body["sources"]}
    assert urls, "expected sources"
    for note in body["notes"]:
        assert note["findings"]
        assert set(note["source_urls"]) <= urls


def test_max_queries_env(monkeypatch):
    monkeypatch.setenv("MAX_QUERIES", "2")
    assert len(client.post("/research", json={"brief": BRIEF}).json()["queries"]) == 2


def test_unknown_urls_are_filtered(monkeypatch):
    from app.schemas import DataPoint, Note, SynthesisOutput

    real = agent.synthesise
    monkeypatch.setattr(
        agent,
        "synthesise",
        lambda brief, findings: SynthesisOutput(
            notes=[Note(heading="h", findings=["f"], source_urls=["https://evil.invalid/x"],
                        data_points=[DataPoint(label="l", value="1", source_url="https://evil.invalid/y")])]
        ),
    )
    body = client.post("/research", json={"brief": BRIEF}).json()
    assert body["notes"][0]["source_urls"] == []
    assert body["notes"][0]["data_points"][0]["source_url"] == ""  # invented data-point URLs are cleared
    monkeypatch.setattr(agent, "synthesise", real)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
