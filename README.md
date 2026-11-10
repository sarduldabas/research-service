# research-service

**Business capability:** research and evidence gathering. An LLM agent plans research questions for the brief, runs a
web search per question using the **Microsoft Foundry `web_search` tool** (Responses API) in parallel, and synthesises
structured notes: findings, an executive overview and **data points** (figure, unit, year, source) that the builder may
use for stat cards and charts. Returns cited sources plus optional image candidates (Wikipedia, no key). Stateless.

## API (Swagger at `/docs`)
| Operation | Endpoint |
|---|---|
| Research | `POST /research` with `{brief}` -> `{queries[], overview, notes[{findings, data_points[]}], sources[], images[]}` |
| Health | `GET /health` |

Every `source_urls` entry in the notes is checked against URLs the search tool actually returned.
Image lookup failures never fail the job.

## Configuration
| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` = canned search results, `azure` = Foundry web search |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` | - | as in intake-service |
| `MAX_QUERIES` | `6` (1-10) | research questions per project; more = deeper but slower and costlier |
| `SEARCH_WORKERS` | `4` | searches run in parallel |
| `ENABLE_IMAGES` | `false` (`true` in the ConfigMap) | look up Wikipedia image candidates |
| `LLM_TIMEOUT` | `180` | seconds per model call |
| `RESEARCH_MODEL`, `SEARCH_MODEL` | `AZURE_OPENAI_DEPLOYMENT` | different deployment for planning/synthesis and for the search calls |
| `RESEARCH_REASONING_EFFORT`, `SEARCH_REASONING_EFFORT` | `LLM_REASONING_EFFORT` | reasoning level per step |

Note: Foundry web search sends queries to Grounding with Bing, outside your compliance boundary. Use non-confidential topics.

## Develop and test
```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
uvicorn app.main:app --port 8002 --reload
```

## Container and Kubernetes
`docker build -t deckpilot/research-service:1.0 .` · manifest in `k8s/research-service.yaml` (2 replicas, ClusterIP).
CI: `.github/workflows/ci.yml`.
