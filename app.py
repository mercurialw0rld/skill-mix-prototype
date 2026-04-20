import json
import os
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from skillmix import run_pipeline


app = FastAPI(title="SkillMix Demo")
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def resolve_api_key() -> str:
    """Get Render API key."""
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key

    raise ValueError(
        "Google API key not found."
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "result": None,
            "error": None,
        },
    )


@app.post("/run", response_class=HTMLResponse)
def run_demo(
    request: Request,
    topic: str = Form("beekeeping"),
    k: int = Form(2),
):
    try:
        resolved_api_key = resolve_api_key()
        cleaned_topic = topic.strip() or "beekeeping"
        k_value = max(1, min(k, 6))

        result = run_pipeline(topic=cleaned_topic, k=k_value, api_key=resolved_api_key)

        view_model = {
            "title": result["title"],
            "topic": result["topic"],
            "k": result["k"],
            "skills": ", ".join(s["name"] for s in result["selected_skills"]),
            "generation": result["generation"],
            "grades": json.dumps(result["grades"], indent=2, ensure_ascii=False),
        }

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "result": view_model,
                "error": None,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "result": None,
                "error": str(exc),
            },
            status_code=400,
        )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
