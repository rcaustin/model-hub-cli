from fastapi import FastAPI
from src.api.routes_models import router as models_router

app = FastAPI(
    title="Trustworthy Model Registry",
    version="0.1.0",
    description="Phase 2 REST API built over the Phase 1 CLI core."
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(models_router, prefix="/models", tags=["models"])
