from fastapi import APIRouter, HTTPException
from typing import Dict, List
from uuid import uuid4
from src.schemas import ModelCreate, ModelInfo

router = APIRouter()
_DB: Dict[str, ModelInfo] = {}

@router.post("", response_model=ModelInfo, summary="Create Model")
def create_model(payload: ModelCreate):
    mid = str(uuid4())
    name = payload.urls[0].rstrip("/").split("/")[-1]
    model = ModelInfo(id=mid, name=name)
    _DB[mid] = model
    return model

@router.get("", response_model=List[ModelInfo], summary="List Models")
def list_models():
    return list(_DB.values())

@router.get("/{model_id}", response_model=ModelInfo, summary="Read Model by ID")
def read_model(model_id: str):
    if model_id not in _DB:
        raise HTTPException(status_code=404, detail="Model not found")
    return _DB[model_id]

@router.put("/{model_id}", response_model=ModelInfo, summary="Update Model")
def update_model(model_id: str, payload: ModelCreate):
    if model_id not in _DB:
        raise HTTPException(status_code=404, detail="Model not found")
    model = _DB[model_id]
    updated = ModelInfo(
        id=model.id,
        name=payload.urls[0].rstrip("/").split("/")[-1],
        version=model.version,
        size_bytes=model.size_bytes,
        net_score=model.net_score
    )
    _DB[model_id] = updated
    return updated

@router.delete("/{model_id}", status_code=204, summary="Delete Model")
def delete_model(model_id: str):
    if model_id not in _DB:
        raise HTTPException(status_code=404, detail="Model not found")
    del _DB[model_id]
    return

@router.post("/rate")
def rate_model(payload: ModelCreate):
    from src.ModelCatalogue import ModelCatalogue
    catalogue = ModelCatalogue()
    # you’ll add a helper like evaluate_from_urls(urls)
    scores = catalogue.evaluate_from_urls(payload.urls)
    return scores  # dict with net_score and per-metric subscores