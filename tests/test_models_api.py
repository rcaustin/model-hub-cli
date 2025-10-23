from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)

def test_crud_models():
    # Create
    payload = {"urls": ["https://huggingface.co/org/model-name"]}
    r = client.post("/models", json=payload)
    assert r.status_code == 200
    created = r.json()
    mid = created["id"]

    # Read
    r = client.get(f"/models/{mid}")
    assert r.status_code == 200

    # Update
    upd = {"urls": ["https://huggingface.co/org/updated"]}
    r = client.put(f"/models/{mid}", json=upd)
    assert r.status_code == 200
    assert r.json()["name"] == "updated"

    # Delete
    r = client.delete(f"/models/{mid}")
    assert r.status_code == 204
