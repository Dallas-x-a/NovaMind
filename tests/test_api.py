import pytest
from fastapi.testclient import TestClient
from novamind.api import app

client = TestClient(app)

def test_healthz():
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_agent_register_and_task():
    agent = {
        "name": "test_agent",
        "role": "researcher",
        "model_config": {"provider": "openai", "model": "gpt-4o"}
    }
    resp = client.post("/agents/register", json=agent)
    assert resp.status_code == 200
    agent_id = resp.json()["agent_id"]
    # 提交任务
    task = {"agent": agent_id, "payload": "hello"}
    resp2 = client.post("/tasks/submit", json=task)
    assert resp2.status_code == 200
    assert "task_id" in resp2.json()

def test_monitor_summary():
    resp = client.get("/monitor/summary")
    assert resp.status_code == 200
    assert "health_status" in resp.json() 