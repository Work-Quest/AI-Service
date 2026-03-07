"""
AI-service skeleton

Goal:
- Retrieve the ACTIVE KMeans model blob from Neon Postgres (tables: kmeans_run + kmeans_model)
- Retrieve the member's feature row from Neon Postgres (table: user_feature_daily) using ONLY project_member_id
- Run inference and return the assigned role
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from typing import Any, Dict
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from db import get_connection
from repository.kmeans_repository import fetch_active_model_blob
from repository.user_feature_repository import fetch_member_feature_row

app = Flask(__name__)
CORS(app)  # TODO: configure allowed origins for production (or disable CORS entirely)


# -------------------------
# Model loading (from DB)
# -------------------------

_MODEL_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "meta": None, "model": None}
_MODEL_CACHE_TTL_SECONDS = int(os.getenv("MODEL_CACHE_TTL_SECONDS") or "60")


def load_active_model(conn):
    """
    Load + cache the active model.
    """
    now = time.time()
    if _MODEL_CACHE.get("kmeans") is not None and _MODEL_CACHE.get("scaler") is not None and (now - float(_MODEL_CACHE["loaded_at"])) < _MODEL_CACHE_TTL_SECONDS:
        return _MODEL_CACHE["meta"], _MODEL_CACHE["kmeans"], _MODEL_CACHE["scaler"]

    meta, kmeans_blob, scaler_blob = fetch_active_model_blob(conn)
    kmeans = joblib.load(io.BytesIO(kmeans_blob))
    if scaler_blob is None:
        raise RuntimeError("Active model is missing scaler_blob in DB. Cannot run X_scaled = scaler.transform(X).")
    scaler = joblib.load(io.BytesIO(scaler_blob))

    _MODEL_CACHE["loaded_at"] = now
    _MODEL_CACHE["meta"] = meta
    _MODEL_CACHE["kmeans"] = kmeans
    _MODEL_CACHE["scaler"] = scaler
    return meta, kmeans, scaler

@app.route("/")
def health_check():
    return "Team Role Clustering Model API is running"


def explain_assignment_stub(*, cluster_id: int, assigned_role: str) -> str:
    """
    Your previous Flask app used `cluster_summary.json` (local artifact).
    In the DB-based approach you likely want ONE of:
    - Persist cluster summary + reasoning into Neon during training
    - Compute cluster summary on demand from `user_feature_daily` (expensive)
    - Drop explanation for now
    """
    return f"explanation_not_implemented(cluster={cluster_id}, role={assigned_role})"

def _parse_project_member_id(payload: dict) -> str:
    """
    Enforce: caller must POST only `project_member_id` (no user-provided features).
    """
    if not isinstance(payload, dict):
        raise ValueError("Invalid JSON payload.")

    raw = payload.get("project_member_id")
    if not raw:
        raise ValueError("Missing required field: project_member_id")

    # Validate UUID format early (DB column is UUID in ETL schema).
    try:
        return str(uuid.UUID(str(raw)))
    except Exception as e:
        raise ValueError("project_member_id must be a UUID string.") from e
    
def _parse_feedback_payload(payload: dict) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid JSON payload.")

    raw_id = payload.get("project_member_id")
    if not raw_id:
        raise ValueError("Missing required field: project_member_id")

    try:
        project_member_id = str(uuid.UUID(str(raw_id)))
    except Exception as e:
        raise ValueError("project_member_id must be a UUID string.") from e

    user_name = payload.get("user_name")
    if not user_name or not isinstance(user_name, str):
        raise ValueError("Missing required field: user_name")

    return project_member_id, user_name

def build_model_input_from_member_row(member_row: Dict[str, Any]) -> pd.DataFrame:
    # work_load_per_day stored as JSON string like "[1,2,3]"
    work_load_list = []
    try:
        work_load_list = json.loads(member_row.get("work_load_per_day") or "[]")
    except Exception:
        work_load_list = []

    avg_workload = (sum(work_load_list) / len(work_load_list)) if work_load_list else 0.0

    # work_speed stored as JSON string like "[10, 12, 8]"
    speed_list = []
    try:
        speed_list = json.loads(member_row.get("work_speed") or "[]")
    except Exception:
        speed_list = []
    avg_work_speed = (sum(speed_list) / len(speed_list)) if speed_list else 0.0

    df = pd.DataFrame(
        [
            {
                "avg_workload": float(avg_workload),
                "team_work": float(member_row.get("team_work") or 0.0),
                "avg_work_speed": float(avg_work_speed),
                "diligence": float(member_row.get("diligence") or 0.0),
                "overall_quality_score": float(member_row.get("work_quality") or 0.0),
            }
        ]
    )
    return df


DEFAULT_FEATURE_NAMES = ["avg_workload", "team_work", "avg_work_speed", "diligence", "overall_quality_score"]
DEFAULT_ROLE_MAPPING = {
    0: "Balancer",
    1: "Perfectionist",
    2: "Task finisher",
    3: "Lone Wolf",
    4: "Leader",
    5: "Helper",
    6: "Genelarist",
}


def _get_feature_names() -> list[str]:
    raw = (os.getenv("KMEANS_FEATURE_NAMES_JSON") or "").strip()
    if not raw:
        return DEFAULT_FEATURE_NAMES
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass
    return DEFAULT_FEATURE_NAMES


def _get_role_mapping() -> Dict[Any, str]:
    raw = (os.getenv("KMEANS_ROLE_MAPPING_JSON") or "").strip()
    if not raw:
        return DEFAULT_ROLE_MAPPING
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed  # keys may be str or int; we handle both at lookup time
    except Exception:
        pass
    return DEFAULT_ROLE_MAPPING


def _predict_role_for_member(conn, *, project_member_id: str) -> Dict[str, Any]:
    """
    Shared internal logic for /role and /feedback.

    Returns:
      {
        "project_member_id": str,
        "model_meta": ActiveModelMeta,
        "member_row": dict,
        "assigned_role": str,
        "role_explanation": str,
      }
    """
    model_meta, kmeans, scaler = load_active_model(conn)
    member_row = fetch_member_feature_row(conn, project_member_id=project_member_id)
    df = build_model_input_from_member_row(member_row)

    feature_names = _get_feature_names()
    role_mapping = _get_role_mapping()

    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    clusters = kmeans.predict(X_scaled)

    roles = []
    for c in clusters:
        # role_mapping might have int keys or str keys
        role = None
        try:
            role = role_mapping.get(int(c))
        except Exception:
            role = None
        if role is None:
            role = role_mapping.get(str(int(c))) if hasattr(role_mapping, "get") else None
        roles.append(role or "Unknown")

    assigned_role = roles[0] if isinstance(roles, (list, tuple)) and roles else roles

    # If you need cluster_id, expose it from the wrapper or compute it here.
    cluster_id = -1
    explanation = explain_assignment_stub(cluster_id=cluster_id, assigned_role=str(assigned_role))

    return {
        "project_member_id": project_member_id,
        "model_meta": model_meta,
        "member_row": member_row,
        "assigned_role": str(assigned_role),
        "role_explanation": explanation,
    }


@app.route("/role", methods=["POST", "OPTIONS"])
def role_predict():
    # Handle preflight CORS requests
    if request.method == "OPTIONS":
        return "", 200

    payload = request.get_json(silent=True) or {}

    try:
        project_member_id = _parse_project_member_id(payload)

        conn = get_connection()
        try:
            out = _predict_role_for_member(conn, project_member_id=project_member_id)
            model_meta = out["model_meta"]
            return jsonify(
                {
                    "project_member_id": out["project_member_id"],
                    "model": {
                        "run_id": model_meta.run_id,
                        "trained_at": model_meta.trained_at,
                        "k": model_meta.k,
                    },
                    "assigned_role": out["assigned_role"],
                    "role_explanation": out["role_explanation"],
                }
            )
        finally:
            conn.close()

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Feedback generation function
def _safe_json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def generate_feedback(
    *,
    user_name: str,
    work_category: str,
    role: str,
    work_load_per_day,
    team_work,
    work_speed,
    overall_quality_score,
) -> str:
    """
    Generate feedback text using OpenAI (optional).

    If `OPENAI_API_KEY` or the `openai` package is missing, returns a clear message instead of crashing.
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return "OPENAI_API_KEY not configured"

    try:
        from openai import OpenAI  # lazy import so /role works without openai installed
    except Exception:
        return "openai package not installed"

    client = OpenAI(api_key=api_key)

    work_load_list = _safe_json_list(work_load_per_day)
    avg_workload = (sum(work_load_list) / len(work_load_list)) if work_load_list else 0.0

    speed_list = _safe_json_list(work_speed)
    avg_speed = (sum(speed_list) / len(speed_list)) if speed_list else 0.0

    metrics = {
        "Workload": float(avg_workload),
        "Teamwork": float(team_work or 0.0),
        "Speed": float(avg_speed),
        "Quality": float(overall_quality_score or 0.0),
    }

    sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
    strengths = [name for name, _ in sorted_metrics[:2]]
    improvements = [name for name, _ in sorted_metrics[-2:]]

    analyzed_data = {
        "Workload": work_load_list,
        "Teamwork": metrics["Teamwork"],
        "Speed": speed_list,
        "Quality": metrics["Quality"],
        "Strengths": strengths,
        "Improvements": improvements,
        "Best Task": work_category,
        "Role": role,
    }

    prompt = f"""With this analyzed data {analyzed_data}, act like you're talking directly to {user_name} after the project end and give unbiased personal feedback.
Use their name and speak casually. Highlight their strengths (especially their best work category: {work_category}).
Mention their highest performance areas without giving exact numeric scores.
Point out areas for improvement based on weaker aspects.
Keep a motivational tone and end with encouragement. Include time management advice too."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for performance reviews."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating feedback: {e}"


# Feedback generation function
@app.route("/feedback", methods=["POST", "OPTIONS"])
def feedback_stub():
    """
    Not implemented (skeleton).
    """
    if request.method == "OPTIONS":
        return "", 200

    payload = request.get_json(silent=True) or {}
    try:
        project_member_id, user_name = _parse_feedback_payload(payload)

        # Do NOT call your own /role endpoint via HTTP (can deadlock + adds latency).
        conn = get_connection()
        try:
            out = _predict_role_for_member(conn, project_member_id=project_member_id)
        finally:
            conn.close()

        assigned_role = out["assigned_role"]
        member_row = out["member_row"]

        # If you have user names elsewhere (Backend DB), enrich here.
        work_load_per_day = member_row.get("work_load_per_day", "[]")
        team_work = member_row.get("team_work", 0.0)
        work_category = member_row.get("strength", "Unknown Category")
        work_speed = member_row.get("work_speed", "[]")
        overall_quality_score = member_row.get("work_quality", 0.0)
        feedback = generate_feedback(
            user_name=user_name,
            role=assigned_role,
            work_category=work_category,
            work_load_per_day=work_load_per_day,
            team_work=team_work,
            work_speed=work_speed,
            overall_quality_score=overall_quality_score
        )

        return jsonify({
            "project_member_id": project_member_id,
            "user_name": user_name,
            "work_load_per_day" : work_load_per_day,
            "team_work" : team_work,
            "work_category" : work_category,
            "work_speed" : work_speed,
            "overall_quality_score" : overall_quality_score,        
            "assigned_role": assigned_role,
            "feedback" : feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # TODO: configure host/port via env vars for container deployments
    app.run(debug=True)