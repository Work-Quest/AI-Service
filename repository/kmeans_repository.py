from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from psycopg2.extras import RealDictCursor


@dataclass(frozen=True)
class ActiveModelMeta:
    run_id: str
    trained_at: str
    k: Optional[int]


def fetch_active_model_blob(conn) -> Tuple[ActiveModelMeta, bytes, Optional[bytes]]:
    """
    Fetch the active KMeans model artifact from Neon.

    Expected schema (created by ETL):
      - kmeans_run(is_active, trained_at, k, ...)
      - kmeans_model(run_id, model_blob BYTEA, scaler_blob BYTEA?)
    """
    sql = """
        SELECT
          r.run_id::text AS run_id,
          r.trained_at::text AS trained_at,
          r.k AS k,
          m.model_blob AS model_blob,
          m.scaler_blob AS scaler_blob
        FROM kmeans_run r
        JOIN kmeans_model m ON m.run_id = r.run_id
        WHERE r.is_active = TRUE
        ORDER BY r.trained_at DESC
        LIMIT 1
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql)
        row: Optional[Dict[str, Any]] = cur.fetchone()

    if not row:
        raise RuntimeError("No active KMeans model found. (kmeans_run.is_active = TRUE)")

    meta = ActiveModelMeta(
        run_id=row["run_id"],
        trained_at=row["trained_at"],
        k=row.get("k"),
    )
    model_blob = bytes(row["model_blob"])
    scaler_blob = bytes(row["scaler_blob"]) if row.get("scaler_blob") is not None else None
    return meta, model_blob, scaler_blob


