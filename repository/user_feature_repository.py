from __future__ import annotations

from typing import Any, Dict, Optional

from psycopg2.extras import RealDictCursor


def fetch_member_feature_row(conn, *, project_member_id: str) -> Dict[str, Any]:
    """
    Fetch a member's features from Neon.

    Source table: user_feature_daily (ETL output).
    """
    sql = """
        SELECT
          project_member_id::text,
          project_id::text,
          work_load_per_day,
          team_work,
          work_speed,
          diligence,
          work_quality,
          strength
        FROM user_feature_daily
        WHERE project_member_id = %s::uuid
        LIMIT 1
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (project_member_id,))
        row: Optional[Dict[str, Any]] = cur.fetchone()

    if not row:
        raise RuntimeError(f"No user_feature_daily row found for project_member_id={project_member_id}")

    return dict(row)


