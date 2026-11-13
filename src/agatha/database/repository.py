"""Short PostgreSQL transactions, including atomic scene ingestion and queue submission.

The direct/session-pooler database URL stays in the server environment. Nothing
returns it to the browser. Storage credentials are separate from the database URL.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

from .ingest import ImportBundle
from .search import SearchFilters, where_clause

_JSON_FIELDS = {"source_refs", "metadata", "archive", "raw_report", "evidence", "contradicting_evidence",
                "raw_labels", "segment_refs", "reconstruction_artifact"}


class Repository:
    def __init__(self, database_url: str):
        self._database_url = database_url

    @classmethod
    def from_env(cls):
        url = os.environ.get("AGATHA_DATABASE_URL")
        if not url:
            raise ValueError("Set AGATHA_DATABASE_URL in the server's private environment file")
        return cls(url)

    def connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self._database_url, row_factory=dict_row, connect_timeout=10,
                               options="-c statement_timeout=60000", prepare_threshold=None)

    def migrate(self, directory: Path, *, storage: bool = True) -> list[str]:
        """Apply new migrations once, atomically; a changed applied file is an error."""
        applied = []
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(73401932)")
            conn.execute("CREATE TABLE IF NOT EXISTS public.agatha_migrations "
                         "(name text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz DEFAULT now())")
            conn.execute("ALTER TABLE public.agatha_migrations ENABLE ROW LEVEL SECURITY")
            conn.execute("REVOKE ALL ON public.agatha_migrations FROM PUBLIC")
            for role in ("anon", "authenticated"):
                if conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone():
                    from psycopg import sql
                    conn.execute(sql.SQL("REVOKE ALL ON public.agatha_migrations FROM {}").format(sql.Identifier(role)))
            paths = sorted(Path(directory).glob("[0-9]*.sql"))
            if not paths:
                raise ValueError("no migration files found")
            for path in paths:
                if not storage and path.name == "002_private_storage.sql":
                    continue
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                previous = conn.execute("SELECT sha256 FROM public.agatha_migrations WHERE name=%s", (path.name,)).fetchone()
                if previous:
                    if previous["sha256"] != checksum:
                        raise ValueError("applied migration changed; add a new migration instead")
                    continue
                conn.execute(path.read_text())
                conn.execute("INSERT INTO public.agatha_migrations(name,sha256) VALUES (%s,%s)", (path.name, checksum))
                applied.append(path.name)
        return applied

    @staticmethod
    def _insert(conn, table: str, row: dict) -> None:
        from psycopg import sql
        from psycopg.types.json import Jsonb

        query = sql.SQL("INSERT INTO public.{} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
            sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, row)),
            sql.SQL(",").join(sql.Placeholder() for _ in row))
        conn.execute(query, [Jsonb(v) if k in _JSON_FIELDS and v is not None else v for k, v in row.items()])

    def ingest(self, bundle: ImportBundle) -> str:
        with self.connect() as conn:
            self._insert(conn, "segments", bundle.segment)
            existing = conn.execute("SELECT metadata FROM public.segments WHERE id=%s", (bundle.segment["id"],)).fetchone()
            if existing["metadata"] != bundle.segment["metadata"]:
                raise ValueError("segment metadata changed without a preprocessing version change")
            self._insert(conn, "stage2_runs", bundle.run)
            for table, rows in (("incidents", bundle.incidents), ("incident_labels", bundle.labels),
                                ("incident_objects", bundle.objects), ("incident_evidence", bundle.evidence)):
                for row in rows:
                    self._insert(conn, table, row)
            for incident in bundle.incidents:
                self._insert(conn, "run_incidents", {"run_id": bundle.run["id"], "incident_id": incident["id"]})
            if bundle.human:
                self._insert(conn, "human_labels", bundle.human)
        return bundle.run["id"]

    def add_asset(self, asset: dict) -> None:
        with self.connect() as conn:
            self._insert(conn, "incident_assets", asset)

    def search(self, filters: SearchFilters, *, limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("invalid page size or offset")
        where, params = where_clause(filters)
        with self.connect() as conn:
            total = conn.execute(f"SELECT count(*) AS n FROM public.incidents i WHERE {where}", params).fetchone()["n"]
            rows = conn.execute(
                "SELECT i.*, s.context_name, s.split FROM public.incidents i "
                f"JOIN public.segments s ON s.id=i.segment_id WHERE {where} "
                "ORDER BY s.context_name, i.start_s, i.model_id, i.id LIMIT %s OFFSET %s", [*params, limit, offset]).fetchall()
        return rows, total

    def detail(self, incident_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute("SELECT i.*, s.context_name, s.split, s.dataset_version, s.source_refs, s.metadata "
                               "FROM public.incidents i JOIN public.segments s ON s.id=i.segment_id WHERE i.id=%s",
                               (incident_id,)).fetchone()
            if row is None:
                raise ValueError("incident not found")
            for name, table, order in (("labels", "incident_labels", "label"),
                                       ("objects", "incident_objects", "object_id"),
                                       ("evidence", "incident_evidence", "evidence_key"),
                                       ("assets", "incident_assets", "created_at DESC, id"),
                                       ("jobs", "simulation_jobs", "created_at DESC, id")):
                from psycopg import sql
                row[name] = conn.execute(sql.SQL("SELECT * FROM public.{} WHERE incident_id=%s ORDER BY " + order)
                                        .format(sql.Identifier(table)), (incident_id,)).fetchall()
        return row

    def scenes(self) -> list[dict]:
        """Includes scenes with zero reports, so human-only events remain reachable."""
        with self.connect() as conn:
            return conn.execute("SELECT id,context_name,split,preprocessing_version FROM public.segments "
                                "ORDER BY context_name,preprocessing_version").fetchall()

    def comparison(self, segment_id: str) -> dict:
        with self.connect() as conn:
            labels = conn.execute("SELECT DISTINCT ON (scene) * FROM public.human_labels WHERE segment_id=%s "
                                  "ORDER BY scene, created_at DESC, id DESC", (segment_id,)).fetchall()
            runs = conn.execute("SELECT * FROM public.stage2_runs WHERE segment_id=%s ORDER BY created_at DESC,id",
                                (segment_id,)).fetchall()
            for run in runs:
                run["reports"] = [r["raw_report"] for r in conn.execute(
                    "SELECT i.raw_report FROM public.incidents i JOIN public.run_incidents r ON r.incident_id=i.id "
                    "WHERE r.run_id=%s ORDER BY i.source_incident_id", (run["id"],)).fetchall()]
        return {"human_labels": labels, "runs": runs}

    def enqueue(self, incident_id: str) -> dict:
        """Queue only. Reuse a pending job atomically; no GPU work happens here."""
        from psycopg.types.json import Jsonb

        with self.connect() as conn:
            incident = conn.execute("SELECT i.*,s.source_refs,s.last_t_s FROM public.incidents i "
                                    "JOIN public.segments s ON s.id=i.segment_id WHERE i.id=%s FOR UPDATE OF i",
                                    (incident_id,)).fetchone()
            if incident is None:
                raise ValueError("incident not found")
            pending = conn.execute("SELECT * FROM public.simulation_jobs WHERE incident_id=%s "
                                   "AND status IN ('QUEUED','RUNNING')", (incident_id,)).fetchone()
            if pending:
                return pending
            return conn.execute(
                "INSERT INTO public.simulation_jobs "
                "(id,incident_id,segment_id,segment_refs,start_s,end_s) VALUES (%s,%s,%s,%s,%s,%s) RETURNING *",
                (uuid4(), incident_id, incident["segment_id"], Jsonb(incident["source_refs"]),
                 max(0.0, min(incident["start_s"], incident["last_t_s"])),
                 max(0.0, min(incident["end_s"], incident["last_t_s"])))).fetchone()
