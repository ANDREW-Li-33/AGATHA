"""Real, disposable PostgreSQL integration tests. No cloud account or Dataset needed."""

import copy
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg")

from agatha.database.ingest import prepare_import, read_json
from agatha.database.repository import Repository
from agatha.database.search import SearchFilters
from stage3_helpers import scene_files, write_json, write_results

SQL = Path(__file__).parents[1] / "infra/sql"


@pytest.fixture(scope="module")
def postgres():
    initdb = shutil.which("initdb")
    if not initdb:
        pytest.skip("local PostgreSQL binaries required for Stage 3 integration tests")
    pg_ctl = str(Path(initdb).with_name("pg_ctl"))
    # Short Unix socket path; no network listener or persistent local service.
    with tempfile.TemporaryDirectory(prefix="agatha-pg-") as temp:
        temp = Path(temp)
        data = temp / "data"
        subprocess.run([initdb, "-D", str(data), "-A", "trust", "-U", "postgres", "--no-locale", "--encoding=UTF8"],
                       check=True, capture_output=True)
        socket = temp / "s"
        socket.mkdir()
        subprocess.run([pg_ctl, "-D", str(data), "-l", str(temp / "postgres.log"), "-o",
                        f"-F -k {socket} -c listen_addresses=''", "-w", "start"], check=True, capture_output=True)
        try:
            yield f"host={socket} user=postgres dbname=postgres"
        finally:
            subprocess.run([pg_ctl, "-D", str(data), "-m", "immediate", "-w", "stop"], check=True, capture_output=True)


@pytest.fixture
def repository(postgres):
    name = "test_" + uuid4().hex
    with psycopg.connect(postgres, autocommit=True) as conn:
        conn.execute(psycopg.sql.SQL("CREATE DATABASE {}").format(psycopg.sql.Identifier(name)))
    repo = Repository(postgres.replace("dbname=postgres", f"dbname={name}"))
    repo.migrate(SQL, storage=False)
    yield repo
    with psycopg.connect(postgres, autocommit=True) as conn:
        conn.execute(psycopg.sql.SQL("DROP DATABASE {}").format(psycopg.sql.Identifier(name)))


def counts(repository):
    tables = ("segments", "stage2_runs", "incidents", "incident_labels", "incident_objects", "incident_evidence",
              "run_incidents", "human_labels", "simulation_jobs", "incident_assets")
    with repository.connect() as conn:
        return {table: conn.execute(psycopg.sql.SQL("SELECT count(*) n FROM public.{}").format(
            psycopg.sql.Identifier(table))).fetchone()["n"] for table in tables}


def test_migrations_repeat_and_bucket_is_private(repository):
    assert repository.migrate(SQL, storage=False) == []
    with repository.connect() as conn:
        # Supabase-shaped storage table so the actual Storage migration is executed too.
        conn.execute("CREATE SCHEMA storage")
        conn.execute("CREATE TABLE storage.buckets(id text PRIMARY KEY,name text,public boolean,"
                     "file_size_limit bigint,allowed_mime_types text[])")
        conn.execute("INSERT INTO storage.buckets VALUES ('agatha-incidents','agatha-incidents',true,null,null)")
    assert repository.migrate(SQL) == ["002_private_storage.sql"]
    assert repository.migrate(SQL) == []
    with repository.connect() as conn:
        assert conn.execute("SELECT public FROM storage.buckets").fetchone()["public"] is False
        assert all(row["relrowsecurity"] for row in conn.execute(
            "SELECT relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname='public' AND c.relkind='r'").fetchall())


def test_changed_applied_migration_is_refused(repository, tmp_path):
    shutil.copytree(SQL, tmp_path / "sql")
    p = tmp_path / "sql/001_stage3.sql"
    p.write_text(p.read_text() + "\n-- changed\n")
    with pytest.raises(ValueError, match="migration changed"):
        repository.migrate(tmp_path / "sql", storage=False)


def test_atomic_idempotent_import_and_revision_preservation(repository, tmp_path):
    folder = scene_files(tmp_path)
    bundle = prepare_import(folder)
    repository.ingest(bundle)
    first = counts(repository)
    repository.ingest(bundle)
    assert counts(repository) == first
    detail = repository.detail(bundle.incidents[0]["id"])
    assert detail["raw_report"] == bundle.incidents[0]["raw_report"]
    assert len(detail["evidence"]) == 3
    raw = read_json(folder / "stage2/incidents.json")
    for r in raw:
        r["model_id"] = "second-model"
    write_results(folder / "stage2/by_model/second", raw)
    repository.ingest(prepare_import(folder, folder / "stage2/by_model/second"))
    assert counts(repository)["incidents"] == 8
    bad = copy.deepcopy(bundle)
    bad.run["id"] = str(uuid4())
    bad.objects[0]["incident_id"] = str(uuid4())
    before = counts(repository)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        repository.ingest(bad)
    assert counts(repository) == before


def test_search_defaults_rejected_labels_any_all_and_or_status_gate(repository, tmp_path):
    folder = scene_files(tmp_path)
    bundle = prepare_import(folder)
    repository.ingest(bundle)
    assert repository.search(SearchFilters())[1] == 1
    assert repository.search(SearchFilters(include_duplicates=True))[1] == 2
    assert repository.search(SearchFilters(statuses=[]))[1] == 0
    rejected = SearchFilters(statuses=["REJECTED"], categories=["DYNAMIC_HAZARD"], labels=["SUDDEN_CUT_IN"])
    assert repository.search(rejected)[1] == 1
    assert repository.search(SearchFilters(labels=["SUDDEN_CUT_IN", "ROAD_DEBRIS"], label_mode="any"))[1] == 1
    assert repository.search(SearchFilters(labels=["SUDDEN_CUT_IN", "ROAD_DEBRIS"], label_mode="all"))[1] == 0
    assert repository.search(SearchFilters(object_classes=["pedestrian"]))[1] == 0
    assert repository.search(SearchFilters(object_classes=["vehicle"], min_duration_s=0.3, max_duration_s=0.5))[1] == 1
    assert repository.search(SearchFilters(labels=["ROAD_DEBRIS"], object_classes=["vehicle"], combine="AND"))[1] == 0
    assert repository.search(SearchFilters(labels=["ROAD_DEBRIS"], object_classes=["vehicle"], combine="OR"))[1] == 1
    # Duration bounds stay together inside OR; status still cannot leak.
    assert repository.search(SearchFilters(min_duration_s=0.5, max_duration_s=0.6, combine="OR"))[1] == 0
    assert repository.search(SearchFilters(statuses=["REJECTED"], combine="OR", min_duration_s=0))[1] == 1
    rows, total = repository.search(SearchFilters(include_duplicates=True), limit=1, offset=1)
    assert len(rows) == 1 and total == 2


def test_rejected_secondary_label_does_not_match_confirmed_event(repository, tmp_path):
    folder = scene_files(tmp_path)
    raw = read_json(folder / "stage2/incidents.json")[:1]
    raw[0]["label_assessments"].append({"label": "ROAD_DEBRIS", "status": "REJECTED", "evidence": [],
                                        "contradicting_evidence": ["No debris"]})
    write_results(folder / "stage2", raw)
    repository.ingest(prepare_import(folder))
    assert repository.search(SearchFilters(labels=["ROAD_DEBRIS"]))[1] == 0
    assert repository.search(SearchFilters(categories=["ROAD_OBSTRUCTION"]))[1] == 0


def test_jobs_are_queue_only_reused_and_keep_stage4_contract(repository, tmp_path):
    bundle = prepare_import(scene_files(tmp_path))
    repository.ingest(bundle)
    iid = bundle.incidents[0]["id"]
    with ThreadPoolExecutor(max_workers=4) as workers:
        jobs = list(workers.map(lambda _: repository.enqueue(iid), range(4)))
    assert len({job["id"] for job in jobs}) == 1
    job = jobs[0]
    assert job["status"] == "QUEUED" and job["reconstruction_artifact"] is None
    assert job["segment_refs"] == bundle.segment["source_refs"]
    assert job["start_s"] == 0.4 and job["end_s"] == 0.8
    assert job["output_clip_path"] is None and job["error"] is None
    with repository.connect() as conn:
        conn.execute("UPDATE public.simulation_jobs SET status='FAILED',error='test failure' WHERE id=%s", (job["id"],))
    assert repository.enqueue(iid)["id"] != job["id"]
    with pytest.raises(ValueError, match="not found"):
        repository.enqueue(str(uuid4()))


def test_human_revisions_and_zero_report_scene_are_retrievable(repository, tmp_path):
    folder = scene_files(tmp_path)
    write_results(folder / "stage2", [])
    label_path = tmp_path / "human.json"
    labels = {"scene": "random15_1", "reviewed": False, "no_incident": None, "incidents": []}
    write_json(label_path, labels)
    first = prepare_import(folder, human_path=label_path, scene="random15_1")
    repository.ingest(first)
    labels.update(reviewed=True, no_incident=True)
    write_json(label_path, labels)
    second = prepare_import(folder, human_path=label_path, scene="random15_1")
    repository.ingest(second)
    repository.ingest(first)  # replaying an old import must not make it the latest human review
    comparison = repository.comparison(first.segment["id"])
    assert comparison["human_labels"][0]["reviewed"]
    assert len(repository.scenes()) == 1 and repository.search(SearchFilters())[1] == 0
    assert comparison["runs"][0]["reports"] == [] and comparison["runs"][0]["complete"]
    assert counts(repository)["human_labels"] == 2


def test_table_privacy_blocks_a_browser_role(repository):
    role = "browser_" + uuid4().hex
    with repository.connect() as conn:
        conn.execute(psycopg.sql.SQL("CREATE ROLE {}").format(psycopg.sql.Identifier(role)))
        conn.execute(psycopg.sql.SQL("SET ROLE {}").format(psycopg.sql.Identifier(role)))
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with conn.transaction():
                conn.execute("SELECT * FROM public.incidents")
        conn.execute("RESET ROLE")
        conn.execute(psycopg.sql.SQL("DROP ROLE {}").format(psycopg.sql.Identifier(role)))


def test_real_database_through_explorer_to_queue(repository, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    bundle = prepare_import(scene_files(tmp_path))
    repository.ingest(bundle)
    at = AppTest.from_string("""
import streamlit as st
from agatha.database.explorer import render_app
render_app(st.session_state.repository)
""")
    at.session_state.repository = repository
    at.run(timeout=15)
    assert not at.exception
    next(button for button in at.button if button.label == "Run Simulation").click().run(timeout=15)
    assert not at.exception
    assert repository.detail(bundle.incidents[0]["id"])["jobs"][0]["status"] == "QUEUED"
