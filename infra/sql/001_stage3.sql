-- Apply once through agatha-stage3 migrate (transactional, checksum tracked).
-- All access is server-side. No browser/anon/authenticated grants or RLS policies.
CREATE TABLE public.segments (
    id uuid PRIMARY KEY,
    context_name text NOT NULL,
    split text NOT NULL CHECK (split IN ('training', 'validation', 'testing', 'test')),
    dataset_version text NOT NULL,
    preprocessing_version text NOT NULL,
    t0_micros bigint NOT NULL,
    last_t_s double precision NOT NULL CHECK (last_t_s >= 0 AND last_t_s < 'Infinity'::float8),
    source_refs jsonb NOT NULL,
    metadata jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (context_name, split, dataset_version, preprocessing_version)
);

CREATE TABLE public.stage2_runs (
    id uuid PRIMARY KEY,
    segment_id uuid NOT NULL REFERENCES public.segments(id),
    model_ids text[] NOT NULL,
    complete boolean NOT NULL,
    archive jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE public.incidents (
    id uuid PRIMARY KEY,
    segment_id uuid NOT NULL REFERENCES public.segments(id),
    source_incident_id text NOT NULL,
    report_hash text NOT NULL,
    duplicate_of uuid REFERENCES public.incidents(id) DEFERRABLE INITIALLY DEFERRED,
    start_s double precision NOT NULL CHECK (start_s >= -0.05),
    end_s double precision NOT NULL CHECK (end_s >= start_s AND end_s < 'Infinity'::float8),
    duration_s double precision GENERATED ALWAYS AS (end_s - start_s) STORED,
    categories text[] NOT NULL,
    subcategories text[] NOT NULL,
    verification_status text NOT NULL CHECK (verification_status IN ('CONFIRMED','UNCERTAIN','REJECTED')),
    vlm_reasoning text NOT NULL,
    driving_impact text NOT NULL,
    raw_report jsonb NOT NULL CHECK (jsonb_typeof(raw_report) = 'object'),
    pipeline_version text NOT NULL,
    model_id text NOT NULL,
    prompt_version text NOT NULL,
    preprocessing_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (segment_id, report_hash),
    CHECK (duplicate_of IS DISTINCT FROM id)
);

CREATE TABLE public.run_incidents (
    run_id uuid NOT NULL REFERENCES public.stage2_runs(id),
    incident_id uuid NOT NULL REFERENCES public.incidents(id),
    PRIMARY KEY (run_id, incident_id)
);

CREATE TABLE public.incident_labels (
    incident_id uuid NOT NULL REFERENCES public.incidents(id),
    label text NOT NULL,
    category text NOT NULL CHECK (category IN ('DYNAMIC_HAZARD','CONSTRUCTION_DISRUPTION','ROAD_OBSTRUCTION')),
    status text NOT NULL CHECK (status IN ('CONFIRMED','UNCERTAIN','REJECTED')),
    evidence jsonb NOT NULL,
    contradicting_evidence jsonb NOT NULL,
    PRIMARY KEY (incident_id, label)
);

CREATE TABLE public.incident_objects (
    incident_id uuid NOT NULL REFERENCES public.incidents(id),
    object_id text NOT NULL,
    object_classes text[] NOT NULL,
    PRIMARY KEY (incident_id, object_id)
);

CREATE TABLE public.incident_evidence (
    incident_id uuid NOT NULL REFERENCES public.incidents(id),
    evidence_key text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('frame','supporting','contradicting')),
    label text,
    observation text,
    camera text,
    timestamp_s double precision,
    frame_timestamp_micros bigint,
    PRIMARY KEY (incident_id, evidence_key),
    FOREIGN KEY (incident_id, label) REFERENCES public.incident_labels(incident_id, label),
    CHECK ((kind = 'frame' AND camera IS NOT NULL AND frame_timestamp_micros IS NOT NULL
            AND timestamp_s IS NOT NULL AND timestamp_s > '-Infinity'::float8
            AND timestamp_s < 'Infinity'::float8)
        OR (kind <> 'frame' AND label IS NOT NULL AND observation IS NOT NULL))
);

CREATE TABLE public.incident_assets (
    id uuid PRIMARY KEY,
    incident_id uuid NOT NULL REFERENCES public.incidents(id),
    kind text NOT NULL CHECK (kind IN ('source_video','file')),
    bucket text NOT NULL,
    object_path text NOT NULL CHECK (object_path NOT LIKE '/%' AND object_path NOT LIKE '%..%'),
    sha256 text NOT NULL,
    content_type text NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes > 0),
    metadata jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (incident_id, kind, bucket, object_path)
);

CREATE TABLE public.human_labels (
    id uuid PRIMARY KEY,
    segment_id uuid NOT NULL REFERENCES public.segments(id),
    scene text NOT NULL,
    annotation_version text NOT NULL,
    reviewed boolean NOT NULL,
    no_incident boolean,
    raw_labels jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (segment_id, scene, annotation_version),
    CHECK (NOT reviewed OR no_incident IS NOT NULL)
);

CREATE TABLE public.simulation_jobs (
    id uuid PRIMARY KEY,
    incident_id uuid NOT NULL REFERENCES public.incidents(id),
    segment_id uuid NOT NULL REFERENCES public.segments(id),
    segment_refs jsonb NOT NULL,
    start_s double precision NOT NULL CHECK (start_s >= 0),
    end_s double precision NOT NULL CHECK (end_s >= start_s AND end_s < 'Infinity'::float8),
    reconstruction_artifact jsonb,
    status text NOT NULL DEFAULT 'QUEUED' CHECK (status IN ('QUEUED','RUNNING','SUCCEEDED','FAILED')),
    output_clip_path text,
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (status <> 'SUCCEEDED' OR output_clip_path IS NOT NULL),
    CHECK (status <> 'FAILED' OR error IS NOT NULL)
);

CREATE INDEX incidents_status_idx ON public.incidents(verification_status);
CREATE INDEX incidents_segment_time_idx ON public.incidents(segment_id, start_s, end_s);
CREATE INDEX incidents_duration_idx ON public.incidents(duration_s);
CREATE INDEX incidents_category_idx ON public.incidents USING gin(categories);
CREATE INDEX incidents_labels_idx ON public.incidents USING gin(subcategories);
CREATE INDEX incident_labels_search_idx ON public.incident_labels(label, status, incident_id);
CREATE INDEX incident_labels_category_idx ON public.incident_labels(category, incident_id);
CREATE INDEX incident_objects_class_idx ON public.incident_objects USING gin(object_classes);
CREATE INDEX incident_assets_incident_idx ON public.incident_assets(incident_id);
CREATE INDEX stage2_runs_segment_idx ON public.stage2_runs(segment_id, created_at DESC);
CREATE INDEX run_incidents_incident_idx ON public.run_incidents(incident_id);
CREATE INDEX human_labels_segment_idx ON public.human_labels(segment_id, scene, created_at DESC);
CREATE INDEX simulation_jobs_status_idx ON public.simulation_jobs(status, created_at);
CREATE INDEX simulation_jobs_incident_idx ON public.simulation_jobs(incident_id, created_at DESC);
-- A double click, browser refresh, or second viewer reuses the existing pending job.
CREATE UNIQUE INDEX simulation_jobs_pending_idx ON public.simulation_jobs(incident_id)
    WHERE status IN ('QUEUED','RUNNING');

DO $$
DECLARE table_name text; role_name text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY['segments','stage2_runs','incidents','run_incidents','incident_labels',
        'incident_objects','incident_evidence','incident_assets','human_labels','simulation_jobs'] LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', table_name);
        EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC', table_name);
        FOREACH role_name IN ARRAY ARRAY['anon','authenticated'] LOOP
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = role_name) THEN
                EXECUTE format('REVOKE ALL ON public.%I FROM %I', table_name, role_name);
            END IF;
        END LOOP;
        IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'service_role') THEN
            EXECUTE format('GRANT ALL ON public.%I TO service_role', table_name);
        END IF;
    END LOOP;
END $$;
