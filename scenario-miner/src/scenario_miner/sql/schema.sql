-- One row per camera image.
CREATE TABLE IF NOT EXISTS samples (
    sample_id     VARCHAR PRIMARY KEY,
    segment_id    VARCHAR NOT NULL,
    timestamp_us  BIGINT  NOT NULL,
    camera        VARCHAR NOT NULL,
    sample_dir    VARCHAR NOT NULL,
    ego_speed     DOUBLE,
    time_of_day   VARCHAR,
    weather       VARCHAR,
    location      VARCHAR,
    source        VARCHAR
);

-- 3D boxes with map-relative features, in the vehicle frame at capture time.
CREATE TABLE IF NOT EXISTS objects (
    sample_id        VARCHAR NOT NULL,
    object_id        VARCHAR NOT NULL,
    type             VARCHAR NOT NULL,
    x DOUBLE, y DOUBLE, z DOUBLE,
    length DOUBLE, width DOUBLE, height DOUBLE, heading DOUBLE,
    speed            DOUBLE,
    range_m          DOUBLE,
    lane_dist_m      DOUBLE,           -- NULL when the frame has no lane map
    in_crosswalk     BOOLEAN NOT NULL,
    in_camera        BOOLEAN NOT NULL, -- projects into this camera's image
    num_lidar_points INTEGER,
    PRIMARY KEY (sample_id, object_id)
);

CREATE TABLE IF NOT EXISTS ground_truth (
    sample_id VARCHAR NOT NULL,
    label     VARCHAR NOT NULL,
    present   BOOLEAN NOT NULL,
    source    VARCHAR NOT NULL,        -- human | derived | stats | synthetic
    PRIMARY KEY (sample_id, label)
);

-- eval: scored. fewshot: example pool, from segments disjoint from eval.
CREATE TABLE IF NOT EXISTS splits (
    sample_id VARCHAR PRIMARY KEY,
    split     VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      VARCHAR PRIMARY KEY,
    backend     VARCHAR NOT NULL,
    model       VARCHAR NOT NULL,
    strategy    VARCHAR NOT NULL,
    votes       INTEGER NOT NULL,
    temperature DOUBLE  NOT NULL,
    created_at  TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS responses (
    run_id        VARCHAR NOT NULL,
    sample_id     VARCHAR NOT NULL,
    raw_text      VARCHAR,
    observations  VARCHAR,
    error         VARCHAR,
    latency_s     DOUBLE,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    PRIMARY KEY (run_id, sample_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id     VARCHAR NOT NULL,
    sample_id  VARCHAR NOT NULL,
    label      VARCHAR NOT NULL,
    present    BOOLEAN NOT NULL,
    confidence DOUBLE  NOT NULL,
    PRIMARY KEY (run_id, sample_id, label)
);

-- Re-running an experiment never pays for the same API call twice.
CREATE TABLE IF NOT EXISTS response_cache (
    cache_key     VARCHAR PRIMARY KEY,
    raw_text      VARCHAR,
    latency_s     DOUBLE,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    created_at    TIMESTAMP DEFAULT current_timestamp
);

-- Precision / recall / F1 per run and label, in plain SQL. Few-shot examples are excluded.
CREATE OR REPLACE VIEW label_metrics AS
SELECT
    p.run_id,
    p.label,
    count(*)                                           AS n,
    count(*) FILTER (WHERE g.present)                  AS support,
    count(*) FILTER (WHERE p.present AND g.present)     AS tp,
    count(*) FILTER (WHERE p.present AND NOT g.present) AS fp,
    count(*) FILTER (WHERE NOT p.present AND g.present) AS fn,
    tp::DOUBLE / nullif(tp + fp, 0)                    AS precision,
    tp::DOUBLE / nullif(tp + fn, 0)                    AS recall,
    2.0 * tp / nullif(2 * tp + fp + fn, 0)             AS f1
FROM predictions p
JOIN ground_truth g USING (sample_id, label)
LEFT JOIN splits s USING (sample_id)
WHERE coalesce(s.split, 'eval') <> 'fewshot'
GROUP BY p.run_id, p.label;
