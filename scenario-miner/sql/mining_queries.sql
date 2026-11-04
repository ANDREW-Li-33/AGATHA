-- Example scenario-mining queries. Run with: miner query sql/mining_queries.sql
-- `latest` = the most recent run of each strategy/backend.

CREATE OR REPLACE TEMP VIEW latest AS
SELECT * FROM runs QUALIFY row_number() OVER (PARTITION BY backend, model, strategy ORDER BY created_at DESC) = 1;

-- 1. Construction zones with a pedestrian on the road within 20 m (best VLM run by recency)
SELECT p.sample_id, s.time_of_day, round(p.confidence, 2) AS conf,
       count(o.object_id) AS peds_on_road
FROM predictions p
JOIN samples s USING (sample_id)
JOIN objects o ON o.sample_id = p.sample_id AND o.in_camera AND o.type = 'pedestrian'
               AND o.range_m <= 20 AND (o.lane_dist_m <= 1.5 OR o.in_crosswalk)
WHERE p.label = 'construction_zone' AND p.present
  AND p.run_id = (SELECT run_id FROM latest WHERE backend <> 'heuristic' ORDER BY created_at DESC LIMIT 1)
GROUP BY ALL
ORDER BY conf DESC;

-- 2. Temporal smoothing: road_change windows where >= 2 of 3 consecutive frames agree
--    (one-frame blips are usually hallucinations or occlusion)
SELECT segment_id, sample_id, timestamp_us, votes_in_window
FROM (
    SELECT s.segment_id, p.sample_id, s.timestamp_us,
           sum(p.present::INT) OVER (PARTITION BY s.segment_id ORDER BY s.timestamp_us
                                     ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS votes_in_window,
           p.present
    FROM predictions p JOIN samples s USING (sample_id)
    WHERE p.label = 'road_change'
      AND p.run_id = (SELECT run_id FROM latest WHERE backend <> 'heuristic' ORDER BY created_at DESC LIMIT 1)
)
WHERE present AND votes_in_window >= 2
ORDER BY segment_id, timestamp_us;

-- 3. Labeling queue: frames where the VLM and the geometric baseline disagree on road_change.
--    Spending the human-labeling budget here finds errors fastest.
SELECT v.sample_id, v.present AS vlm, round(v.confidence, 2) AS vlm_conf, h.present AS geometry,
       g.present AS human
FROM predictions v
JOIN predictions h ON h.sample_id = v.sample_id AND h.label = v.label
LEFT JOIN ground_truth g ON g.sample_id = v.sample_id AND g.label = v.label
WHERE v.label = 'road_change' AND v.present <> h.present
  AND v.run_id = (SELECT run_id FROM latest WHERE backend <> 'heuristic' ORDER BY created_at DESC LIMIT 1)
  AND h.run_id = (SELECT run_id FROM latest WHERE backend = 'heuristic' LIMIT 1)
ORDER BY abs(v.confidence - 0.5) ASC;

-- 4. Map-discrepancy candidates straight from 3D: static objects sitting on mapped travel lanes
SELECT s.segment_id, o.sample_id, count(*) AS static_on_lane, round(min(o.range_m), 1) AS nearest_m
FROM objects o JOIN samples s USING (sample_id)
WHERE o.type = 'sign' AND o.in_camera AND o.lane_dist_m <= 1.5
GROUP BY ALL
ORDER BY static_on_lane DESC
LIMIT 20;

-- 5. Cost and robustness per run: what would 1M frames cost in tokens and time?
SELECT r.strategy, r.backend || ':' || r.model AS model, count(*) AS frames,
       round(100.0 * count(*) FILTER (WHERE x.error IS NOT NULL) / count(*), 1) AS pct_errors,
       round(avg(x.latency_s), 2) AS s_per_frame,
       round(avg(x.input_tokens)) AS in_tok, round(avg(x.output_tokens)) AS out_tok,
       round(avg(x.input_tokens + x.output_tokens) * 1e6 / 1e9, 2) AS billion_tokens_per_1M_frames
FROM runs r JOIN responses x USING (run_id)
GROUP BY ALL
ORDER BY r.strategy;
