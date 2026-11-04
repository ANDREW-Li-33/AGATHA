-- Ground truth that costs no human time: exact rules over 3D boxes, the HD map,
-- and segment metadata. Only objects visible in this camera count, otherwise a
-- pedestrian behind the car would be a "miss" for the front camera.
CREATE OR REPLACE TEMP VIEW derived AS
SELECT
    s.sample_id,
    coalesce(bool_or(o.type = 'pedestrian' AND o.range_m <= 40
                     AND (o.lane_dist_m <= 1.5 OR o.in_crosswalk)), false)            AS pedestrian_in_roadway,
    coalesce(bool_or(o.type = 'cyclist' AND o.range_m <= 25), false)                  AS cyclist_nearby,
    coalesce(bool_or(o.type = 'vehicle' AND o.range_m <= 30
                     AND (o.length >= 7.0 OR o.height >= 3.0)), false)                AS large_vehicle,
    count(o.object_id) FILTER (WHERE o.type = 'pedestrian' AND o.range_m <= 30) >= 5  AS crowded_scene,
    any_value(s.time_of_day) = 'Night'                                                AS night,
    lower(any_value(s.weather)) LIKE '%rain%'                                         AS rain
FROM samples s
LEFT JOIN objects o ON o.sample_id = s.sample_id AND o.in_camera
GROUP BY s.sample_id;

-- A human correction made while auditing these rules always wins.
INSERT OR REPLACE INTO ground_truth
SELECT u.sample_id, u.label, u.present,
       CASE WHEN u.label IN ('night', 'rain') THEN 'stats' ELSE 'derived' END
FROM (
    UNPIVOT derived
    ON pedestrian_in_roadway, cyclist_nearby, large_vehicle, crowded_scene, night, rain
    INTO NAME label VALUE present
) u
WHERE u.present IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM ground_truth g
                  WHERE g.sample_id = u.sample_id AND g.label = u.label AND g.source = 'human');
