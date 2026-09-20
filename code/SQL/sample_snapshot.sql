CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_nonmeds_sample_1m` AS
SELECT sl.*, t.person_id AS person_id_1, t.max_number
FROM `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_nonmeds` AS sl
JOIN (
    SELECT person_id, MAX(row_num) AS max_number
    FROM `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_nonmeds`
    GROUP BY person_id
) AS t
ON t.person_id = sl.person_id
WHERE sl.SDI IS NOT NULL AND t.max_number < 1000
ORDER BY RAND()
LIMIT 1000000;

-- 39 is the 99th percentile of month_difference.
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered_nonmeds_sample_1m` AS
SELECT *
FROM `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered_nonmeds_all`
WHERE SDI IS NOT NULL AND month_difference < 39
ORDER BY RAND()
LIMIT 1000000;
