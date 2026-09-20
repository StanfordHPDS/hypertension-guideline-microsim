--Creates a table for every individual and their first/last exposure to an anti-hypertensive medication
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` as 
select person_id, min(drug_era_start_date) as first_exposure, max(drug_era_end_date) as last_exposure,
count(distinct drug_concept_id) as drug_number from 
`som-nero-phi-sherrir-afc.afc0724phiomop.drug_era` de 
join `som-nero-phi-sherrir-afc.afc0724phiomop.concept` c 
on c.concept_id = de.drug_concept_id
where concept_id in (1395058, 974166, 978555, 907013, 1335471, 1340128, 1341927, 
1340128,1341927, 1363749,1308216,1310756,1373225,1331235, 1334456, 1342439,40235485, 
1351557,1346686, 1347384,1367500, 40226742, 1317640, 1308842,1332418, 1353776, 
1326012, 1318137, 1318853, 1319880, 1328165, 1307863)
group by person_id;

--create demographics table with birth date, race/ethnicity, gender, and SDI.
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` AS
    SELECT
        p.person_id,
        p.birth_datetime,
        c.concept_name AS race,
        c1.concept_name AS ethnicity,
        c2.concept_name AS gender,
        sd.SDI_CT
    FROM `som-nero-phi-sherrir-afc.afc0724phiomop.person` p
    LEFT JOIN `som-nero-phi-sherrir-afc.afc0724phiomop.concept` c ON c.concept_id = p.race_concept_id
    LEFT JOIN `som-nero-phi-sherrir-afc.afc0724phiomop.concept` c1 ON c1.concept_id = p.ethnicity_concept_id
    LEFT JOIN `som-nero-phi-sherrir-afc.afc0724phiomop.concept` c2 ON c2.concept_id = p.gender_concept_id
    LEFT JOIN `som-nero-phi-sherrir-afc.afc0724phi.GeneratedPatientBaseline` sd ON sd.patientuid = p.person_source_value;


--obtain systolic BP ranges
select APPROX_QUANTILES(value_as_number, 100)[OFFSET(1)] AS percentile_1,
APPROX_QUANTILES(value_as_number, 100)[OFFSET(99)] AS percentile_99 from
`som-nero-phi-sherrir-afc.afc0724phiomop.measurement`
where measurement_concept_id = 3004249;

--obtain systolic BP ranges
select APPROX_QUANTILES(value_as_number, 100)[OFFSET(1)] AS percentile_1,
APPROX_QUANTILES(value_as_number, 100)[OFFSET(99)] AS percentile_99 from
`som-nero-phi-sherrir-afc.afc0724phiomop.measurement`
where measurement_concept_id = 3012888;

--Create a set of systolic and diastolic blood pressure observations that occured at the same time 
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` as 
WITH BP_measurements AS (
    SELECT
    DISTINCT
    person_id,
    measurement_datetime,
    measurement_concept_id,
    value_as_number
FROM `som-nero-phi-sherrir-afc.afc0724phiomop.measurement`
WHERE measurement_concept_id in (3004249,3012888) and value_as_number > 0
)
SELECT 
    s.person_id, 
    s.measurement_datetime, 
    s.measurement_concept_id, 
    s.value_as_number as systolic_bp, 
    s2.value_as_number as diastolic_bp,
    ROW_NUMBER() OVER (PARTITION BY s.person_id ORDER BY s.measurement_datetime) AS row_num
 FROM BP_measurements s
 JOIN BP_measurements s2
ON s.person_id = s2.person_id and s.measurement_datetime = s2.measurement_datetime
WHERE s.measurement_concept_id = 3004249 and s.value_as_number >= 86 and s.value_as_number <= 178
AND s2.measurement_concept_id = 3012888 and s2.value_as_number >= 50 and s2.value_as_number <=102;

--Find the previous systolic and diastolic BP value and the number of days/months since the previous time
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered`  AS 
SELECT
    A.person_id,
    case when p.gender = 'FEMALE' then 1
    else 0 end as female,
    date_diff(b.measurement_datetime, p.birth_datetime, year) as age,
    p.race as race, 
    p.ethnicity as ethnicity,
    p.SDI_CT as SDI,
    a.systolic_bp as last_systolic_bp,
    a.diastolic_bp as last_diastolic_bp,
    A.measurement_datetime as last_date,
    B.measurement_datetime AS this_date,
    b.systolic_bp as systolic_bp,
    b.diastolic_bp as diastolic_bp,
    a.row_num as row_number,
    DATE_DIFF(B.measurement_datetime,A.measurement_datetime, day) AS days_difference,
    DATE_DIFF(B.measurement_datetime,A.measurement_datetime, month) AS month_difference
FROM
    `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` A
JOIN
    `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` B 
    ON A.person_id = B.person_id AND A.row_num = B.row_num - 1
JOIN `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on A.person_id = p.person_id
where date_diff(b.measurement_datetime, p.birth_datetime, year) > 21 and b.measurement_datetime >= '2018-01-01';

--all measurements that took place within 6 months of their previous and while they had not been exposed to 
--any anti-hypertensive medications
--used for systolic and diastolic correlations 
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered_nonmeds` AS 
select 
    sl.*, 
    hm.first_exposure, 
    hm.last_exposure, 
    DATE_DIFF(sl.this_date,hm.first_exposure, day) as days_to_medicine from 
    `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered` sl 
    left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
    on sl.person_id = hm.person_id
    where (sl.this_date < hm.first_exposure or hm.first_exposure is NULL) and (month_difference >0 and month_difference <7);

--ordered version of the BP measurements prior to any anti-hypertensive medication exposure 
--at least one month difference between the previous measurement
--used to model the measurement frequency prior to anti-hypertensive treatment
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered_nonmeds_all` AS 
select 
    sl.*, 
    hm.first_exposure, 
    hm.last_exposure, 
    DATE_DIFF(sl.this_date,hm.first_exposure, day) as days_to_medicine from 
    `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered` sl 
    left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
    on sl.person_id = hm.person_id
    where (sl.this_date < hm.first_exposure or hm.first_exposure is NULL) and month_difference > 0;

--ordered version of the BP measurements after any anti-hypertensive medication exposure 
--at least one month difference between the previous measurement
--used to model the measurement frequency after anti-hypertensive treatment
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered_postmeds_all` AS 
select 
    sl.*, 
    hm.first_exposure, 
    hm.last_exposure, 
    DATE_DIFF(sl.this_date,hm.first_exposure, day) as days_to_medicine from 
    `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered` sl 
    left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
    on sl.person_id = hm.person_id
    where sl.this_date >= hm.first_exposure and month_difference > 0;

--SQL table used to create the systolic and diastolic quantile regressions used in the hypertension case study model
--all measurements prior to any anti-hypertensive medication exposure
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_nonmeds` AS 
select 
    DISTINCT
    sl.*,
    case when p.gender = 'FEMALE' then 1
    else 0 end as female,
    date_diff(sl.measurement_datetime, p.birth_datetime, year) as age,
    p.race as race, 
    p.ethnicity as ethnicity,
    p.SDI_CT as SDI,
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` sl 
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
on sl.person_id = hm.person_id
JOIN `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on sl.person_id = p.person_id
where (sl.measurement_datetime < hm.first_exposure or hm.first_exposure is NULL) and sl.measurement_datetime >= '2018-01-01';

--Create SDI table 
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.CT_SDI` as 
select distinct CONCAT(fips_state, fips_county, tract) as CT, SDI_CT from `som-nero-phi-sherrir-afc.afc0724phi.GeneratedPatientBaseline`;


-- first BP measurement per person with stage 1 hypertension
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.first_S1_stage` as 
select t.person_id, t.min_date, avg(m.systolic_bp) as systolic_bp , avg(m.diastolic_bp) as diastolic_bp
from 
(select person_id, min(measurement_datetime) as min_date
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values`
where (systolic_bp >= 130 or diastolic_bp >= 80)
group by person_id) as t 
join `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` m
on m.person_id = t.person_id and t.min_date = m.measurement_datetime 
group by t.person_id, t.min_date;

--first BP measurement per person and merged with demographics 
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.first_S1_stage_person` as 
select s.*, m.first_exposure, m.last_exposure, m.drug_number,
date_diff(s.min_date, d.birth_dateTIME, year) as age, 
d.race, d.ethnicity, d.gender, d.SDI_CT
from `som-nero-phi-sherrir-afc.mmcusick_afc.first_S1_stage` s
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` m 
on s.person_id = m.person_id
left join `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` d 
on d.person_id = s.person_id
where date_diff(s.min_date, d.birth_dateTIME, year) > 21;

--stage 1 and 2 measurements prior to any anti-hypertensive medication exposure 
--used in the treatment_logit_model.py file 
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.stage_1_2_obs`  AS
select distinct 
s.*, 
m.first_exposure, 
m.last_exposure, 
m.drug_number,
date_diff(s.measurement_datetime, d.birth_dateTIME, year) as age, 
d.race, 
d.ethnicity, 
d.gender, 
d.SDI_CT, 
ROW_NUMBER() OVER (PARTITION BY s.person_id ORDER BY measurement_datetime) -1 AS prev_num --how high BP observations they had before
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` s 
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` m 
on s.person_id = m.person_id
left join `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` d 
on d.person_id = s.person_id
where (systolic_bp >= 130 or diastolic_bp >= 80) and date_diff(s.measurement_datetime, d.birth_dateTIME, year) > 21
and (measurement_datetime <= first_exposure or first_exposure is null);

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.cumulative_med_exposure` AS
select 
    DISTINCT
    sl.*,
    case when p.gender = 'FEMALE' then 1
    else 0 end as female,
    date_diff(sl.measurement_datetime, p.birth_datetime, year) as age,
    p.race as race, 
    p.ethnicity as ethnicity,
    p.SDI_CT as SDI,
    first_exposure,
    case when hm.first_exposure <= sl.measurement_datetime then 1 
    else 0 end as cum_exposure,
    case when (lower(race) like '%black%' or lower(race) like '%african%') and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHB, 
    case when race = 'White' and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHW 
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` sl 
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
on sl.person_id = hm.person_id
JOIN `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on sl.person_id = p.person_id
where sl.measurement_datetime >= '2018-01-01' and date_diff(sl.measurement_datetime, p.birth_datetime, year) > 39
and date_diff(sl.measurement_datetime, p.birth_datetime, year) <= 90;

--COHORTS for constructing the Kaplan-Meier curve
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_40y` as
select * from (
select sl.person_id, 
    p.birth_datetime, 
    p.gender,
    max(sl.measurement_datetime) as max_follow_up, 
    date_diff(max(sl.measurement_datetime), 
    p.birth_datetime, year) as follow_up_age, 
    case when (lower(race) like '%black%' or lower(race) like '%african%') and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHB, 
    case when race = 'White' and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHW
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` sl 
join  `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on sl.person_id = p.person_id
WHERE (EXTRACT(YEAR FROM DATE '2018-01-01') - EXTRACT(YEAR FROM DATE(p.birth_datetime)) = 40)
group by sl.person_id, p.birth_datetime, p.gender, race, ethnicity) t 
where follow_up_age >= 40 and (NHB = 1 or NHW = 1);

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_40y_eligible`  AS
select distinct t.person_id, t.gender, NHW, NHB, hm.first_exposure,t.follow_up_age, date_diff(hm.first_exposure, t.birth_datetime, year) as age
from 
(select distinct t.person_id, t.gender,  NHW, NHB, t.birth_datetime, t.follow_up_age from
`som-nero-phi-sherrir-afc.mmcusick_afc.patients_40y` t 
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` c 
on t.person_id = c.person_id
where c.first_exposure is null or c.first_exposure >= '2018-01-01') t
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm
on hm.person_id = t.person_id;

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_46y` as
select * from (
select sl.person_id, 
    p.birth_datetime,
    p.gender,
    max(sl.measurement_datetime) as max_follow_up, 
    date_diff(max(sl.measurement_datetime), p.birth_datetime, year) as follow_up_age, 
    case when (lower(race) like '%black%' or lower(race) like '%african%') and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHB, 
    case when race = 'White' and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHW
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` sl 
join  `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on sl.person_id = p.person_id
WHERE (EXTRACT(YEAR FROM DATE '2018-01-01') - EXTRACT(YEAR FROM DATE(p.birth_datetime)) = 46)
group by sl.person_id, p.birth_datetime, p.gender, race, ethnicity) t 
where follow_up_age >= 46 and (NHB = 1 or NHW = 1);

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_46y_eligible`  AS
select distinct t.person_id, t.gender, NHW, NHB, hm.first_exposure, t.follow_up_age, date_diff(hm.first_exposure, t.birth_datetime, year) as age
from 
(select distinct t.person_id,t.gender, NHW, NHB, t.birth_datetime, t.follow_up_age from
`som-nero-phi-sherrir-afc.mmcusick_afc.patients_46y` t 
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` c 
on t.person_id = c.person_id
where c.first_exposure is null or c.first_exposure >= '2018-01-01') t
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
on hm.person_id = t.person_id;


CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_52y` as
select * from (
select sl.person_id, p.gender, p.birth_datetime, max(sl.measurement_datetime) as max_follow_up, 
date_diff(max(sl.measurement_datetime), p.birth_datetime, year) as follow_up_age, 
case when (lower(race) like '%black%' or lower(race) like '%african%') and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHB, 
    case when race = 'White' and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHW
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` sl 
join  `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on sl.person_id = p.person_id
WHERE (EXTRACT(YEAR FROM DATE '2018-01-01') - EXTRACT(YEAR FROM DATE(p.birth_datetime)) = 52)
group by sl.person_id, p.birth_datetime, p.gender, race, ethnicity) t 
where follow_up_age >= 52 and (NHB = 1 or NHW = 1);

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_52y_eligible`  AS
select distinct t.person_id, t.gender, NHB, NHW, hm.first_exposure, t.follow_up_age, date_diff(hm.first_exposure, t.birth_datetime, year) as age
from 
(select distinct t.person_id, t.gender, NHB, NHW, t.birth_datetime, t.follow_up_age from
`som-nero-phi-sherrir-afc.mmcusick_afc.patients_52y` t 
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` c 
on t.person_id = c.person_id
where c.first_exposure is null or c.first_exposure >= '2018-01-01') t
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
on hm.person_id = t.person_id;

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_58y` as
select * from (
select sl.person_id, p.gender, p.birth_datetime, max(sl.measurement_datetime) as max_follow_up, 
date_diff(max(sl.measurement_datetime), p.birth_datetime, year) as follow_up_age, 
case when (lower(race) like '%black%' or lower(race) like '%african%') and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHB, 
    case when race = 'White' and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHW
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` sl 
join  `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on sl.person_id = p.person_id
WHERE (EXTRACT(YEAR FROM DATE '2018-01-01') - EXTRACT(YEAR FROM DATE(p.birth_datetime)) = 58)
group by sl.person_id, p.birth_datetime, p.gender, race, ethnicity) t 
where follow_up_age >= 58 and (NHB = 1 or NHW = 1);

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_58y_eligible`  AS
select distinct t.person_id, t.gender,NHB, NHW ,hm.first_exposure, t.follow_up_age, date_diff(hm.first_exposure, t.birth_datetime, year) as age
from 
(select distinct t.person_id, t.gender, NHB, NHW, t.birth_datetime, t.follow_up_age from
`som-nero-phi-sherrir-afc.mmcusick_afc.patients_58y` t 
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` c 
on t.person_id = c.person_id
where c.first_exposure is null or c.first_exposure >= '2018-01-01') t
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
on hm.person_id = t.person_id;


CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_64y` as
select * from (
select sl.person_id, p.gender, p.birth_datetime, max(sl.measurement_datetime) as max_follow_up, 
date_diff(max(sl.measurement_datetime), p.birth_datetime, year) as follow_up_age, case when (lower(race) like '%black%' or lower(race) like '%african%') and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHB, 
    case when race = 'White' and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHW
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` sl 
join  `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on sl.person_id = p.person_id
WHERE (EXTRACT(YEAR FROM DATE '2018-01-01') - EXTRACT(YEAR FROM DATE(p.birth_datetime)) = 64)
group by sl.person_id, p.birth_datetime, p.gender, race, ethnicity) t 
where follow_up_age >= 64 and (NHW = 1 or NHB = 1);

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_64y_eligible`  AS
select distinct t.person_id,t.gender, NHB, NHW, hm.first_exposure, t.follow_up_age, date_diff(hm.first_exposure, t.birth_datetime, year) as age
from 
(select distinct t.person_id, t.gender,NHB, NHW, t.birth_datetime, t.follow_up_age from
`som-nero-phi-sherrir-afc.mmcusick_afc.patients_64y` t 
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` c 
on t.person_id = c.person_id
where c.first_exposure is null or c.first_exposure >= '2018-01-01') t
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm 
on hm.person_id = t.person_id;

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_70y` as
select * from (
select sl.person_id, p.gender, p.birth_datetime, max(sl.measurement_datetime) as max_follow_up, 
date_diff(max(sl.measurement_datetime), p.birth_datetime, year) as follow_up_age, case when (lower(race) like '%black%' or lower(race) like '%african%') and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHB, 
    case when race = 'White' and
    ethnicity = 'Not Hispanic or Latino' then 1 
    else 0 end as NHW
from `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values` sl 
join  `som-nero-phi-sherrir-afc.mmcusick_afc.demographics` p 
on sl.person_id = p.person_id
WHERE (EXTRACT(YEAR FROM DATE '2018-01-01') - EXTRACT(YEAR FROM DATE(p.birth_datetime)) = 70)
group by sl.person_id, p.birth_datetime, p.gender, race, ethnicity) t 
where follow_up_age >= 70 and (NHB = 1 or NHW = 1);

CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.patients_70y_eligible`  AS
select distinct t.person_id, t.gender,NHB, NHW, hm.first_exposure, t.follow_up_age, date_diff(hm.first_exposure, t.birth_datetime, year) as age
from
(select distinct t.person_id, t.gender,NHB, NHW, t.birth_datetime, t.follow_up_age from
`som-nero-phi-sherrir-afc.mmcusick_afc.patients_70y` t
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` c
on t.person_id = c.person_id
where c.first_exposure is null or c.first_exposure >= '2018-01-01') t
left join `som-nero-phi-sherrir-afc.mmcusick_afc.hypertension_meds` hm
on hm.person_id = t.person_id;

-- Pre-filtered follow-up dataset shared by validation_analysis.py
-- (export_validation_SDI_histogram) and systolic_diastolic_correlations.py
-- (read_in_afc_data). Self-joins SBP_DBP_values_ordered_nonmeds with the
-- per-person max(row_number) so downstream filters can cap rows-per-person.
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered_nonmeds_followups` AS
SELECT sl.*, t.person_id AS person_id_1, t.max_number
FROM `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered_nonmeds` sl
JOIN (
    SELECT person_id, MAX(row_number) AS max_number
    FROM `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_ordered_nonmeds`
    GROUP BY person_id
) t ON sl.person_id = t.person_id
WHERE sl.SDI IS NOT NULL
  AND sl.month_difference > 0 AND sl.month_difference < 7
  AND sl.age < 90
  AND t.max_number < 1000;

-- Pre-filtered input for treatment_logit_model.py (read_in_afc_data). Stage 1/2
-- observations after 2018-01-01 with non-null SDI_CT, joined with per-person
-- max(prev_num) to cap at fewer than 10 prior measurements per person.
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.stage_1_2_obs_with_max_prev_num` AS
SELECT s.*, t.max_number
FROM `som-nero-phi-sherrir-afc.mmcusick_afc.stage_1_2_obs` s
JOIN (
    SELECT person_id, MAX(prev_num) AS max_number
    FROM `som-nero-phi-sherrir-afc.mmcusick_afc.stage_1_2_obs`
    GROUP BY person_id
) t ON s.person_id = t.person_id
WHERE s.measurement_datetime > '2018-01-01'
  AND s.SDI_CT IS NOT NULL
  AND t.max_number < 10;

-- Pre-filtered input for cohort_sampling_validation.py (read_in_afc_data).
-- Age-40 rows with non-null SDI, joined with per-person max(row_num) to cap at
-- fewer than 1000 measurements per person.
CREATE OR REPLACE TABLE `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_nonmeds_age40_with_row_counts` AS
SELECT sl.*, t.person_id AS person_id_1, t.max_number
FROM `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_nonmeds` sl
JOIN (
    SELECT person_id, MAX(row_num) AS max_number
    FROM `som-nero-phi-sherrir-afc.mmcusick_afc.SBP_DBP_values_nonmeds`
    GROUP BY person_id
) t ON sl.person_id = t.person_id
WHERE sl.SDI IS NOT NULL AND sl.age = 40 AND t.max_number < 1000;
