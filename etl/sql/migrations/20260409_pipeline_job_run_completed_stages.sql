ALTER TABLE pipeline_job_runs
ADD COLUMN IF NOT EXISTS completed_stages_json LONGTEXT NULL AFTER current_stage;
