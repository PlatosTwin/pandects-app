# src/etl/definitions.py
from pathlib import Path
from dagster import Definitions, load_from_defs_folder
from etl.defs.jobs import etl_pipeline, cleanup_pipeline
# from etl.defs.resources import get_resources

defs = Definitions.merge(
    load_from_defs_folder(project_root=Path(__file__).parent.parent.parent),
    Definitions(
        jobs=[etl_pipeline, cleanup_pipeline],
        # resources=get_resources(),
    ),
)
