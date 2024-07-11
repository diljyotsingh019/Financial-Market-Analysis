import os
from pathlib import Path
import logging

list_of_files = [
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_collection.py",
    "src/components/data_ingestion.py",
    "src/pipeline/__init__.py",
    "src/pipeline/building_pipeline.py",
    "src/pipeline/implementation_pipeline.py",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "src/logger/__init__.py",
    "src/logger/logging.py",
    "src/exception/__init__.py",
    "src/exception/exception.py",
    "test/unit/__init__.py",
    "test/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiment/experiments.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok= True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # Create an empty file