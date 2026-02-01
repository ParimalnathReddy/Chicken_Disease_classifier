import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

project_name = "chicken_disease_classifier"  # keep package names lowercase

list_of_files = [
    Path(".github/workflows/.gitkeep"),
    Path(f"src/{project_name}/__init__.py"),
    Path(f"src/{project_name}/components/__init__.py"),
    Path(f"src/{project_name}/utils/__init__.py"),
    Path(f"src/{project_name}/config/__init__.py"),
    Path(f"src/{project_name}/config/configuration.py"),
    Path(f"src/{project_name}/pipeline/__init__.py"),
    Path(f"src/{project_name}/entity/__init__.py"),
    Path(f"src/{project_name}/constants/__init__.py"),
    Path("config/config.yaml"),
    Path("setup.py"),
    Path("dvc.yaml"),
    Path("params.yaml"),
    Path("requirements.txt"),
    Path("research/trials.ipynb"),
    Path("README.md"),
    
]

for file_path in list_of_files:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.exists():
        file_path.touch()
        logging.info(f"Created file: {file_path}")
    else:
        logging.info(f"Already exists: {file_path}")
