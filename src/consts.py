from pathlib import Path

PATH_PROJECT_DIR = Path(__file__).resolve().parents[1]
PATH_PROJECT_DATA = Path.joinpath(PATH_PROJECT_DIR, 'data')
PATH_PROJECT_REPORTs = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'reports')
PATH_PROJECT_FIGURES = Path.joinpath(PATH_PROJECT_DIR, 'reports', 'figures')
