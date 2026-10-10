"""uvicorn entry point: `uvicorn main:app`. All logic lives in api.py (create_app) so it can be tested."""
from api import create_app_from_env

app = create_app_from_env()
