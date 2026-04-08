"""
meta_server/main.py — entrypoint for uvicorn
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("meta_server.log"),
    ],
)

from meta_server.api import app  # noqa: F401  (uvicorn needs this import)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("meta_server.main:app", host="0.0.0.0", port=8000, reload=False)
