"""
Utility script to wipe both Qdrant collections (missing & found) and re-create them.
Usage:
    python scripts/reset_vector_db.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from services.vector_db import VectorDatabaseService


def main():
    db = VectorDatabaseService()
    for collection in (db.missing_collection, db.found_collection):
        try:
            db.client.delete_collection(collection_name=collection)
            print(f"Deleted collection: {collection}")
        except Exception as exc:
            print(f"Failed to delete {collection}: {exc}")
    db.initialize_collections()
    print("Collections re-created. Database is now empty.")


if __name__ == "__main__":
    main()

