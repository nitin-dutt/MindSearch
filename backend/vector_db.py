"""Qdrant connections, shared by everything that stores something there (chunk_store.py, doc_registry.py).

A location is either a folder (embedded Qdrant: tests and the evaluation harness) or the URL of a Qdrant server (the application, so
the API and the worker share one database). Embedded mode takes an exclusive file lock, so a folder must only ever be opened once per
process: that is why the clients are kept here, one per location, and not created by their users.
"""
import atexit

_clients = {}


def is_server(location: str) -> bool:
    return str(location).startswith(("http://", "https://"))


def get_qdrant_client(location: str = "./qdrant_db"):
    if location not in _clients:
        from qdrant_client import QdrantClient
        _clients[location] = QdrantClient(url=location, timeout=30) if is_server(location) else QdrantClient(path=location)
    return _clients[location]


def close_qdrant_client(location: str):
    """Release the file lock for a location (used before rebuilding an evaluation index, and at the end of tests)."""
    client = _clients.pop(location, None)
    if client is not None:
        client.close()


def close_all_qdrant_clients():
    for location in list(_clients):
        close_qdrant_client(location)


atexit.register(close_all_qdrant_clients)  # avoids a noisy local-client __del__ error at interpreter exit
