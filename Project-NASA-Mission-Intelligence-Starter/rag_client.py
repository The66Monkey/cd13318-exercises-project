import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # 1. Find directories that look like ChromaDB persistent stores
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and ("chroma" in d.name.lower() or "db" in d.name.lower())
    ]

    # 2. Loop through each discovered directory
    for directory in chroma_dirs:
        try:
            # 3. Try connecting to the directory
            client = chromadb.PersistentClient(
                path=str(directory),
                settings=Settings(anonymized_telemetry=False)
            )

            # 4. Get collections
            collections = client.list_collections()

            # 5. Loop through collections
            for col in collections:
                key = f"{directory.name}:{col.name}"

                # Try to get document count (may fail on older versions)
                try:
                    count = col.count()
                except Exception:
                    count = "N/A"

                backends[key] = {
                    "directory": str(directory),
                    "collection": col.name,
                    "collection_name": col.name,   # <-- REQUIRED FOR chat.py
                    "display_name": f"{directory.name} / {col.name} ({count} docs)",
                    "count": count
                }


        except Exception as e:
            # 6. Handle inaccessible directories
            truncated_error = str(e)[:60] + "..."
            key = f"{directory.name}:ERROR"

            backends[key] = {
                "directory": str(directory),
                "collection": None,
                "display_name": f"{directory.name} (ERROR: {truncated_error})",
                "count": 0
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    return client.get_collection(collection_name)


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # 1. Default: no filter
    filter_dict = None

    # 2. Apply mission filter if valid
    if mission_filter and mission_filter.lower() not in ["all", "none"]:
        filter_dict = {"mission": mission_filter}

    # 3. Query the database
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filter_dict
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["### Retrieved Context"]

    # 1. Loop through documents + metadata
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):

        # Mission
        mission = meta.get("mission", "Unknown Mission")
        mission = mission.replace("_", " ").title()

        # Source
        source = meta.get("source", "Unknown Source")

        # Category
        category = meta.get("category", "Unknown Category")
        category = category.replace("_", " ").title()

        # Header
        header = f"\n---\nSource {idx}: {mission} | {category} | {source}"
        context_parts.append(header)

        # Truncate long documents
        if len(doc) > 1500:
            doc = doc[:1500] + "... [truncated]"

        context_parts.append(doc)

    return "\n".join(context_parts)
