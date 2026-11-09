
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

from qdrant_client import QdrantClient, models
from fastapi import FastAPI, HTTPException, APIRouter

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent

SCHEMA_PATH = BASE_DIR / "enriched_schema.json"

COLLECTION_NAME = "oracle_schema_tables"
MODEL_NAME = "all-MiniLM-L6-v2"
QDRANT_URL = "http://localhost:6333"


# ==========================================
# 1. LOAD MODELS AND CLIENT
# ==========================================

print("Loading Sentence Transformer model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")

client = QdrantClient(url=QDRANT_URL)


# ==========================================
# 2. CREATE TABLE CHUNK
# ==========================================

def create_table_chunk(table, database):

    owner = table.get("owner", "")
    table_name = table.get("table_name", "")

    description = (
        table.get("table_description")
        or "Not available"
    )

    lines = [
        f"Database: {database}",
        f"Schema: {owner}",
        f"Table: {table_name}",
        f"Table Description: {description}",
        "",
        "Columns:"
    ]

    for column in table.get("columns", []):

        lines.append(
            f"""
Column Name: {column.get('column_name')}
Data Type: {column.get('data_type')}
Data Length: {column.get('data_length')}
Data Precision: {column.get('data_precision')}
Data Scale: {column.get('data_scale')}
Nullable: {column.get('nullable')}
Column Description: {column.get('column_description') or 'Not available'}
Key Type: {column.get('key_type')}
Primary Key Constraint: {column.get('primary_key_constraint')}
Foreign Key: {column.get('foreign_key')}
Unique Key Constraint: {column.get('unique_key_constraint')}
""".strip()
        )

    return "\n".join(lines)


# ==========================================
# 3. INDEX SCHEMA FUNCTION
# ==========================================

def index_schema():

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    tables = schema_data.get("tables", [])

    if not tables:
        raise ValueError("No tables found in schema file.")

    chunks = []
    payloads = []

    for table in tables:

        chunk = create_table_chunk(
            table,
            schema_data.get("database", "Oracle")
        )

        chunks.append(chunk)

        payloads.append({
            "owner": table.get("owner"),
            "table_name": table.get("table_name"),
            "table_description": table.get("table_description"),
            "column_count": len(table.get("columns", [])),
            "content": chunk
        })

    print(f"Total tables: {len(tables)}")

    # Generate embeddings

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Create collection only if it does not exist

    if not client.collection_exists(COLLECTION_NAME):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

    # Index embeddings

    points = []

    for index, (embedding, payload) in enumerate(
        zip(embeddings, payloads)
    ):

        points.append(
            models.PointStruct(
                id=index + 1,
                vector=embedding.tolist(),
                payload=payload
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    return {
        "message": "Schema indexing completed successfully.",
        "collection": COLLECTION_NAME,
        "total_tables": len(tables),
        "total_indexed": len(points)
    }


# ==========================================
# 4. FASTAPI ENDPOINTS
# ==========================================

@router.get("/")
def home():

    return {
        "message": "Oracle Schema Retrieval API is running."
    }


@router.post("/index-schema")
def index_schema_endpoint():

    try:

        result = index_schema()

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/collection-info")
def collection_info():

    try:

        if not client.collection_exists(COLLECTION_NAME):

            return {
                "exists": False,
                "message": "Collection does not exist."
            }

        info = client.get_collection(COLLECTION_NAME)

        return {
            "exists": True,
            "collection": COLLECTION_NAME,
            "points_count": info.points_count,
            "status": str(info.status)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
