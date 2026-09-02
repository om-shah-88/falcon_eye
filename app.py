"""
FalconEye — Web API for vehicle search.

FastAPI backend that:
  1. Serves the static frontend (index.html)
  2. Accepts vehicle description searches via /api/search
  3. Queries ChromaDB for similar captures
  4. Cross-references with ground truth for evaluation
  5. Returns ranked results with probabilities and camera trails
"""

import json
import os
from pathlib import Path

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# Startup: Load models and data
# ──────────────────────────────────────────────
print("🚀  Starting FalconEye API …")

# Load embedding model
print("🧠  Loading embedding model …")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load ChromaDB (already ingested)
print("🗄️   Connecting to ChromaDB …")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("camera_captures")
print(f"    Collection loaded: {collection.count():,} documents")

# Load ground truth for cross-referencing
print("📂  Loading ground truth …")
with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)
gt_captures = ground_truth["capture_ground_truth"]
gt_vehicles = ground_truth["vehicle_profiles"]
print(f"    {len(gt_captures):,} capture entries, {len(gt_vehicles):,} vehicle profiles")

print("✅  FalconEye API ready!\n")

# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(title="FalconEye", description="Vehicle Re-Identification Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────
class SearchRequest(BaseModel):
    color: str = ""
    body_type: str = ""
    license_plate: str = ""
    markers: str = ""           # comma-separated
    free_text: str = ""         # optional free-form description
    num_results: int = 30


class VehicleMatch(BaseModel):
    rank: int
    vehicle_id: str
    probability: float          # match confidence
    match_count: int            # how many captures matched
    last_seen_camera: str
    last_seen_time: str
    camera_trail: list[str]     # ordered list of cameras
    true_color: str
    true_body_type: str
    true_plate: str
    true_markers: list[str]
    all_sightings: list[dict]   # detailed sightings


class SearchResponse(BaseModel):
    query_text: str
    total_captures_searched: int
    matches: list[VehicleMatch]


# ──────────────────────────────────────────────
# Search Endpoint
# ──────────────────────────────────────────────
@app.post("/api/search", response_model=SearchResponse)
async def search_vehicle(req: SearchRequest):
    """
    Convert form input → text description → embedding → ChromaDB query
    → group by vehicle → return ranked results with probabilities.
    """
    # ── Build query text from form fields ──
    parts = []
    if req.color:
        parts.append(req.color)
    if req.body_type:
        parts.append(req.body_type)
    if req.license_plate:
        parts.append(f"plate {req.license_plate}")
    if req.markers:
        parts.append(f"markers: {req.markers}")
    if req.free_text:
        parts.append(req.free_text)

    query_text = ". ".join(parts) if parts else "vehicle"

    # ── Embed and query ChromaDB ──
    query_embedding = embed_model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(req.num_results, 100),
    )

    # ── Group results by vehicle (via ground truth) ──
    vehicle_groups: dict[str, list[dict]] = {}
    total_score = 0.0

    for cid, doc, distance, meta in zip(
        results["ids"][0],
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        similarity = round(1 - distance, 4)  # cosine distance → similarity
        total_score += similarity

        gt_entry = gt_captures.get(cid, {})
        vid = gt_entry.get("vehicle_id", "unknown")

        sighting = {
            "capture_id": cid,
            "camera": gt_entry.get("camera_node_id", meta.get("camera_node_id", "?")),
            "timestamp": gt_entry.get("timestamp", meta.get("timestamp", "?")),
            "similarity": similarity,
            "description": doc,
        }

        if vid not in vehicle_groups:
            vehicle_groups[vid] = []
        vehicle_groups[vid].append(sighting)

    # ── Rank vehicles by aggregate match strength ──
    ranked: list[VehicleMatch] = []

    for vid, sightings in vehicle_groups.items():
        # Sort sightings by timestamp (most recent last)
        sightings.sort(key=lambda s: s["timestamp"])

        # Aggregate probability: sum of similarities for this vehicle / total
        group_score = sum(s["similarity"] for s in sightings)
        probability = round(group_score / total_score, 4) if total_score > 0 else 0

        # Camera trail (ordered by time)
        camera_trail = [s["camera"] for s in sightings]

        # Vehicle ground truth profile
        profile = gt_vehicles.get(vid, {})

        ranked.append(VehicleMatch(
            rank=0,  # will be set after sorting
            vehicle_id=vid,
            probability=probability,
            match_count=len(sightings),
            last_seen_camera=sightings[-1]["camera"],
            last_seen_time=sightings[-1]["timestamp"],
            camera_trail=camera_trail,
            true_color=profile.get("color", "?"),
            true_body_type=profile.get("body_type", "?"),
            true_plate=profile.get("license_plate", "?"),
            true_markers=profile.get("unique_markers", []),
            all_sightings=sightings,
        ))

    # Sort by probability (highest first)
    ranked.sort(key=lambda m: m.probability, reverse=True)
    for i, match in enumerate(ranked, 1):
        match.rank = i

    return SearchResponse(
        query_text=query_text,
        total_captures_searched=collection.count(),
        matches=ranked,
    )


# ──────────────────────────────────────────────
# Serve Frontend
# ──────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# Serve any static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
