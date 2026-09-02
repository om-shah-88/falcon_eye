"""
FalconEye — Ingest captures into ChromaDB for vector-based vehicle re-identification.

This script:
  1. Reads synthetic_camera_data.json  (what the cameras "see")
  2. Converts each capture into a meaningful text description
  3. Embeds the text using all-MiniLM-L6-v2
  4. Stores in ChromaDB with structured metadata for hybrid search
  5. Demonstrates querying + cross-referencing with ground_truth.json

KEY CONCEPT — Linking the two files:
  ┌─────────────────────────────┐      ┌────────────────────────────┐
  │  synthetic_camera_data.json │      │     ground_truth.json      │
  │                             │      │                            │
  │  capture_id  ◄──── FOREIGN KEY ───►  capture_ground_truth       │
  │  timestamp                  │      │    [capture_id] → {        │
  │  camera_node_id             │      │      vehicle_id,           │
  │  inferred_features {}       │      │      actual_path           │
  │                             │      │    }                       │
  └─────────────────────────────┘      │                            │
                                       │  vehicle_profiles          │
                                       │    [vehicle_id] → {        │
                                       │      license_plate,        │
                                       │      color, body_type,     │
                                       │      unique_markers        │
                                       │    }                       │
                                       └────────────────────────────┘

  capture_id is the KEY that links everything.
  After a vector search returns capture_ids, you look them up in ground_truth.json.
"""

import json
import time
import chromadb
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────
# 1. CONVERT CAPTURE → MEANINGFUL TEXT STRING
# ──────────────────────────────────────────────
def capture_to_text(capture: dict) -> str:
    """
    Convert a capture event's inferred features into a natural-language
    description that the embedding model can understand.

    WHY TEXT? Because embedding models like MiniLM are trained on natural
    language. A structured dict like {"color": "Silver", "body_type": "Sedan"}
    doesn't carry semantic meaning for the model. But the sentence
    "Silver Sedan spotted at cam_03" does — the model understands that
    "Silver" is a color and "Sedan" is a vehicle type.

    The text format is designed so that:
    - Similar vehicles produce similar embeddings (close in vector space)
    - Noisy plates still partially match (e.g., "plate: AB_-1234" vs "plate: ABC-1234")
    - Missing plates still match on other features (color, body type, markers)
    """
    features = capture["inferred_features"]

    # Build description parts
    parts = []

    # Color + Body type (most important visual features)
    color = features.get("color", "Unknown")
    body = features.get("body_type", "Unknown")
    parts.append(f"{color} {body}")

    # License plate (include confidence so the model can learn plate reliability)
    plate = features.get("license_plate")
    confidence = features.get("plate_confidence", 0)
    if plate:
        parts.append(f"plate {plate} confidence {confidence}")
    else:
        parts.append("plate unreadable")

    # Unique markers (visual distinguishing features)
    markers = features.get("unique_markers", [])
    if markers:
        parts.append(f"markers: {', '.join(markers)}")

    # Camera location (spatial context)
    parts.append(f"camera {capture['camera_node_id']}")

    # Combine into a single descriptive sentence
    return ". ".join(parts)


# ──────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────
def load_data():
    """Load camera captures and ground truth."""
    print("📂  Loading synthetic_camera_data.json …")
    with open("synthetic_camera_data.json", "r") as f:
        captures = json.load(f)
    print(f"    {len(captures):,} captures loaded")

    print("📂  Loading ground_truth.json …")
    with open("ground_truth.json", "r") as f:
        ground_truth = json.load(f)
    print(f"    {len(ground_truth['capture_ground_truth']):,} ground-truth entries")
    print(f"    {len(ground_truth['vehicle_profiles']):,} vehicle profiles")

    return captures, ground_truth


# ──────────────────────────────────────────────
# 3. INGEST INTO CHROMADB
# ──────────────────────────────────────────────
def ingest_to_chromadb(captures: list[dict]) -> chromadb.Collection:
    """
    Embed all captures and store in ChromaDB.

    ChromaDB stores three things per document:
      1. The TEXT (the meaningful description we created)
      2. The EMBEDDING (384-dim vector from MiniLM)
      3. METADATA (structured fields for filtering — color, body_type, camera, etc.)

    Metadata lets you do HYBRID SEARCH:
      - Vector similarity finds "looks like a Silver Sedan"
      - Metadata filter narrows to "only from cam_03"
    """

    # ── Initialize embedding model ──
    print("\n🧠  Loading embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("    Model loaded (384-dim embeddings)")

    # ── Initialize ChromaDB (persistent, saved to disk) ──
    print("\n🗄️   Initializing ChromaDB …")
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if re-running
    try:
        client.delete_collection("camera_captures")
    except Exception:
        pass

    collection = client.create_collection(
        name="camera_captures",
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )
    print("    Collection 'camera_captures' created (cosine similarity)")

    # ── Convert all captures to text ──
    print("\n📝  Converting captures to text descriptions …")
    texts = [capture_to_text(c) for c in captures]

    # Show a few examples
    print("    Examples:")
    for i in range(min(5, len(texts))):
        print(f"      [{i}] {texts[i]}")

    # ── Batch embed and insert ──
    # ChromaDB has a batch limit, so we process in chunks of 500
    BATCH_SIZE = 500
    total_batches = (len(captures) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n⚡  Embedding + inserting {len(captures):,} documents "
          f"in {total_batches} batches …")

    start_time = time.time()

    for batch_idx in range(0, len(captures), BATCH_SIZE):
        batch_end = min(batch_idx + BATCH_SIZE, len(captures))
        batch_captures = captures[batch_idx:batch_end]
        batch_texts = texts[batch_idx:batch_end]

        # Generate embeddings for this batch
        batch_embeddings = model.encode(batch_texts).tolist()

        # Prepare IDs and metadata
        batch_ids = [c["capture_id"] for c in batch_captures]
        batch_metadatas = []
        for c in batch_captures:
            feat = c["inferred_features"]
            batch_metadatas.append({
                "timestamp":      c["timestamp"],
                "camera_node_id": c["camera_node_id"],
                "color":          feat.get("color", "Unknown"),
                "body_type":      feat.get("body_type", "Unknown"),
                "license_plate":  feat.get("license_plate") or "UNREADABLE",
                "plate_confidence": feat.get("plate_confidence", 0.0),
                "has_markers":    len(feat.get("unique_markers", [])) > 0,
                "markers":        ", ".join(feat.get("unique_markers", [])),
            })

        # Insert into ChromaDB
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

        batch_num = batch_idx // BATCH_SIZE + 1
        print(f"    Batch {batch_num}/{total_batches} — "
              f"{batch_end:,}/{len(captures):,} documents inserted")

    elapsed = time.time() - start_time
    print(f"\n✅  Ingestion complete in {elapsed:.1f}s")
    print(f"    Collection size: {collection.count():,} documents")

    return collection


# ──────────────────────────────────────────────
# 4. QUERY + CROSS-REFERENCE WITH GROUND TRUTH
# ──────────────────────────────────────────────
def demo_query(collection: chromadb.Collection, ground_truth: dict):
    """
    Demonstrate:
      1. Vector similarity search ("find vehicles that look like …")
      2. Hybrid search (vector + metadata filter)
      3. Cross-referencing results with ground_truth.json using capture_id
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gt_captures = ground_truth["capture_ground_truth"]
    gt_vehicles = ground_truth["vehicle_profiles"]

    print("\n" + "═" * 60)
    print("  DEMO: Vector Similarity Search + Ground Truth Lookup")
    print("═" * 60)

    # ── Query 1: Pure vector search ──
    query_text = "Red SUV with roof_rack"
    print(f"\n🔍  Query: \"{query_text}\"")
    print("    Mode: Pure vector similarity (no filters)")

    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
    )

    print(f"\n    Top 5 matches:")
    print(f"    {'Rank':<5} {'Score':<8} {'capture_id':<20} {'Description':<50} {'Vehicle (GT)':<12} {'Correct?'}")
    print(f"    {'─'*5} {'─'*8} {'─'*20} {'─'*50} {'─'*12} {'─'*8}")

    for rank, (cid, doc, dist) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ), 1):
        # ── THIS IS HOW YOU LINK TO GROUND TRUTH ──
        # capture_id from search result → look up in ground_truth.json
        gt_entry = gt_captures.get(cid, {})
        vehicle_id = gt_entry.get("vehicle_id", "?")
        vehicle_profile = gt_vehicles.get(vehicle_id, {})
        actual_color = vehicle_profile.get("color", "?")
        actual_body = vehicle_profile.get("body_type", "?")
        actual_markers = vehicle_profile.get("unique_markers", [])

        # Check if the match is actually correct
        is_correct = (actual_color == "Red" and actual_body == "SUV"
                      and "roof_rack" in actual_markers)

        print(f"    {rank:<5} {1 - dist:<8.4f} {cid[:18]+'…':<20} "
              f"{doc[:48]+'…':<50} {vehicle_id:<12} "
              f"{'✅' if is_correct else '❌'}")

    # ── Query 2: Hybrid search (vector + metadata filter) ──
    query_text = "White Sedan"
    print(f"\n🔍  Query: \"{query_text}\"")
    print("    Mode: Hybrid — vector similarity + filter camera=cam_03")

    query_embedding = model.encode([query_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        where={"camera_node_id": "cam_03"},  # Only from this camera
    )

    print(f"\n    Top 5 matches (cam_03 only):")
    print(f"    {'Rank':<5} {'Score':<8} {'Camera':<10} {'Description':<55} {'Vehicle (GT)'}")
    print(f"    {'─'*5} {'─'*8} {'─'*10} {'─'*55} {'─'*12}")

    for rank, (cid, doc, dist, meta) in enumerate(zip(
        results["ids"][0], results["documents"][0],
        results["distances"][0], results["metadatas"][0]
    ), 1):
        gt_entry = gt_captures.get(cid, {})
        vehicle_id = gt_entry.get("vehicle_id", "?")
        print(f"    {rank:<5} {1 - dist:<8.4f} {meta['camera_node_id']:<10} "
              f"{doc[:53]+'…':<55} {vehicle_id}")

    # ── Query 3: Find all sightings of a specific vehicle ──
    # Pick a vehicle from ground truth and find all its captures
    sample_vid = list(gt_vehicles.keys())[42]  # Pick vehicle #42
    sample_vehicle = gt_vehicles[sample_vid]
    search_text = (f"{sample_vehicle['color']} {sample_vehicle['body_type']} "
                   f"plate {sample_vehicle['license_plate']}")

    print(f"\n🔍  Query: \"{search_text}\"")
    print(f"    Mode: Track vehicle {sample_vid} across cameras")

    query_embedding = model.encode([search_text]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10,
    )

    print(f"\n    Sightings found (checking ground truth):")
    correct = 0
    for rank, (cid, doc, dist) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["distances"][0]
    ), 1):
        gt_entry = gt_captures.get(cid, {})
        actual_vid = gt_entry.get("vehicle_id", "?")
        match = "✅" if actual_vid == sample_vid else "❌"
        if actual_vid == sample_vid:
            correct += 1
        camera = gt_entry.get("camera_node_id", "?")
        ts = gt_entry.get("timestamp", "?")
        print(f"    {rank:<3} {match} score={1 - dist:.4f}  "
              f"vehicle={actual_vid}  camera={camera}  time={ts}")

    print(f"\n    Precision@10: {correct}/10 = {correct/10*100:.0f}%")


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────
def main():
    captures, ground_truth = load_data()
    collection = ingest_to_chromadb(captures)
    demo_query(collection, ground_truth)

    print("\n\n" + "═" * 60)
    print("  💡  HOW TO USE IN YOUR OWN CODE")
    print("═" * 60)
    print("""
    # Load existing ChromaDB (no re-ingestion needed)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("camera_captures")

    # Search
    results = collection.query(
        query_texts=["Blue Truck with bumper_sticker"],
        n_results=10,
    )

    # Cross-reference with ground truth
    for capture_id in results["ids"][0]:
        gt = ground_truth["capture_ground_truth"][capture_id]
        print(f"Vehicle: {gt['vehicle_id']}, Path: {gt['actual_path']}")
    """)


if __name__ == "__main__":
    main()
