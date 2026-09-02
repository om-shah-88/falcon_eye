#!/usr/bin/env python3
from __future__ import annotations

"""
Synthetic Camera Capture Data Generator
========================================
Generates a realistic dataset simulating a city-wide camera network capturing
vehicle traffic. Designed for ingestion into Document/Key-Value stores and
subsequent Vector Database embedding.

Simulation Parameters:
  - 10 camera nodes on a directed city-road graph
  - 2,000 distinct vehicles with persistent ground-truth profiles
  - 15,000 total capture events over a 2-hour window
  - 15% license plate noise (null or partial reads)
  - 5% color / unique-feature misclassification
"""

import json
import random
import string
import uuid
from datetime import datetime, timedelta, timezone

import networkx as nx

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
NUM_VEHICLES = 2_000
TOTAL_CAPTURES = 15_000
SIM_START = datetime(2026, 9, 2, 8, 0, 0, tzinfo=timezone.utc)
SIM_DURATION = timedelta(hours=2)

PLATE_NOISE_RATE = 0.15          # 15 % of captures
COLOR_FEATURE_NOISE_RATE = 0.05  # 5 % of captures

COLORS = ["Black", "White", "Silver", "Red", "Blue", "Gray",
          "Green", "Navy", "Maroon", "Beige"]
BODY_TYPES = ["SUV", "Sedan", "Truck"]
UNIQUE_MARKERS = ["roof_rack", "bumper_sticker", "dent", "tinted_windows",
                  "custom_rims", "bike_rack", "trailer_hitch", "flag_decal",
                  "racing_stripe", "mud_flaps"]

# Alternate colors used when injecting color-misclassification noise
COLOR_CONFUSION_MAP = {
    "Black":  ["Dark Gray", "Navy"],
    "White":  ["Silver", "Beige"],
    "Silver": ["White", "Light Gray"],
    "Red":    ["Maroon", "Orange"],
    "Blue":   ["Navy", "Teal"],
    "Gray":   ["Silver", "Charcoal"],
    "Green":  ["Teal", "Olive"],
    "Navy":   ["Blue", "Black"],
    "Maroon": ["Red", "Brown"],
    "Beige":  ["White", "Tan"],
}

SEED = 42
random.seed(SEED)


# ──────────────────────────────────────────────
# 1. Build the City Camera Network (Directed Graph)
# ──────────────────────────────────────────────
def build_city_graph() -> nx.DiGraph:
    """
    Create a directed graph of 10 camera nodes with realistic road
    connectivity.  The topology loosely models a grid-like downtown
    with arterial feeders:

        cam_01 ─► cam_02 ─► cam_03
          │         │╲        │
          ▼         ▼  ╲      ▼
        cam_04 ─► cam_05 ─► cam_06
          │         │         │
          ▼         ▼         ▼
        cam_07 ─► cam_08 ─► cam_09
                    │
                    ▼
                  cam_10
    Plus several reverse / diagonal edges for realism.
    """
    G = nx.DiGraph()
    nodes = [f"cam_{i:02d}" for i in range(1, 11)]
    G.add_nodes_from(nodes)

    edges = [
        # Row 1 ─► Row 2
        ("cam_01", "cam_02"), ("cam_02", "cam_03"),
        ("cam_01", "cam_04"), ("cam_02", "cam_05"), ("cam_03", "cam_06"),
        # Row 2 ─► Row 3
        ("cam_04", "cam_05"), ("cam_05", "cam_06"),
        ("cam_04", "cam_07"), ("cam_05", "cam_08"), ("cam_06", "cam_09"),
        # Row 3 connections
        ("cam_07", "cam_08"), ("cam_08", "cam_09"),
        ("cam_08", "cam_10"),
        # Diagonal / shortcut roads
        ("cam_02", "cam_06"), ("cam_05", "cam_09"),
        # Reverse-direction roads (two-way streets)
        ("cam_05", "cam_02"), ("cam_08", "cam_05"),
        ("cam_06", "cam_03"), ("cam_09", "cam_06"),
        ("cam_10", "cam_08"),
        # Arterial feeders back to entry points
        ("cam_07", "cam_04"), ("cam_09", "cam_10"),
        ("cam_03", "cam_02"),
    ]
    G.add_edges_from(edges)
    return G


# ──────────────────────────────────────────────
# 2. Generate Vehicle Pool
# ──────────────────────────────────────────────
def _random_plate() -> str:
    """Generate a realistic North-American-style license plate (e.g. ABC-1234)."""
    letters = "".join(random.choices(string.ascii_uppercase, k=3))
    digits = "".join(random.choices(string.digits, k=4))
    return f"{letters}-{digits}"


def generate_vehicles(n: int) -> list[dict]:
    """Return *n* unique vehicle profiles with persistent ground-truth data."""
    plates_seen: set[str] = set()
    vehicles: list[dict] = []

    for i in range(1, n + 1):
        # Ensure plate uniqueness
        plate = _random_plate()
        while plate in plates_seen:
            plate = _random_plate()
        plates_seen.add(plate)

        # 40 % chance of having 1-2 unique markers
        markers: list[str] = []
        if random.random() < 0.40:
            markers = random.sample(UNIQUE_MARKERS, k=random.randint(1, 2))

        vehicles.append({
            "vehicle_id": f"v_{i:05d}",
            "license_plate": plate,
            "color": random.choice(COLORS),
            "body_type": random.choice(BODY_TYPES),
            "unique_markers": markers,
        })

    return vehicles


# ──────────────────────────────────────────────
# 3. Simulate Vehicle Traversals
# ──────────────────────────────────────────────
def _random_path(G: nx.DiGraph, min_hops: int = 2, max_hops: int = 6) -> list[str]:
    """
    Walk a random path through the directed graph.
    Starts at a random node and follows outgoing edges up to *max_hops*.
    """
    nodes = list(G.nodes)
    current = random.choice(nodes)
    path = [current]

    for _ in range(random.randint(min_hops, max_hops)):
        neighbors = list(G.successors(current))
        if not neighbors:
            break
        current = random.choice(neighbors)
        path.append(current)

    return path


def generate_trips(G: nx.DiGraph, vehicles: list[dict],
                   total_captures: int) -> list[tuple]:
    """
    Generate vehicle trips until we accumulate exactly *total_captures*
    individual camera sightings.

    Returns a list of (vehicle_dict, path_list) tuples.
    """
    trips: list[tuple] = []
    captures_so_far = 0

    while captures_so_far < total_captures:
        vehicle = random.choice(vehicles)
        path = _random_path(G)

        remaining = total_captures - captures_so_far
        if len(path) > remaining:
            # Trim the path so we land on exactly the target
            path = path[:remaining]

        trips.append((vehicle, path))
        captures_so_far += len(path)

    return trips


# ──────────────────────────────────────────────
# 4. Apply "Vision Model" Noise
# ──────────────────────────────────────────────
def _noisy_plate(plate: str) -> tuple[str | None, float]:
    """
    Corrupt a license plate to simulate OCR failures.
    Returns (possibly_corrupted_plate, confidence_score).
    """
    roll = random.random()

    if roll < 0.40:
        # Total read failure → null plate
        return None, round(random.uniform(0.05, 0.25), 2)

    # Partial read: drop 1-3 random characters
    chars = list(plate)
    n_drop = random.randint(1, min(3, len(chars)))
    for idx in random.sample(range(len(chars)), n_drop):
        chars[idx] = "_"
    return "".join(chars), round(random.uniform(0.30, 0.65), 2)


def _noisy_color(true_color: str) -> str:
    """Swap the color for a plausible misclassification."""
    alternatives = COLOR_CONFUSION_MAP.get(true_color, ["Unknown"])
    return random.choice(alternatives)


def _noisy_markers(true_markers: list[str]) -> list[str]:
    """Either drop a real marker or hallucinate a spurious one."""
    markers = list(true_markers)

    if markers and random.random() < 0.5:
        # Drop one real marker
        markers.pop(random.randrange(len(markers)))
    else:
        # Hallucinate a marker the vehicle doesn't actually have
        extras = [m for m in UNIQUE_MARKERS if m not in markers]
        if extras:
            markers.append(random.choice(extras))

    return markers


# ──────────────────────────────────────────────
# 5. Build Capture Events
# ──────────────────────────────────────────────
def build_capture_events(trips: list[tuple]) -> list[dict]:
    """
    Convert each node visit into a capture-event document.
    Timestamps are distributed uniformly across the 2-hour window and
    then sorted chronologically.
    """
    raw_events: list[dict] = []

    for vehicle, path in trips:
        # Pick a random trip-start time within the simulation window
        start_offset = random.uniform(0, SIM_DURATION.total_seconds() - 120)
        trip_start = SIM_START + timedelta(seconds=start_offset)

        for hop_idx, node in enumerate(path):
            # Each hop adds 20-90 s of travel time
            ts = trip_start + timedelta(seconds=hop_idx * random.uniform(20, 90))

            # ── Inferred features (with possible noise) ──
            plate = vehicle["license_plate"]
            confidence = round(random.uniform(0.82, 0.99), 2)

            if random.random() < PLATE_NOISE_RATE:
                plate, confidence = _noisy_plate(vehicle["license_plate"])

            color = vehicle["color"]
            markers = list(vehicle["unique_markers"])

            if random.random() < COLOR_FEATURE_NOISE_RATE:
                # Flip a coin: corrupt color or markers (or both)
                if random.random() < 0.5:
                    color = _noisy_color(vehicle["color"])
                else:
                    markers = _noisy_markers(vehicle["unique_markers"])

            raw_events.append({
                "capture_id": str(uuid.uuid4()),
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "camera_node_id": node,
                "inferred_features": {
                    "license_plate": plate,
                    "plate_confidence": confidence,
                    "color": color,
                    "body_type": vehicle["body_type"],
                    "unique_markers": markers,
                },
                "ground_truth": {
                    "vehicle_id": vehicle["vehicle_id"],
                    "actual_path": path,
                },
            })

    # Sort all events chronologically
    raw_events.sort(key=lambda e: e["timestamp"])
    return raw_events


# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────
def main() -> None:
    print("🔧  Building city camera network …")
    G = build_city_graph()
    print(f"    Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")

    print(f"🚗  Generating {NUM_VEHICLES:,} vehicle profiles …")
    vehicles = generate_vehicles(NUM_VEHICLES)

    print(f"🛣️   Simulating trips to reach {TOTAL_CAPTURES:,} capture events …")
    trips = generate_trips(G, vehicles, TOTAL_CAPTURES)
    print(f"    Total trips generated: {len(trips):,}")

    print("📸  Building capture event documents (with CNN noise) …")
    events = build_capture_events(trips)

    # ── Quick sanity stats ──
    null_plates = sum(1 for e in events
                      if e["inferred_features"]["license_plate"] is None)
    partial_plates = sum(1 for e in events
                         if e["inferred_features"]["license_plate"] is not None
                         and "_" in e["inferred_features"]["license_plate"])
    noisy_plates = null_plates + partial_plates

    print(f"\n✅  Dataset Summary")
    print(f"    Total capture events : {len(events):,}")
    print(f"    Unique vehicles      : {len(vehicles):,}")
    print(f"    Noisy plates         : {noisy_plates:,}  "
          f"({noisy_plates / len(events) * 100:.1f}%)")
    print(f"    Simulation window    : {events[0]['timestamp']}  →  "
          f"{events[-1]['timestamp']}")

    # ── Split into camera data (what the system sees) and ground truth (answer key) ──
    camera_events: list[dict] = []
    ground_truth_lookup: dict[str, dict] = {}

    for event in events:
        capture_id = event["capture_id"]

        # Camera data — everything EXCEPT ground_truth
        camera_events.append({
            "capture_id": capture_id,
            "timestamp": event["timestamp"],
            "camera_node_id": event["camera_node_id"],
            "inferred_features": event["inferred_features"],
        })

        # Ground truth — keyed by capture_id for easy lookup during evaluation
        ground_truth_lookup[capture_id] = {
            "capture_id": capture_id,
            "timestamp": event["timestamp"],
            "camera_node_id": event["camera_node_id"],
            "vehicle_id": event["ground_truth"]["vehicle_id"],
            "actual_path": event["ground_truth"]["actual_path"],
        }

    # ── Write camera capture data (for ingestion / embedding) ──
    capture_path = "synthetic_camera_data.json"
    print(f"\n💾  Writing camera captures to {capture_path} …")
    with open(capture_path, "w", encoding="utf-8") as f:
        json.dump(camera_events, f, indent=2, ensure_ascii=False)
    print(f"    {len(camera_events):,} capture documents written.")

    # ── Write ground truth (answer key for evaluation) ──
    gt_path = "ground_truth.json"
    ground_truth_output = {
        "metadata": {
            "total_captures": len(ground_truth_lookup),
            "total_vehicles": len(vehicles),
            "simulation_start": SIM_START.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "simulation_end": (SIM_START + SIM_DURATION).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "noise_config": {
                "plate_noise_rate": PLATE_NOISE_RATE,
                "color_feature_noise_rate": COLOR_FEATURE_NOISE_RATE,
            },
        },
        "vehicle_profiles": {v["vehicle_id"]: v for v in vehicles},
        "capture_ground_truth": ground_truth_lookup,
    }
    print(f"💾  Writing ground truth to {gt_path} …")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_output, f, indent=2, ensure_ascii=False)
    print(f"    {len(ground_truth_lookup):,} ground-truth entries + "
          f"{len(vehicles):,} vehicle profiles written.")

    print("\n🎯  Files ready:")
    print(f"    📷  {capture_path}  — feed this into your vector DB")
    print(f"    🔑  {gt_path}       — use this to evaluate matches")


if __name__ == "__main__":
    main()
