#!/usr/bin/env python3
"""
FalconEye End-to-End Test Suite
================================
Tests the full pipeline: server → API → ChromaDB → ground truth linkage
Run with: .venv/bin/python3 test_e2e.py
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅  {name}")
    else:
        FAIL += 1
        print(f"  ❌  {name}")
        if detail:
            print(f"      → {detail}")


def get(path):
    """GET request, returns (status_code, body)."""
    try:
        req = urllib.request.Request(f"{BASE}{path}")
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)


def post(path, data):
    """POST JSON, returns (status_code, parsed_json)."""
    try:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{BASE}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)


def main():
    print("=" * 60)
    print("  FalconEye End-to-End Test Suite")
    print("=" * 60)

    # ── 1. Server Health ──
    print("\n📡  Server Connectivity")
    status, body = get("/")
    test("Server is reachable", status == 200, f"Got status {status}")
    test("Frontend HTML loads", "FalconEye" in body, "Missing FalconEye in page")
    test("HTML has search form", 'id="search-form"' in body)
    test("HTML has results section", 'id="results-section"' in body)

    # ── 2. API: Basic search ──
    print("\n🔍  Search API — Basic Query")
    status, data = post("/api/search", {
        "color": "Red",
        "body_type": "SUV",
        "markers": "roof_rack",
        "num_results": 10
    })
    test("POST /api/search returns 200", status == 200, f"Got status {status}")
    test("Response has query_text", "query_text" in data)
    test("Query text is correct", data.get("query_text") == "Red. SUV. markers: roof_rack",
         f"Got: {data.get('query_text')}")
    test("Response has matches", "matches" in data and len(data["matches"]) > 0,
         f"Got {len(data.get('matches', []))} matches")
    test("total_captures_searched is 15000", data.get("total_captures_searched") == 15000)

    # ── 3. Match quality ──
    print("\n🎯  Match Quality — Red SUV with roof_rack")
    if data.get("matches"):
        top = data["matches"][0]
        test("Top match has rank 1", top["rank"] == 1)
        test("Top match has probability > 0", top["probability"] > 0)
        test("Top match has vehicle_id", top["vehicle_id"].startswith("v_"))
        test("Top match has camera trail", len(top["camera_trail"]) > 0)
        test("Top match has last_seen_camera", top["last_seen_camera"].startswith("cam_"))
        test("Top match has last_seen_time", "2026-09-02" in top["last_seen_time"])
        test("Top match has sightings", len(top["all_sightings"]) > 0)
        test("Top match true_color is Red", top["true_color"] == "Red",
             f"Got: {top['true_color']}")
        test("Top match true_body_type is SUV", top["true_body_type"] == "SUV",
             f"Got: {top['true_body_type']}")
        test("Probability sums ≤ 1.0",
             sum(m["probability"] for m in data["matches"]) <= 1.01)

    # ── 4. Sighting details ──
    print("\n📸  Sighting Details")
    if data.get("matches") and data["matches"][0].get("all_sightings"):
        s = data["matches"][0]["all_sightings"][0]
        test("Sighting has capture_id", "capture_id" in s)
        test("Sighting has camera", "camera" in s and s["camera"].startswith("cam_"))
        test("Sighting has timestamp", "timestamp" in s)
        test("Sighting has similarity score", "similarity" in s and 0 < s["similarity"] <= 1)
        test("Sighting has description text", "description" in s and len(s["description"]) > 10)

    # ── 5. Ground truth linkage ──
    print("\n🔗  Ground Truth Linkage")
    with open("ground_truth.json") as f:
        gt = json.load(f)

    if data.get("matches"):
        top = data["matches"][0]
        vid = top["vehicle_id"]
        test("Vehicle ID exists in ground truth profiles",
             vid in gt["vehicle_profiles"])
        if vid in gt["vehicle_profiles"]:
            profile = gt["vehicle_profiles"][vid]
            test("Ground truth plate matches response",
                 profile["license_plate"] == top["true_plate"],
                 f"GT: {profile['license_plate']}, Resp: {top['true_plate']}")
            test("Ground truth color matches response",
                 profile["color"] == top["true_color"])

        # Check capture_id linkage
        if top["all_sightings"]:
            cid = top["all_sightings"][0]["capture_id"]
            test("Capture ID exists in ground truth",
                 cid in gt["capture_ground_truth"])
            if cid in gt["capture_ground_truth"]:
                gt_entry = gt["capture_ground_truth"][cid]
                test("Ground truth maps capture to correct vehicle",
                     gt_entry["vehicle_id"] == vid)

    # ── 6. Edge cases ──
    print("\n⚡  Edge Cases")

    # Empty query
    status, data = post("/api/search", {"num_results": 5})
    test("Empty query doesn't crash", status == 200)
    test("Empty query returns matches", len(data.get("matches", [])) > 0)

    # Plate-only query
    status, data = post("/api/search", {"license_plate": "FTS-3516", "num_results": 5})
    test("Plate-only search works", status == 200)
    test("Plate search returns matches", len(data.get("matches", [])) > 0)

    # Free text query
    status, data = post("/api/search", {
        "free_text": "dark colored truck with dents",
        "num_results": 5
    })
    test("Free text search works", status == 200)

    # Large num_results
    status, data = post("/api/search", {
        "color": "Black",
        "num_results": 100
    })
    test("Large result set works", status == 200)
    test("All probabilities sum to ~1.0",
         abs(sum(m["probability"] for m in data.get("matches", [])) - 1.0) < 0.01,
         f"Sum: {sum(m['probability'] for m in data.get('matches', [])):.4f}")

    # ── 7. Frontend fetch URL ──
    print("\n🌐  Frontend Configuration")
    with open("static/index.html") as f:
        html = f.read()
    test("HTML uses dynamic API_BASE", "API_BASE" in html)
    test("HTML has file:// fallback to localhost", "http://localhost:8000" in html)
    test("HTML doesn't hardcode relative-only /api/search",
         "fetch('/api/search'" not in html)

    # ── Summary ──
    total = PASS + FAIL
    print("\n" + "=" * 60)
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("  🎉  All tests passed!")
    else:
        print("  ⚠️  Some tests failed — check output above")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
