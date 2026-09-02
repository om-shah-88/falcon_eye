# FalconEye: Project Documentation

## 1. Project Overview
**FalconEye** is a decentralized, highly scalable vehicle tracking and image-recognition software system inspired by "Flock Safety". The core objective of the project is to utilize a city-wide network of intelligent camera nodes to track vehicle movements, allowing users (such as law enforcement or security personnel) to search for specific vehicles based on partial descriptions, license plates, and visual markers.

Due to the logistical, hardware, and privacy constraints of deploying physical cameras and edge AI pipelines for an academic capstone, the project is structured into three phases. This document outlines the current state (Phase 1 prototype) and the roadmap for future phases (Phase 2 and Phase 3).

---

## 2. Phase 1: The Synthetic Prototype (Current State)

### 2.1 The Objective
The goal of Phase 1 is to validate the backend data architecture and search mechanics. By simulating the outputs of a city-wide camera network, we can build and test the vector similarity search without needing physical hardware.

### 2.2 Synthetic Data Generation
We developed a sophisticated Python-based simulator (`generate_synthetic_data.py`) to create a highly realistic dataset:
- **The Graph:** A directional graph (`networkx`) representing a city map with 10 camera nodes.
- **The Entities:** A ground-truth pool of 2,000 distinct vehicles, each with specific attributes (license plate, color, body type, unique markers like roof racks or dents).
- **The Volume:** Simulated random traversal resulting in 15,000 capture events over a 2-hour window.
- **Simulated Imperfections:** To mimic real edge-hardware limitations, the data includes deterministic noise: 15% of license plates are partially or fully obscured (null), and 5% of categorical visual features are misclassified.

### 2.3 Backend Search Architecture
The prototype uses a modern AI search stack:
- **Semantic Transformation:** Structured JSON outputs (what the camera "sees") are converted into natural language strings (e.g., *"Silver Sedan. plate XYZ-982 confidence 0.92. markers: roof_rack."*).
- **Embeddings:** These strings are vectorized using HuggingFace's `all-MiniLM-L6-v2` model into 384-dimensional embeddings.
- **Storage & Querying:** The vectors are stored in **ChromaDB**. We built a FastAPI backend (`app.py`) to handle similarity queries and serve a web UI.

### 2.4 Key Finding: The Limitation of Pure Vector Search
During Phase 1 testing, a critical discovery was made: **Semantic embedding models excel at visual features (Color, Body Type) but fail at exact string identifiers (License Plates).** 

Because models perceive alphanumeric strings as semantic tokens, a search for plate `RBY-0867` might return `RBX-1234` with a 95% similarity score. This proved that pure vector search is insufficient for identity-based tracking, directly informing the architecture for Phase 2.

---

## 3. Future Roadmap

### 3.1 Phase 2: Hybrid Search & Graph Constraints
Based on the findings from Phase 1, the next iteration of the software will implement a **Hybrid Search Pipeline**:
1. **Identifier Filtering:** Using exact or fuzzy string matching (via metadata filters or a relational DB) to handle license plates.
2. **Semantic Ranking:** Using the vector database to rank the remaining candidates based on visual similarity.
3. **Graph Traceability:** Integrating spatial-temporal logic to filter out impossible results (e.g., if a car was at Node 1 at 08:00, it cannot logically be at Node 10 at 08:01).

### 3.2 Phase 3: The Cascaded Edge AI Pipeline
The final phase involves replacing the synthetic data generator with real-world edge computing hardware (e.g., NVIDIA Jetson Nano or Google Coral TPU). Standard CCTV cameras lack the compute power for monolithic models, so we will implement a cascaded pipeline:
- **Object & Plate Localization:** A lightweight model like `YOLOv8n` to draw bounding boxes around the vehicle and license plate.
- **OCR (Optical Character Recognition):** Models like `LPRNet` or `PaddleOCR` to read skewed or blurry plate characters and generate a confidence score.
- **Vehicle Attribute Recognition:** An efficient classifier like `MobileNetV3` to determine color and body type.
- **Anomaly/Marker Detection:** Secondary fine-tuned models to spot unique identifiers like bumper stickers or roof racks.

---

## 4. Current Technology Stack
- **Backend Framework:** FastAPI (Python)
- **Vector Database:** ChromaDB
- **Embedding Model:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Data Simulation:** `networkx`
- **Frontend UI:** HTML, CSS (Glassmorphism design), JavaScript (Vanilla)
