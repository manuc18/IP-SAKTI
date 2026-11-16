import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `scripts/` for the Phase 2-3 pipeline packages (`ingest`, `evaluation`);
# the project root for the Phase 4 retrieval package (`ai`).
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
