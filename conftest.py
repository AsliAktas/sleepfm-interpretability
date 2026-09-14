"""
pytest configuration — workspace root discovery.

Adds three directories to sys.path so imports resolve regardless of where
pytest is invoked from:
- workspace root (for cross-cutting imports)
- src/  (active modules: real_embeddings, rigor_analysis, ...)
- src/legacy/  (legacy mock evren: mock_data, similarity_engine, ...)

Legacy modules stayed as-is (from mock_data import ...) after being moved
into src/legacy/ — this sys.path setup keeps the old import syntax working
without needing to touch dozens of module-level imports.
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, os.path.join(_HERE, "src", "legacy"))
