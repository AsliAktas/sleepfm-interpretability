"""
pytest configuration — workspace root discovery.

Adding this file ensures pytest treats the sleepfm_interpretability directory
as the root, so `from mock_data import ...` and `from config import ...`
resolve correctly regardless of where pytest is invoked from.
"""
import sys
import os

# Guarantee that the workspace root is on sys.path before any test is collected.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
