#!/usr/bin/env python
"""Entry point for Agentic Harness Streamlit Dashboard."""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from harness.dashboard import main

if __name__ == "__main__":
    main()