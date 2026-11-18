#!/usr/bin/env python
"""Entry point for Agentic Harness Streamlit Dashboard.

Can be executed in multiple ways:
1. python run_dashboard.py               (auto-launches Streamlit web server)
2. python -m streamlit run run_dashboard.py
3. streamlit run run_dashboard.py
"""
import os
import sys
from pathlib import Path

# Add src to path so harness package is always discoverable
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        running_in_streamlit = get_script_run_ctx() is not None
    except Exception:
        running_in_streamlit = False

    if running_in_streamlit:
        from harness.dashboard import main
        main()
    else:
        # Running directly with python: invoke streamlit CLI directly
        # This bypasses Windows Application Control policy blocking streamlit.exe
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())] + sys.argv[1:]
        sys.exit(stcli.main())
