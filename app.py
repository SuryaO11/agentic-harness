#!/usr/bin/env python
"""Standard application entrypoint for Streamlit Cloud and local hosting."""
import os
import sys
from pathlib import Path

# Add src to path
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
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())] + sys.argv[1:]
        sys.exit(stcli.main())
else:
    # When imported by Streamlit runner
    from harness.dashboard import main
    main()
