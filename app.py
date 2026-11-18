#!/usr/bin/env python
"""Standard application entrypoint for Streamlit Cloud and Vercel."""
import os
import sys
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler

# Add src to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# --- WSGI callable for Vercel / serverless runtime ---
def app(environ, start_response):
    """Standard WSGI entrypoint for Vercel and serverless platforms."""
    status = "200 OK"
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Agentic Harness v1.2</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #1f2937; background: #f9fafb; }
        .card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 32px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        h1 { color: #111827; margin-top: 0; }
        .badge { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; display: inline-block; }
        .btn { display: inline-block; background: #2563eb; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; margin-top: 16px; margin-right: 12px; }
        .btn:hover { background: #1d4ed8; }
        .btn-secondary { background: #4b5563; }
        .btn-secondary:hover { background: #374151; }
        .note-box { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 16px; margin: 20px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">Service: Active</span>
        <h1>🤖 Agentic Harness v1.2</h1>
        <p>Explainable Self-Correcting Reliability Framework for Large Language Models.</p>
        
        <div class="note-box">
            <strong>🌐 Streamlit Observability Dashboard:</strong><br>
            The full interactive UI (with query runner, validator breakdown, critic diagnostic cards, and SQLite cache metrics) runs on <strong>Streamlit</strong>, which uses persistent stateful WebSockets.<br><br>
            Deploy or access it for free with 1 click on <strong>Streamlit Community Cloud</strong> (<a href="https://share.streamlit.io" target="_blank">share.streamlit.io</a>) using file <code>app.py</code> or <code>run_dashboard.py</code>.
        </div>

        <a class="btn" href="https://github.com/SuryaO11/agentic-harness" target="_blank">View GitHub Repository</a>
        <a class="btn btn-secondary" href="https://share.streamlit.io" target="_blank">Open Streamlit Cloud</a>
    </div>
</body>
</html>"""
    response_body = html_content.encode("utf-8")
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(response_body))),
    ]
    start_response(status, headers)
    return [response_body]


# --- Standard BaseHTTPRequestHandler for Vercel Python runtime ---
class handler(BaseHTTPRequestHandler):
    """Vercel HTTP handler."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        data = {
            "status": "online",
            "name": "Agentic Harness v1.2",
            "description": "Explainable Self-Correcting Reliability Framework for LLMs",
            "repository": "https://github.com/SuryaO11/agentic-harness"
        }
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))


# --- Streamlit Execution Path (when launched via streamlit run or python) ---
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
