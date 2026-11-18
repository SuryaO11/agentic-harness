"""Vercel Serverless Function entrypoint for Agentic Harness."""
from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class handler(BaseHTTPRequestHandler):
    """Vercel HTTP handler for Agentic Harness."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response = {
            "status": "online",
            "service": "Agentic Harness v1.2",
            "description": "Explainable Self-Correcting Reliability Framework for LLMs",
            "endpoints": {
                "health": "/api/index",
                "repository": "https://github.com/SuryaO11/agentic-harness"
            },
            "notice": (
                "The interactive dashboard uses Streamlit (stateful WebSockets), "
                "which is natively hosted for free on Streamlit Community Cloud "
                "at https://share.streamlit.io (deploy using entrypoint 'run_dashboard.py' or 'app.py')."
            )
        }
        self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))

    def do_POST(self):
        """Optional lightweight API evaluation endpoint."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        res = {
            "status": "received",
            "message": "Agentic Harness API endpoint active."
        }
        self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))

