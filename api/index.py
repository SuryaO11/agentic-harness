"""Vercel Serverless Function entrypoint for Agentic Harness."""
import os
import sys
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler

# Add src to Python path
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def app(scope_or_environ, receive_or_start_response=None, send=None):
    """Hybrid WSGI/ASGI entrypoint for Vercel."""
    body = json.dumps({
        "status": "online",
        "service": "Agentic Harness v1.2 API",
        "repository": "https://github.com/SuryaO11/agentic-harness",
        "interactive_dashboard": "https://share.streamlit.io"
    }, indent=2).encode("utf-8")

    if send is not None:
        async def asgi_handler():
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode("utf-8")],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })
        return asgi_handler()

    status = "200 OK"
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
    ]
    receive_or_start_response(status, headers)
    return [body]


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
            "repository": "https://github.com/SuryaO11/agentic-harness",
            "notice": "Interactive UI runs on Streamlit Community Cloud (https://share.streamlit.io)."
        }
        self.wfile.write(json.dumps(response, indent=2).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        res = {"status": "received", "message": "Agentic Harness API active."}
        self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))
