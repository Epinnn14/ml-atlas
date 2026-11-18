"""Serve the downloaded ML Atlas site on this computer and open it in the browser.

Usage: python serve.py [port]      (start.bat / start.sh run this for you)
Browsers refuse to run the Python engine from a file:// page, so the site needs a small local
web server. This one sets the right content types for .mjs and .wasm on every system.
"""
import http.server
import os
import sys
import threading
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
        ".json": "application/json", ".wasm": "application/wasm", ".svg": "image/svg+xml",
        ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".zip": "application/zip",
        ".whl": "application/zip", ".png": "image/png",
    }

    def log_message(self, *args):
        pass


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}/"
    print(f"ML Atlas is running at {url}  (close this window or press Ctrl+C to stop)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
