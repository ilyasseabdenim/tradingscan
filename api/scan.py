from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import json
import os
import sys

# Make sibling module importable on Vercel and locally.
sys.path.insert(0, os.path.dirname(__file__))

from scanner_core import ALL_SYMBOLS, run_scan, settings_from_query


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            raw_query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
            settings = settings_from_query(raw_query)

            # Optional symbol subset for faster testing, e.g. ?symbols=AAPL,NVDA,BTC-USD
            symbols_param = raw_query.get("symbols", "").strip()
            if symbols_param:
                wanted = [s.strip().upper() for s in symbols_param.split(",") if s.strip()]
                allowed = set(ALL_SYMBOLS)
                symbols = [s for s in wanted if s in allowed]
                if not symbols:
                    symbols = ALL_SYMBOLS[:10]
            else:
                symbols = ALL_SYMBOLS

            payload = run_scan(settings, symbols=symbols)
            self._send_json(200, payload)
        except Exception as e:
            self._send_json(500, {"error": str(e)})
