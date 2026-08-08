import os
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("PORT", 10000))


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

        self.wfile.write(
            b"Render port test is working!"
        )

    def log_message(self, format, *args):
        print(format % args)


server = HTTPServer(
    ("0.0.0.0", PORT),
    Handler
)

print(f"Starting test server on 0.0.0.0:{PORT}", flush=True)

server.serve_forever()
