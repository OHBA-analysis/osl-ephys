#!/usr/bin/env python3
"""Tiny HTTP server for ICA review - serves files AND saves label.txt + bads.txt.

Usage (run from the ICA root folder, i.e. the folder containing subject dirs):

    osl-ica-review [port] [--host HOST]                     # console_script
    python -m osl_ephys.preprocessing.manual_ica.review_server [port] [--host HOST]

By default the server binds to ``127.0.0.1`` so only browsers on the same
machine can hit the save endpoints. Pass ``--host 0.0.0.0`` if you really
want to expose the review page on the network (the save endpoints write
files inside the cwd; see _safe_relpath below for the path-traversal guard
that protects against ``POST /../etc/save_label`` style attacks).

Then open:  http://localhost:<port>/<subject>/single_ic.html

POST endpoints (per-subject):
    /<subject>/save_label  -> writes /<subject>/label.txt
    /<subject>/save_bads   -> writes /<subject>/bads.txt
All other requests are served as static files (same as python -m http.server).
"""
import argparse
import http.server
import os
import tempfile
import threading
from functools import partial
from pathlib import Path
from urllib.parse import unquote, urlsplit


_FILE_FOR_ENDPOINT = {
    'save_label': 'label.txt',
    'save_bads':  'bads.txt',
}

# label.txt and bads.txt are normally only a few KiB. Keep a generous bound so
# a malformed or exposed request cannot make the review server allocate an
# arbitrary body in memory.
MAX_POST_BYTES = 1024 * 1024
_WRITE_LOCK = threading.Lock()


def _safe_relpath(server_root, requested):
    """Resolve ``requested`` (a server-relative path) under ``server_root``.

    Returns the absolute Path if it stays inside server_root, else None.
    Defends against ``..`` segments, absolute paths, and symlink escapes.
    """
    server_root = Path(server_root).resolve()
    candidate = (server_root / requested.lstrip('/')).resolve()
    try:
        candidate.relative_to(server_root)
    except ValueError:
        return None
    return candidate


class ReviewHandler(http.server.SimpleHTTPRequestHandler):
    def _send_text(self, status, message):
        """Send a complete response that proxies can delimit reliably."""
        body = message.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request_path = unquote(urlsplit(self.path).path)
        endpoint = request_path.rsplit('/', 1)[-1]
        out_name = _FILE_FOR_ENDPOINT.get(endpoint)
        if out_name is None:
            self._send_text(405, 'method not allowed')
            return

        # Strip the trailing endpoint name --- the rest is the subject dir.
        rel_dir = os.path.dirname(request_path.lstrip('/'))
        target_dir = _safe_relpath(self.server.review_root, rel_dir)
        if target_dir is None:
            self._send_text(403, 'forbidden: path escapes server root')
            print(f'[review] REFUSED escape attempt: {self.path!r}')
            return
        if not target_dir.is_dir():
            self._send_text(404, 'review subject directory not found')
            return

        length_header = self.headers.get('Content-Length')
        if length_header is None:
            self._send_text(411, 'Content-Length is required')
            return
        try:
            length = int(length_header)
        except ValueError:
            self._send_text(400, 'invalid Content-Length')
            return
        if length < 0:
            self._send_text(400, 'invalid Content-Length')
            return
        if length > MAX_POST_BYTES:
            self.close_connection = True
            self._send_text(413, 'request body is too large')
            return
        try:
            body = self.rfile.read(length).decode('utf-8')
        except UnicodeDecodeError:
            self._send_text(400, 'request body must be UTF-8')
            return

        out_path = target_dir / out_name
        # Atomic replacement means apply_manual_ica can never observe a
        # partially-written review, even if autosave and reading overlap.
        with _WRITE_LOCK:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', dir=target_dir,
                    prefix='.' + out_name + '.', delete=False,
                ) as tmp:
                    tmp.write(body)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    tmp_path = Path(tmp.name)
                os.replace(tmp_path, out_path)
            finally:
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()

        self._send_text(200, 'saved')
        print(f'[review] saved {out_path}')

    def log_message(self, fmt, *args):
        print(f'[review] {self.address_string()} {fmt % args}')


class ReviewServer(http.server.ThreadingHTTPServer):
    """Concurrent local server resilient to idle browser/proxy connections."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, server_address, review_root):
        self.review_root = Path(review_root).resolve()
        handler = partial(ReviewHandler, directory=str(self.review_root))
        super().__init__(server_address, handler)

    def get_request(self):
        request, address = super().get_request()
        # An idle VS Code port-forward connection must not retain a worker
        # forever. This timeout is per blocking socket operation, not a total
        # transfer deadline.
        request.settimeout(30)
        return request, address


def create_server(host='127.0.0.1', port=8000, review_root=None):
    """Create a review server; separated from ``main`` for testing/reuse."""
    return ReviewServer((host, port), review_root or Path.cwd())


def main():
    p = argparse.ArgumentParser(description='ICA manual-review HTTP server')
    p.add_argument('port', nargs='?', type=int, default=8000)
    p.add_argument('--host', default='127.0.0.1',
                   help='Interface to bind. Default 127.0.0.1 (loopback). '
                        'Use 0.0.0.0 to expose on the network.')
    args = p.parse_args()

    server = create_server(args.host, args.port)
    print(f'ICA review server -> http://{args.host}:{args.port}/')
    if args.host == '0.0.0.0':
        print('[review] WARNING: bound to 0.0.0.0 --- POST endpoints '
              'are reachable from the network.')
    print('Ctrl-C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[review] stopping')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
