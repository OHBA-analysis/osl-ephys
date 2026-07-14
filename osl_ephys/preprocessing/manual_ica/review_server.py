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
import sys
from pathlib import Path


_FILE_FOR_ENDPOINT = {
    'save_label': 'label.txt',
    'save_bads':  'bads.txt',
}


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
    def do_POST(self):
        endpoint = self.path.rsplit('/', 1)[-1]
        out_name = _FILE_FOR_ENDPOINT.get(endpoint)
        if out_name is None:
            self.send_response(405)
            self.end_headers()
            return

        # Strip the trailing endpoint name --- the rest is the subject dir.
        rel_dir = os.path.dirname(self.path.lstrip('/'))
        target_dir = _safe_relpath(os.getcwd(), rel_dir)
        if target_dir is None:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'forbidden: path escapes server root')
            print(f'[review] REFUSED escape attempt: {self.path!r}')
            return

        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8')
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / out_name
        out_path.write_text(body, encoding='utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'saved')
        print(f'[review] saved {out_path}')

    def log_message(self, fmt, *args):
        print(f'[review] {self.address_string()} {fmt % args}')


def main():
    p = argparse.ArgumentParser(description='ICA manual-review HTTP server')
    p.add_argument('port', nargs='?', type=int, default=8000)
    p.add_argument('--host', default='127.0.0.1',
                   help='Interface to bind. Default 127.0.0.1 (loopback). '
                        'Use 0.0.0.0 to expose on the network.')
    args = p.parse_args()

    server = http.server.HTTPServer((args.host, args.port), ReviewHandler)
    print(f'ICA review server -> http://{args.host}:{args.port}/')
    if args.host == '0.0.0.0':
        print('[review] WARNING: bound to 0.0.0.0 --- POST endpoints '
              'are reachable from the network.')
    print('Ctrl-C to stop.')
    server.serve_forever()


if __name__ == '__main__':
    main()
