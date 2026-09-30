"""Tests for the concurrent manual-ICA review HTTP server."""

import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from osl_ephys.preprocessing.manual_ica.review_server import (
    MAX_POST_BYTES,
    _safe_relpath,
    create_server,
)
from osl_ephys.preprocessing.manual_ica.html import _write_review_html
from osl_ephys.preprocessing.manual_ica.helpers import _build_scores_list


@pytest.fixture
def review_server(tmp_path):
    subject_dir = tmp_path / 'subject01'
    subject_dir.mkdir()
    (subject_dir / 'single_ic.html').write_text('review page', encoding='utf-8')
    server = create_server('127.0.0.1', 0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, tmp_path
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _url(server, path):
    return 'http://127.0.0.1:' + str(server.server_port) + path


def test_idle_connection_does_not_block_static_requests(review_server):
    server, _ = review_server
    idle = socket.create_connection(server.server_address, timeout=2)
    try:
        # A single-threaded HTTPServer blocks here waiting for the idle socket.
        with ThreadPoolExecutor(max_workers=8) as pool:
            bodies = list(pool.map(
                lambda _: urlopen(
                    _url(server, '/subject01/single_ic.html'), timeout=2
                ).read(),
                range(8),
            ))
        assert bodies == [b'review page'] * 8
    finally:
        idle.close()


def test_post_is_saved_and_response_is_complete(review_server):
    server, root = review_server
    body = b'IC000: bad\nIC001: good\n'
    request = Request(
        _url(server, '/subject01/save_label'), data=body, method='POST'
    )
    with urlopen(request, timeout=2) as response:
        assert response.status == 200
        assert response.headers['Content-Length'] == str(len(b'saved'))
        assert response.read() == b'saved'
    assert (root / 'subject01' / 'label.txt').read_bytes() == body
    assert not list((root / 'subject01').glob('.label.txt.*'))


def test_post_rejects_unknown_subject_and_large_body(review_server):
    server, root = review_server
    missing = Request(
        _url(server, '/missing/save_label'), data=b'x', method='POST'
    )
    with pytest.raises(HTTPError, match='HTTP Error 404'):
        urlopen(missing, timeout=2)

    large = Request(
        _url(server, '/subject01/save_label'),
        data=b'x' * (MAX_POST_BYTES + 1),
        method='POST',
    )
    with pytest.raises(HTTPError, match='HTTP Error 413'):
        urlopen(large, timeout=2)
    assert not (root / 'subject01' / 'label.txt').exists()


def test_safe_relpath_rejects_parent_and_symlink_escape(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'link').symlink_to(outside, target_is_directory=True)

    assert _safe_relpath(root, 'subject01') == root / 'subject01'
    assert _safe_relpath(root, '../outside') is None
    assert _safe_relpath(root, 'link') is None


def test_review_badge_includes_flagged_displayed_score(tmp_path):
    scores = [
        {'ecg_flag': True, 'eog': [], 'ga_flag': False},
        {'ecg_flag': False, 'eog': [], 'ga_flag': False,
         'ga_local': 5.95, 'ga_local_thr': 8.0,
         'ga_dominance': 1.04, 'ga_dominance_thr': 1.0},
    ]
    _write_review_html(2, 'fixture', tmp_path, flagged_ics=set(),
                       scores_list=scores)
    for name in ('single_ic.html', 'between_ic.html'):
        page = (tmp_path / name).read_text(encoding='utf-8')
        assert 'const flags       = [true, false];' in page


def test_review_scores_omit_ga_without_slice_timing():
    scores = _build_scores_list(
        1, ecg_scores=None, ecg_idx_auto=None, ecg_threshold=0.1,
        eog_scores_list=[], eog_idx_auto=None, eog_threshold=0.35,
        slice_scores=None,
    )
    assert scores == [{'eog': []}]
