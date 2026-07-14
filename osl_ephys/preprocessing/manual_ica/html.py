"""Render the per-subject HTML review pages from the bundled templates.

Templates live in ``templates/`` next to this file. They use a tiny
``__PLACEHOLDER__`` substitution scheme (no Jinja or other dep) so they're
trivially serveable as static files by the bundled review server.
"""
import json
from pathlib import Path

_TEMPLATE_DIR        = Path(__file__).parent / 'templates'
_SINGLE_IC_TEMPLATE  = _TEMPLATE_DIR / 'single_ic.html'
_BETWEEN_IC_TEMPLATE = _TEMPLATE_DIR / 'between_ic.html'


def _write_review_html(n_components, subject, save_dir,
                       flagged_ics=None, scores_list=None,
                       n_zoom=0, total_dur=0.0, zoom_window=20.0,
                       supp_chs=None):
    """Render the two review pages (``single_ic.html``, ``between_ic.html``)
    for one subject.

    Each page only fetches the SVGs it needs, so the prefetcher isn't dragged
    across both views. Pages cross-link via the ``V`` shortcut and preserve
    the current IC + zoom window via the URL hash.

    Returns
    -------
    Path
        Path to ``single_ic.html`` (the default landing page).
    """
    img_list   = json.dumps([f'ic_{i:03d}.svg' for i in range(n_components)])
    first_img  = f'ic_{0:03d}.svg' if n_components > 0 else ''
    first_zoom = f'zoomed/ic_{0:03d}_w{0:02d}.svg' if n_components > 0 and n_zoom > 0 else ''
    fset       = set(flagged_ics) if flagged_ics is not None else set()
    flag_list  = json.dumps([i in fset for i in range(n_components)])
    if scores_list is None:
        scores_list = [{'eog': []} for _ in range(n_components)]
    scores_json = json.dumps(scores_list)
    supp_json   = json.dumps([
        {'label': d, 'file': s} for d, s in (supp_chs or [])
    ])

    def _fill(tmpl):
        return (tmpl
                .replace('__SUBJECT__',      subject)
                .replace('__N_COMP_03D__',   f'{n_components:03d}')
                .replace('__N_COMP__',       str(n_components))
                .replace('__N_ZOOM__',       str(n_zoom))
                .replace('__TOTAL_DUR__',    f'{float(total_dur):.6f}')
                .replace('__ZOOM_WINDOW__',  f'{float(zoom_window):.6f}')
                .replace('__FIRST_IMG__',    first_img)
                .replace('__FIRST_ZOOM__',   first_zoom)
                .replace('__IMG_LIST__',     img_list)
                .replace('__FLAG_LIST__',    flag_list)
                .replace('__SCORES_LIST__',  scores_json)
                .replace('__SUPP_CH_LIST__', supp_json))

    single_path  = save_dir / 'single_ic.html'
    between_path = save_dir / 'between_ic.html'
    single_path.write_text(
        _fill(_SINGLE_IC_TEMPLATE.read_text(encoding='utf-8')),
        encoding='utf-8',
    )
    between_path.write_text(
        _fill(_BETWEEN_IC_TEMPLATE.read_text(encoding='utf-8')),
        encoding='utf-8',
    )
    return single_path
