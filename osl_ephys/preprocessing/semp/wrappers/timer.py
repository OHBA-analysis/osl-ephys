from datetime import datetime

import numpy as np
from osl_ephys.utils.logger import log_or_print

from ..utils import proc_userargs


class _TimerRegistry:
    """
    Singleton registry to store running timers and their history across subjects/runs.
    No threading or dataset dependency — purely global, group-level stats.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_TimerRegistry, cls).__new__(cls)
            cls._instance._running = {}   # timer_idx -> {'start': datetime, 'meta': ...}
            cls._instance._history = {}   # timer_idx -> [float durations in seconds]
        return cls._instance

    def start(self, timer_idx, meta=None):
        """Start a timer.

        If one is already running under this idx, the previous subject never
        called ``end_timer`` -- either it is missing from the config, or that
        subject failed mid-chain. Rather than abort the whole batch (the old
        behaviour raised), discard the stale timer with a warning and start
        fresh; its elapsed time is simply not recorded. The rolling group
        history for this idx is untouched, so ``end_timer``'s mean +/- std
        stats stay valid across the remaining subjects.
        """
        if timer_idx in self._running:
            stale = self._running.pop(timer_idx)
            log_or_print(
                f"[TIMER] Warning: timer '{timer_idx}' was still running "
                f"(meta={stale.get('meta')!r}) -- the previous subject never "
                f"ended it (missing end_timer in the config, or that subject "
                f"failed). Discarding it and starting fresh; its time is not "
                f"recorded.")
        self._running[timer_idx] = {'start': datetime.now(), 'meta': meta}

    def end(self, timer_idx):
        """End a timer, return elapsed seconds. Raise if timer not running."""
        if timer_idx not in self._running:
            raise KeyError(f"Timer '{timer_idx}' not found (not running).")
        rec = self._running.pop(timer_idx)
        elapsed = (datetime.now() - rec['start']).total_seconds()
        self._history.setdefault(timer_idx, []).append(elapsed)
        return elapsed

    def get_history(self, timer_idx):
        """Return the list of historical durations for timer_idx (may be empty)."""
        return list(self._history.get(timer_idx, []))

    def clear_history(self, timer_idx=None):
        """Clear history for one timer or all if timer_idx is None."""
        if timer_idx is None:
            self._history.clear()
        else:
            self._history.pop(timer_idx, None)


# module-level singleton instance
_timer_registry = _TimerRegistry()


def start_timer(dataset, userargs):
    """
    Start a named timer.

    Optional userargs:
      - timer_idx: identifier for this timer (default 'default_timer', so the
        common single-timer case needs no args)
      - meta: any object to store alongside the start (for debugging / tracing)
    """
    userargs = proc_userargs(userargs, {'timer_idx': 'default_timer', 'meta': None})
    timer_idx = userargs['timer_idx']
    meta = userargs['meta']

    _timer_registry.start(timer_idx, meta=meta)
    log_or_print(f"[TIMER] Started timer '{timer_idx}'.")
    return dataset


def end_timer(dataset, userargs):
    """
    End a named timer, log its elapsed time and group-level history stats (mean ± std).

    Required userargs:
      - timer_idx: identifier for the timer to end

    Optional userargs:
      - timer_idx: identifier for the timer to end (default 'default_timer')
      - print_stats: bool (default True) whether to log mean±std after appending this run
      - fmt: format for printing numeric values (default '%.3f')
      - ddof: degrees of freedom for std calculation (default 0 -> population std)
    """
    userargs = proc_userargs(userargs, {
        'timer_idx': 'default_timer',
        'print_stats': True,
        'fmt': '%.3f',
        'ddof': 0,
    })
    timer_idx = userargs['timer_idx']
    print_stats = userargs['print_stats']
    fmt = userargs['fmt']
    ddof = userargs['ddof']

    elapsed = _timer_registry.end(timer_idx)

    hist = _timer_registry.get_history(timer_idx)
    hist_arr = np.array(hist, dtype=np.float64)
    n = hist_arr.size
    mean = float(hist_arr.mean()) if n > 0 else float('nan')
    std = float(hist_arr.std(ddof=ddof)) if n > 0 else float('nan')

    log_or_print(f"[TIMER] Ended timer '{timer_idx}'. Elapsed = {fmt % elapsed} s.")
    if print_stats:
        log_or_print(f"[TIMER] History (n={n}) for '{timer_idx}': {fmt % mean} +/- {fmt % std} s.")

    return dataset
