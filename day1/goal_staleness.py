"""Goal-staleness sweep: how closed-loop behaviour decays as the cached goal ages.

GlanceWAM conditions the action head on a goal latent the video DiT generates, refreshed
every ``H_g`` native rows (``goal_horizon_rows``, 48 = 2.4 s at LIBERO's 20 Hz). Within a
refresh cycle the held goal ages while the model is told a *shrinking* effective distance
(H_g -> 0). Past H_g that distance clamps to 1 row, yet the goal keeps getting older -- the
goal-side analogue of executing a stale plan. The upstream participants already showed the
generated goal does not survive closed loop when used as a plan; this sweep isolates HOW
staleness, specifically, degrades control.

The knob is ``refresh_rows``: the interval between goal regenerations. ``refresh_rows == H_g``
is the trained cadence (the baseline). ``refresh_rows > H_g`` holds one goal longer than it
was ever trained to be, so ``refresh_rows - H_g`` is the extra staleness in native rows. A
very large value never refreshes after the first generation -- the frozen-goal extreme.

This module holds the pure cadence arithmetic (the eval client imports the same helpers so
the simulator and the live loop cannot drift) plus the aggregation that turns per-run results
into the decay curve. No simulator, no model, no I/O beyond reading result JSON.
"""
import json
from pathlib import Path


def refresh_grid(H_g, multipliers):
    """Refresh intervals (native rows) for a staleness sweep, always including H_g.

    ``multipliers`` are multiples of H_g (e.g. [1.0, 1.5, 2.0, 3.0]). The trained cadence
    H_g is always added so every curve has its own baseline, and duplicates collapse — so a
    multiplier of 1.0 and the injected H_g do not produce two identical arms.
    """
    if float(H_g) <= 0:
        raise ValueError('H_g must be positive')
    rows = {int(round(float(m) * float(H_g))) for m in multipliers}
    rows = {r for r in rows if r > 0}
    rows.add(int(round(float(H_g))))
    return sorted(rows)


def should_refresh(has_goal, steps_since_refresh, refresh_rows):
    """Whether the client regenerates the goal on this query step."""
    if int(refresh_rows) < 1:
        raise ValueError('refresh_rows must be at least 1')
    return (not bool(has_goal)) or int(steps_since_refresh) >= int(refresh_rows)


def effective_goal_rows(H_g, steps_since_refresh):
    """The goal distance the action head is told: H_g right after a refresh, shrinking
    toward 0, and clamped to 1 row once the held goal is older than H_g (the staleness
    signal the model can no longer see)."""
    return float(max(1.0, float(H_g) - float(steps_since_refresh)))


def staleness_rows(refresh_rows, H_g):
    """Extra staleness a longer refresh interval introduces, in native rows (>= 0)."""
    return max(0, int(round(float(refresh_rows))) - int(round(float(H_g))))


def aggregate_staleness(records, H_g=None):
    """Turn per-episode results into the decay curve.

    ``records`` is an iterable of dicts carrying at least ``refresh_rows`` (int) and
    ``success`` (bool), and optionally ``steps`` (float). Returns ``{'curve': [...]}`` sorted
    by refresh_rows; each curve point reports episodes, successes, success_rate, mean_steps,
    and -- when ``H_g`` is given -- the extra staleness in rows.
    """
    bucket = {}
    for record in records:
        rows = int(record['refresh_rows'])
        entry = bucket.setdefault(rows, dict(refresh_rows=rows, episodes=0, successes=0, steps=[]))
        entry['episodes'] += 1
        entry['successes'] += int(bool(record['success']))
        if record.get('steps') is not None:
            entry['steps'].append(float(record['steps']))
    curve = []
    for rows in sorted(bucket):
        entry = bucket[rows]
        n = entry['episodes']
        point = dict(refresh_rows=rows, episodes=n, successes=entry['successes'],
                     success_rate=(entry['successes'] / n) if n else 0.0,
                     mean_steps=(sum(entry['steps']) / len(entry['steps'])) if entry['steps'] else None)
        if H_g is not None:
            point['staleness_rows'] = staleness_rows(rows, H_g)
        curve.append(point)
    return dict(curve=curve)


def load_driver_results(path):
    """Read one diagnose-driver results.json into the flat records this module aggregates.

    The driver writes ``episode_results`` per (task, episode); each already carries
    ``success`` and ``steps``. ``refresh_rows`` is injected by the sweep runner so the
    aggregator knows which arm a file belongs to.
    """
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return list(payload.get('episode_results') or [])
