#!/usr/bin/env python3
"""Inline profiler for 7c steering validation."""

import time
from collections import defaultdict
from contextlib import contextmanager
import torch

_timings = defaultdict(list)
_enabled = True

def reset():
    """Reset all timings."""
    global _timings
    _timings = defaultdict(list)

def enable():
    """Enable profiling."""
    global _enabled
    _enabled = True

def disable():
    """Disable profiling."""
    global _enabled
    _enabled = False

@contextmanager
def timed(name: str, sync_cuda: bool = True):
    """Context manager for timing a block of code.

    Args:
        name: Name for this timing section
        sync_cuda: If True, synchronize CUDA before and after to get accurate GPU timing
    """
    if not _enabled:
        yield
        return

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    yield

    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    _timings[name].append(time.perf_counter() - t0)

def print_timings(sort_by: str = "total"):
    """Print timing summary.

    Args:
        sort_by: "total" (default), "avg", or "calls"
    """
    if not _timings:
        print("No timings recorded.")
        return

    print("\n" + "=" * 80)
    print("PROFILING RESULTS")
    print("=" * 80)

    # Compute stats
    stats = []
    for name, times in _timings.items():
        total = sum(times)
        avg = total / len(times) if times else 0
        stats.append({
            "name": name,
            "total": total,
            "calls": len(times),
            "avg": avg,
            "min": min(times) if times else 0,
            "max": max(times) if times else 0,
        })

    # Sort
    if sort_by == "avg":
        stats.sort(key=lambda x: -x["avg"])
    elif sort_by == "calls":
        stats.sort(key=lambda x: -x["calls"])
    else:
        stats.sort(key=lambda x: -x["total"])

    # Print
    total_time = sum(s["total"] for s in stats)

    print(f"{'Section':<45} {'Total':>10} {'Calls':>8} {'Avg':>10} {'%':>6}")
    print("-" * 80)

    for s in stats:
        pct = (s["total"] / total_time * 100) if total_time > 0 else 0
        print(f"{s['name']:<45} {s['total']:>9.3f}s {s['calls']:>8} {s['avg']*1000:>9.2f}ms {pct:>5.1f}%")

    print("-" * 80)
    print(f"{'TOTAL':<45} {total_time:>9.3f}s")
    print("=" * 80 + "\n")

def get_timings():
    """Return raw timings dict."""
    return dict(_timings)
