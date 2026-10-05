"""
Logging utilities for HRRR Weather Lake pipelines.
Provides consistent timestamped log formatting and structured run summary banners.
"""

import datetime
import logging
import sys
import time
from typing import Any, Dict, Optional


def setup_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure root logging with a standardized timestamped format.
    Flushes line-buffering on stdout/stderr so cron /tmp log files update in real time.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Retrieve attribute or dictionary key safely."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_streaming_metrics(query: Any) -> Dict[str, Any]:
    """
    Extract row count and batch statistics from a completed Spark StreamingQuery.
    Safely handles both dictionary and object representations of StreamingQueryProgress.
    """
    if query is None:
        return {"batches": 0, "rows": 0, "batch_details": []}

    progress_list = getattr(query, "recentProgress", None) or []
    total_input_rows = 0
    batches_run = len(progress_list)
    batch_details = []

    for p in progress_list:
        num_rows = _get_attr_or_key(p, "numInputRows", 0) or 0
        batch_id = _get_attr_or_key(p, "batchId", None)
        timestamp = _get_attr_or_key(p, "timestamp", None)
        total_input_rows += int(num_rows)
        batch_details.append({
            "batch_id": batch_id,
            "rows": int(num_rows),
            "timestamp": timestamp,
        })

    return {
        "batches": batches_run,
        "rows": total_input_rows,
        "batch_details": batch_details,
    }


def log_run_summary(
    logger: logging.Logger,
    stage_name: str,
    start_time: float,
    metrics: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a distinct, structured summary banner indicating a successful pipeline run.
    Ensures the last successful run stands out clearly in cron /tmp log files.
    """
    end_time = time.time()
    duration_sec = end_time - start_time
    completed_dt = datetime.datetime.now()
    completed_str = completed_dt.strftime("%Y-%m-%d %H:%M:%S")

    metrics = metrics or {}
    extra_info = extra_info or {}

    rows = metrics.get("rows", 0)
    batches = metrics.get("batches", 0)

    banner_width = 80
    border = "=" * banner_width

    lines = [
        border,
        f"LAST SUCCESSFUL RUN SUMMARY — {stage_name.upper()}",
        f"Status:             SUCCESS",
        f"Completed At:       {completed_str}",
        f"Duration:           {duration_sec:.2f}s ({datetime.timedelta(seconds=int(duration_sec))})",
        f"Batches Executed:   {batches}",
        f"Rows/Files Handled: {rows:,}",
    ]

    for k, v in extra_info.items():
        lines.append(f"{k + ':':<20} {v}")

    lines.append(border)

    summary_block = "\n" + "\n".join(lines)
    logger.info(summary_block)

    # Grep-friendly one-liner
    logger.info(
        "[LAST_RUN_SUCCESS] stage=\"%s\" completed_at=\"%s\" duration_s=%.2f rows=%d batches=%d",
        stage_name,
        completed_str,
        duration_sec,
        rows,
        batches,
    )
