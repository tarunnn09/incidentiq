"""CSV upload validation: check columns, clean bad rows, and tell the user what was dropped."""
import io

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp", "service", "log_level", "message",
    "latency_ms", "error_rate", "cpu_pct", "memory_pct",
]
NUMERIC_COLUMNS = ["latency_ms", "error_rate", "cpu_pct", "memory_pct"]
MAX_ROWS = 20000
MAX_BYTES = 5 * 1024 * 1024  # 5 MB

LEVEL_MAP = {
    "INFO": "INFO", "DEBUG": "INFO", "WARN": "WARN", "WARNING": "WARN",
    "ERROR": "ERROR", "ERR": "ERROR", "CRITICAL": "CRITICAL", "FATAL": "CRITICAL",
}


class CSVValidationError(ValueError):
    """Raised when the uploaded file cannot be analysed. The message is safe to show to users."""


def parse_csv(raw):
    """Turn uploaded bytes into a clean DataFrame. Returns (dataframe, list_of_notes)."""
    if not raw or not raw.strip():
        raise CSVValidationError("The uploaded file is empty.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        raise CSVValidationError("The file is not valid UTF-8 text. Please upload a plain CSV file.")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise CSVValidationError(f"Could not read the file as CSV: {exc}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CSVValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}.")
    if len(df) > MAX_ROWS:
        raise CSVValidationError(f"Too many rows ({len(df)}). The limit is {MAX_ROWS} rows.")

    df = df[REQUIRED_COLUMNS].copy()
    notes = []

    # Convert types. errors="coerce" turns unreadable values into NaN/NaT so we can drop them.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["service"] = df["service"].fillna("").astype(str).str.strip()
    df["message"] = df["message"].fillna("").astype(str).str.strip()

    # Drop bad rows step by step so every dropped row is counted once.
    def drop(mask, reason):
        nonlocal df
        count = int(mask.sum())
        if count:
            notes.append(f"Dropped {count} row(s): {reason}.")
            df = df[~mask]

    drop(df["timestamp"].isna(), "timestamp missing or not a valid date/time")
    drop(df[NUMERIC_COLUMNS].isna().any(axis=1), "a metric value was missing or not a number")
    drop((df["service"] == "") | (df["message"] == ""), "service or message was blank")
    out_of_range = (
        ~np.isfinite(df[NUMERIC_COLUMNS]).all(axis=1)
        | (df["latency_ms"] < 0)
        | ~df["error_rate"].between(0, 100)
        | ~df["cpu_pct"].between(0, 100)
        | ~df["memory_pct"].between(0, 100)
    )
    drop(out_of_range, "a metric was outside its valid range (percentages must be 0-100, latency >= 0)")

    if df.empty:
        raise CSVValidationError("No valid rows were left after cleaning. " + " ".join(notes))

    # Normalise log levels to INFO / WARN / ERROR / CRITICAL.
    levels = df["log_level"].fillna("").astype(str).str.strip().str.upper()
    mapped = levels.map(LEVEL_MAP)
    unknown = int(mapped.isna().sum())
    if unknown:
        notes.append(f"{unknown} row(s) had an unknown log_level and were treated as INFO.")
    df["log_level"] = mapped.fillna("INFO")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, notes
