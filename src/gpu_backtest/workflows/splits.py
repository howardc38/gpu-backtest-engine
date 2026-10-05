"""Split OHLCV CSVs and record boundaries and content hashes."""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

SPLIT_CONTRACT_VERSION = 2


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_date_column(df):
    candidates = {"time", "date", "datetime", "timestamp"}
    for col in df.columns:
        if col.lower() in candidates:
            return col
    raise ValueError("No date column found")


def _read_input(input_file, date_column=None):
    df = pd.read_csv(input_file)
    if date_column is None:
        date_column = detect_date_column(df)
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found")
    if df.empty:
        raise ValueError(f"Input CSV is empty: {input_file}")
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors="raise")
    except Exception as exc:
        raise ValueError(f"Column '{date_column}' contains invalid timestamps") from exc
    if df[date_column].isna().any():
        raise ValueError(f"Column '{date_column}' contains missing timestamps")
    if df[date_column].duplicated().any():
        raise ValueError(f"Column '{date_column}' contains duplicate timestamps")
    if not df[date_column].is_monotonic_increasing:
        raise ValueError(f"Column '{date_column}' must be strictly increasing")
    return (df, date_column)


def _bound(value, tz, *, end_exclusive=False):
    result = pd.to_datetime(value, format="%d-%m-%Y")
    if tz is not None:
        result = result.tz_localize(tz)
    if end_exclusive:
        result += pd.Timedelta(days=1)
    return result


def _output_dir(input_file):
    parent = Path(input_file).resolve().parent
    output = parent if parent.name == "output" else parent / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output


def _iso(value):
    return None if value is None else value.isoformat()


def _write_splits(input_file, date_column, source_rows, filtered_df, specs, mode):
    output_dir = _output_dir(input_file)
    base_name = Path(input_file).stem
    records = []
    for index, (label, split_start, split_end, split_df) in enumerate(specs, start=1):
        if split_df.empty:
            raise ValueError(f"Split {index} ('{label}') is empty")
        start_str = split_start.strftime("%d-%m-%Y")
        end_str = (split_end - pd.Timedelta(nanoseconds=1)).strftime("%d-%m-%Y")
        filename = f"{base_name}_split{index}_{start_str}_to_{end_str}.csv"
        output_file = output_dir / filename
        split_df.to_csv(output_file, index=False)
        records.append(
            {
                "index": index,
                "label": label,
                "file": filename,
                "rows": int(len(split_df)),
                "first_timestamp": _iso(split_df[date_column].iloc[0]),
                "last_timestamp": _iso(split_df[date_column].iloc[-1]),
                "requested_start": _iso(split_start),
                "requested_end_exclusive": _iso(split_end),
                "sha256": _sha256(output_file),
            }
        )
        print(f"Saved split {index}: {output_file} ({len(split_df)} rows)")
    manifest = {
        "schema_version": 2,
        "split_contract_version": SPLIT_CONTRACT_VERSION,
        "mode": mode,
        "source_file": Path(input_file).name,
        "source_sha256": _sha256(input_file),
        "date_column": date_column,
        "source_rows": int(source_rows),
        "filtered_rows": int(len(filtered_df)),
        "filtered_first_timestamp": _iso(filtered_df[date_column].iloc[0]),
        "filtered_last_timestamp": _iso(filtered_df[date_column].iloc[-1]),
        "splits": records,
    }
    manifest_path = output_dir / "split_manifest.json"
    temp_path = manifest_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, manifest_path)
    print(f"Split manifest: {manifest_path}")
    return manifest_path


def split_csv_by_date_range(input_file, start_date, end_date, num_splits, date_column=None):
    """Split by date. Three splits mean full range, first half, and second half."""
    if num_splits < 1:
        raise ValueError("num_splits must be at least 1")
    print(f"Reading CSV file: {input_file}")
    df, date_column = _read_input(input_file, date_column)
    tz = df[date_column].dt.tz
    start = _bound(start_date, tz)
    end_exclusive = _bound(end_date, tz, end_exclusive=True)
    if start >= end_exclusive:
        raise ValueError("start_date must not be after end_date")
    filtered_df = df[(df[date_column] >= start) & (df[date_column] < end_exclusive)].copy()
    if filtered_df.empty:
        raise ValueError(f"No data found between {start_date} and {end_date}")
    print(f"Found {len(filtered_df)} rows between {start_date} and {end_date}")
    if num_splits == 3:
        midpoint = start + (end_exclusive - start) / 2
        first = filtered_df[filtered_df[date_column] <= midpoint]
        second = filtered_df[filtered_df[date_column] > midpoint]
        first_ids = set(first.index)
        second_ids = set(second.index)
        full_ids = set(filtered_df.index)
        if first_ids & second_ids or first_ids | second_ids != full_ids:
            raise RuntimeError(
                "Split contract failed: half ranges must be disjoint and cover the full range"
            )
        specs = [
            ("full", start, end_exclusive, filtered_df),
            ("first_half", start, midpoint + pd.Timedelta(nanoseconds=1), first),
            ("second_half", midpoint + pd.Timedelta(nanoseconds=1), end_exclusive, second),
        ]
        return _write_splits(
            input_file, date_column, len(df), filtered_df, specs, "full_and_halves"
        )
    boundaries = [start + (end_exclusive - start) * i / num_splits for i in range(num_splits + 1)]
    specs = []
    covered = set()
    for i in range(num_splits):
        split_start, split_end = (boundaries[i], boundaries[i + 1])
        split_df = filtered_df[
            (filtered_df[date_column] >= split_start) & (filtered_df[date_column] < split_end)
        ]
        ids = set(split_df.index)
        if covered & ids:
            raise RuntimeError("Split contract failed: generated ranges overlap")
        covered |= ids
        specs.append((f"part_{i + 1}", split_start, split_end, split_df))
    if covered != set(filtered_df.index):
        raise RuntimeError("Split contract failed: generated ranges do not cover the filtered data")
    return _write_splits(input_file, date_column, len(df), filtered_df, specs, "partition")


def split_csv_by_custom_ranges(input_file, ranges, date_column=None):
    """Write explicit DD-MM-YYYY ranges; overlap is allowed by definition."""
    df, date_column = _read_input(input_file, date_column)
    tz = df[date_column].dt.tz
    specs = []
    included = set()
    for i, (start_text, end_text) in enumerate(ranges, start=1):
        start = _bound(start_text, tz)
        end_exclusive = _bound(end_text, tz, end_exclusive=True)
        if start >= end_exclusive:
            raise ValueError(f"Custom range {i} has start after end")
        split_df = df[(df[date_column] >= start) & (df[date_column] < end_exclusive)]
        included |= set(split_df.index)
        specs.append((f"range_{i}", start, end_exclusive, split_df))
    filtered_df = df.loc[sorted(included)]
    return _write_splits(input_file, date_column, len(df), filtered_df, specs, "custom_ranges")
