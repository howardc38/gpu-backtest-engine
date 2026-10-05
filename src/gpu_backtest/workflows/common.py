"""Find parameter combinations present in every ranked top artifact."""

import csv
import os
from decimal import Decimal, InvalidOperation, getcontext
from pathlib import Path

getcontext().prec = 50
VALUE_COLUMN = "effect_size"


def extract_label_from_filename(filename):
    base_name = Path(filename).stem
    parts = base_name.split("_")
    for i, part in enumerate(parts):
        if part == "to" and 0 < i < len(parts) - 1:
            before_years = [
                segment
                for segment in parts[i - 1].split("-")
                if segment.isdigit() and len(segment) == 4
            ]
            after_years = [
                segment
                for segment in parts[i + 1].split("-")
                if segment.isdigit() and len(segment) == 4
            ]
            if before_years and after_years:
                return f"{before_years[0]}_{after_years[0]}"
    for i, part in enumerate(parts[:-1]):
        if part.isdigit() and len(part) == 4:
            following = parts[i + 1]
            if following.isdigit() and len(following) == 4:
                return f"{part}_{following}"
    return base_name


def _decimal(raw_value, context):
    try:
        value = Decimal(str(raw_value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{context}: expected a decimal number, got {raw_value!r}") from exc
    if not value.is_finite():
        raise ValueError(f"{context}: value must be finite")
    return value


def _format_decimal(value):
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _read_top_artifact(filename, key_columns, value_column=VALUE_COLUMN):
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {filename}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{filename}: duplicate CSV column names")
        required = [*key_columns, value_column]
        missing = [column for column in required if column not in fieldnames]
        if missing:
            raise ValueError(f"{filename}: missing required columns: {missing}")
        data = {}
        previous_effect = None
        for row_number, row in enumerate(reader, start=2):
            key = tuple(
                (
                    _decimal(row[column], f"{filename}:{row_number}:{column}")
                    for column in key_columns
                )
            )
            effect = _decimal(row[value_column], f"{filename}:{row_number}:{value_column}")
            if key in data:
                raise ValueError(f"{filename}:{row_number}: duplicate parameter combination {key}")
            if previous_effect is not None and effect > previous_effect:
                raise ValueError(
                    f"{filename}:{row_number}: {value_column} is not sorted descending"
                )
            previous_effect = effect
            data[key] = effect
    if not data:
        raise ValueError(f"{filename}: no data rows")
    return data


def _resolve_labels(input_files, configured_labels=None):
    labels = configured_labels or [extract_label_from_filename(path) for path in input_files]
    if len(labels) != len(input_files):
        raise ValueError("The number of labels must match the number of input files")
    labels = [label.strip() for label in labels]
    if any((not label for label in labels)):
        raise ValueError("Labels must not be empty")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Labels must be unique: {labels}")
    return labels


def process_data(config):
    input_files = config["input_files"]
    key_columns = config["key_columns"]
    output_file = Path(config["output_file"])
    output_file.unlink(missing_ok=True)
    if len(input_files) < 2:
        raise ValueError("At least two input files are required")
    if not key_columns or len(key_columns) != len(set(key_columns)):
        raise ValueError("key_columns must contain unique column names")
    labels = _resolve_labels(input_files, config.get("file_labels"))
    all_data = [_read_top_artifact(path, key_columns) for path in input_files]
    common_keys = set(all_data[0])
    for data in all_data[1:]:
        common_keys.intersection_update(data)
    if not common_keys:
        pairwise = []
        for i in range(len(all_data)):
            for j in range(i + 1, len(all_data)):
                pairwise.append(
                    f"{labels[i]} & {labels[j]}={len(set(all_data[i]) & set(all_data[j]))}"
                )
        raise ValueError(
            "No common combinations found; pairwise intersections: " + ", ".join(pairwise)
        )
    result = []
    for key in common_keys:
        effects = [data[key] for data in all_data]
        average = sum(effects, Decimal(0)) / Decimal(len(effects))
        result.append({"key": key, "effects": effects, "average": average, "minimum": min(effects)})
    result.sort(key=lambda item: (-item["average"], *item["key"]))
    effect_columns = [f"{VALUE_COLUMN}_{label}" for label in labels]
    fieldnames = [*key_columns, *effect_columns, f"avg_{VALUE_COLUMN}", f"min_{VALUE_COLUMN}"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_file.with_name(output_file.name + ".tmp")
    try:
        with temp_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for item in result:
                row = {
                    column: _format_decimal(value)
                    for column, value in zip(key_columns, item["key"])
                }
                row.update(
                    {
                        column: _format_decimal(value)
                        for column, value in zip(effect_columns, item["effects"])
                    }
                )
                row[f"avg_{VALUE_COLUMN}"] = _format_decimal(item["average"])
                row[f"min_{VALUE_COLUMN}"] = _format_decimal(item["minimum"])
                writer.writerow(row)
        os.replace(temp_file, output_file)
    finally:
        if temp_file.exists():
            temp_file.unlink()
    print(f"Common combinations: {len(result)}")
    print(f"Results saved to: {output_file}")
    return result
