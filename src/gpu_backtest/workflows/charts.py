"""Render generic parameter/effect-size plots as a directly openable HTML file."""

from pathlib import Path

import numpy as np
import pandas as pd

from gpu_backtest.core.statistics import _STAT_COLS


def write_charts(input_files, output_file):
    try:
        import altair as alt
    except ImportError as exc:
        raise ValueError("Charts require: pip install 'gpu-backtest-engine[viz]'") from exc
    charts = []
    for filename in input_files:
        data = pd.read_csv(filename)
        if "effect_size" not in data or data.empty:
            raise ValueError(f"{filename}: requires non-empty top data with effect_size")
        parameters = [column for column in data if column not in _STAT_COLS]
        if not parameters:
            raise ValueError(f"{filename}: no parameter columns")
        if not np.isfinite(data[[*parameters, "effect_size"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{filename}: parameter/effect-size values must be finite")
        for parameter in parameters:
            charts.append(
                alt.Chart(data)
                .mark_circle(size=40, opacity=0.6)
                .encode(
                    x=alt.X(f"{parameter}:Q", title=parameter.replace("_", " ").title()),
                    y=alt.Y("effect_size:Q", title="Effect size"),
                    tooltip=[*parameters, "effect_size"],
                )
                .properties(width=700, height=240, title=f"{Path(filename).name}: {parameter}")
                .interactive(name=f"zoom_{len(charts)}")
            )
    if not charts:
        raise ValueError("At least one top CSV is required")
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with alt.data_transformers.enable("default", max_rows=None):
        alt.vconcat(*charts).save(str(output))
    return output
