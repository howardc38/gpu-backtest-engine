"""Regenerate only the public synthetic RSI example."""

from pathlib import Path

from gpu_backtest_examples.data import generate_csv

if __name__ == "__main__":
    print(generate_csv(Path(__file__).with_name("synthetic.csv")))
