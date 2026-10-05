"""Analysis commands consume generic input/results rather than strategy code."""


def register(subparsers):
    split = subparsers.add_parser("split")
    split.add_argument("--input", required=True)
    split.add_argument("--start-date")
    split.add_argument("--end-date")
    split.add_argument("--num-splits", type=int, default=3)
    split.set_defaults(handler=execute)
    common = subparsers.add_parser("common")
    common.add_argument("--input", nargs="+", required=True)
    common.add_argument("--keys", required=True)
    common.add_argument("--labels")
    common.add_argument("--output", required=True)
    common.set_defaults(handler=execute)
    charts = subparsers.add_parser("charts")
    charts.add_argument("--input", nargs="+", required=True)
    charts.add_argument("--output", required=True)
    charts.set_defaults(handler=execute)


def execute(args):
    if args.command == "split":
        from gpu_backtest.workflows.splits import _read_input, split_csv_by_date_range

        data, column = _read_input(args.input, "Time")
        split_csv_by_date_range(
            args.input,
            args.start_date or data[column].min().strftime("%d-%m-%Y"),
            args.end_date or data[column].max().strftime("%d-%m-%Y"),
            args.num_splits,
            column,
        )
    elif args.command == "common":
        from gpu_backtest.workflows.common import process_data

        process_data(
            {
                "input_files": args.input,
                "key_columns": [p.strip() for p in args.keys.split(",") if p.strip()],
                "file_labels": args.labels.split(",") if args.labels else None,
                "output_file": args.output,
            }
        )
    else:
        from gpu_backtest.workflows.charts import write_charts

        print(write_charts(args.input, args.output))
