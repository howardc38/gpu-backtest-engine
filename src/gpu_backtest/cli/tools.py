"""CLI adapters for optional tools; they do not implement engine computation."""


def register(subparsers):
    check = subparsers.add_parser("gpu-check", help="Real-GPU numerical smoke checks")
    check.add_argument("--output")
    check.set_defaults(handler=execute)
    benchmark = subparsers.add_parser(
        "benchmark", help="Public RSI CPU/GPU performance measurement"
    )
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--bars", type=int, default=1024)
    benchmark.add_argument("--cpu-threads", type=int, default=8)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--billion", action="store_true")
    benchmark.add_argument("--billion-cpu", action="store_true")
    benchmark.set_defaults(handler=execute)
    from gpu_backtest_tools.runpod.launcher import DEFAULT_IMAGE

    runpod = subparsers.add_parser("runpod", help="Run a job on a leased GPU and clean it up")
    runpod.add_argument("--config")
    runpod.add_argument("--output-dir", required=True)
    runpod.add_argument("--ssh-key")
    runpod.add_argument("--plugin-dir")
    runpod.add_argument("--mode", choices=("run", "check", "benchmark"), default="run")
    runpod.add_argument("--image", default=DEFAULT_IMAGE)
    runpod.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    runpod.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE")
    runpod.add_argument("--max-seconds", type=int, default=1800)
    runpod.add_argument("--max-hourly-rate", type=float, default=1.0)
    runpod.add_argument("--keep-pod", action="store_true")
    runpod.add_argument("--dry-run", action="store_true")
    runpod.set_defaults(handler=execute)


def execute(args):
    if args.command == "gpu-check":
        from gpu_backtest_tools.checks.gpu import check_gpu

        check_gpu(args.output)
    elif args.command == "benchmark":
        from gpu_backtest_tools.benchmarks.runner import benchmark

        benchmark(
            args.output,
            bars=args.bars,
            cpu_threads=args.cpu_threads,
            repeats=args.repeats,
            billion=args.billion,
            billion_cpu=args.billion_cpu,
        )
    else:
        from gpu_backtest_tools.runpod.launcher import launch, terminate_on_signal

        with terminate_on_signal():
            launch(
                config_path=args.config,
                output_dir=args.output_dir,
                ssh_key=args.ssh_key,
                mode=args.mode,
                plugin_dir=args.plugin_dir,
                image=args.image,
                gpu=args.gpu,
                cloud=args.cloud,
                max_seconds=args.max_seconds,
                max_hourly_rate=args.max_hourly_rate,
                keep_pod=args.keep_pod,
                dry_run=args.dry_run,
            )
