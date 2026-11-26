import argparse
from pathlib import Path

from tensorboard import program


def main():
    parser = argparse.ArgumentParser(description="View live Hypergrid training metrics.")
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    tensorboard = program.TensorBoard()
    tensorboard.configure(argv=[
        "tensorboard", "--logdir", str(args.log_dir.resolve()),
        "--host", "127.0.0.1", "--port", str(args.port), "--reload_interval", "5",
    ])
    tensorboard.main()


if __name__ == "__main__":
    main()
