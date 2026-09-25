import argparse
import os
from pathlib import Path

from curvspec.config import load_config


def _common_arguments(parser):
    parser.add_argument("--dataset", "-d", choices=["tvr", "act", "cha"], default="tvr")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--num-workers", type=int)


def build_parser():
    parser = argparse.ArgumentParser(prog="curvspec")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    _common_arguments(train_parser)
    train_parser.add_argument("--resume")

    evaluate_parser = commands.add_parser("evaluate")
    _common_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if "," in args.gpu:
        raise ValueError("Use one GPU per process, for example: --gpu 0")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    project_root = Path(__file__).resolve().parents[2]
    config = load_config(args.dataset, project_root, args.data_root, args.output_root)
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    from curvspec.runtime.engine import evaluate, train

    if args.command == "train":
        train(config, args.resume)
    else:
        evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
