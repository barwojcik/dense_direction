"""
Module for training a deep learning model.

This module provides the main entry point for training a model.
"""

import argparse
import mmengine
from mmengine import Config, DictAction, RUNNERS
import mmcv
import mmseg
import dense_direction


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training a model"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", help="config file path")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )

    return parser.parse_args()


def main():
    """
    Main entry point of the application.

    This function initializes the configuration, builds and runs the runner based on the provided arguments and configuration file.
    """

    args: argparse.Namespace = parse_args()
    cfg: Config = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    runner = RUNNERS.build(cfg)
    runner.train()


if __name__ == "__main__":
    main()
