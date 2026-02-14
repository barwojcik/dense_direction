"""
Script for training/testing models based on provided config.

This script provides a basic structure for training and testing a model using the provided
configuration file. It includes functions to parse command-line arguments, initialize the
configuration, build and run the runner, and save the trained model weights to the specified
directory.

For more information about the available command-line arguments, please refer to the MMEngine
documentation.
"""

import argparse

from mmengine import RUNNERS, Config, DictAction

import dense_direction  # noqa: F401


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Train/test a model")
    parser.add_argument(
        "--phase",
        default="both",
        type=str.lower,
        choices=["train", "test", "both"],
        help=(
            "run training, testing or both, one after another. It defaults to both, so first "
            "model will be trained then tested."
        ),
    )
    parser.add_argument("--config", help="config file path")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help=(
            "override some settings in the used config, the key-value pair  in xxx=yyy format "
            "will be merged into config file. If the value to be overwritten is a list, it "
            'should be like key="[a,b]" or key=a,b. It also allows nested list/tuple values, '
            'e.g. key="[(a,b),(c,d)]". Note that the quotation marks are necessary and that no '
            "white space  is allowed."
        ),
    )

    return parser.parse_args()


def main():
    """
    Main entry point of the training/testing loop.

    This function initializes the configuration, builds and runs the runner based on the provided
    arguments and configuration file.
    """

    args: argparse.Namespace = parse_args()
    cfg: Config = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    runner = RUNNERS.build(cfg)

    if args.phase in ["train", "both"]:
        runner.train()

    if args.phase in ["test", "both"]:
        runner.test()


if __name__ == "__main__":
    main()
