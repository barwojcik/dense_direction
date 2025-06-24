"""
Ottawa-Dataset extraction script.
"""

import argparse
from zipfile import ZipFile, ZipInfo


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Ottawa-Dataset extractor",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        help="path to the Ottawa-Dataset.zip file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory",
    )

    return parser.parse_args()


def main():
    """
    Main entry point of the extraction script.
    """

    args: argparse.Namespace = parse_args()

    zip_path: str = args.zip_path
    output_dir: str = args.output_dir

    with ZipFile(zip_path, 'r') as zip_data:
        for idx in range(1, 21):
            image_path: str = f'Ottawa-Dataset/{idx}/Ottawa-{idx}.tif'
            image_info: ZipInfo = zip_data.getinfo(image_path)
            image_info.filename = f'Ottawa-Dataset/images/Ottawa-{idx}.tif'
            print(zip_data.extract(image_info, path=output_dir))

            mask_path: str = f'Ottawa-Dataset/{idx}/segmentation.png'
            mask_info: ZipInfo = zip_data.getinfo(mask_path)
            mask_info.filename = f'Ottawa-Dataset/masks/Ottawa-{idx}.png'
            print(zip_data.extract(mask_info, path=output_dir))


if __name__ == "__main__":
    main()
