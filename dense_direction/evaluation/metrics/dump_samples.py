"""
DumpSamples metric.

This module provides a DumpSample class that dumps model predictions to a directory for offline evaluation.
"""

import logging

from typing import Any, Sequence, Optional

from mmengine.evaluator import BaseMetric
from mmengine.evaluator.metric import _to_cpu
from mmengine.fileio import dump
from mmengine.logging import print_log
from mmseg.registry import METRICS


@METRICS.register_module()
class DumpSamples(BaseMetric):
    """
    Dump model predictions to a directory for offline evaluation.

    Dumps model predictions to a directory as a separate pickle file per data sample.

    Arguments:
        output_directory (str): Output directory.
        fields (Optional[Sequence[str]]): List of fields in a data sample to save if not provided
            dumps all fields. Default: None.
        collect_device (str): Device name used for collecting results from different ranks during
            distributed training. Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric names to disambiguate
            homonymous metrics of different evaluators. If prefix is not provided in the argument,
            self.default_prefix will be used instead. Default: None
        collect_dir: (str, optional): Synchronize directory for collecting data from different
            ranks. This argument should only be configured when ``collect_device`` is 'cpu'.
            Defaults to None.
    """

    default_prefix = "dump_samples"

    def __init__(
        self,
        output_directory: str,
        fields: Optional[Sequence[str]] = None,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        collect_dir: Optional[str] = None,
    ) -> None:
        """
        Dump model predictions to a directory for offline evaluation.

        Dumps model predictions to a directory as a separate pickle file per data sample.

        Arguments:
            output_directory (str): Output directory.
            fields (Optional[Sequence[str]]): List of fields in a data sample to save if not
                provided dumps all fields. Default: None.
            collect_device (str): Device name used for collecting results from different ranks
                during distributed training. Must be 'cpu' or 'gpu'. Defaults to 'cpu'.
            prefix (str, optional): The prefix that will be added in the metric names to
                disambiguate homonymous metrics of different evaluators. If prefix is not provided
                in the argument, self.default_prefix will be used instead. Default: None
            collect_dir: (str, optional): Synchronize directory for collecting data from different
                ranks. This argument should only be configured when ``collect_device`` is 'cpu'.
                Defaults to None.
        """
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
            collect_dir=collect_dir,
        )
        self.output_directory = output_directory
        self.fields = fields

    def compute_metrics(self, results: list) -> dict:
        """
        Computes the number of saved files.

        Args:
            results (list): The list of paths to saved files.

        Returns:
            Dict[str, int]: The number of saved data samples.
        """
        return {"samples": len(results)}

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """
        Dumps results to a pickle file for offline evaluation.

        Process data samples one by one and save them in the output directory.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        for data_sample in data_samples:
            output_path = self.output_directory + f"/{str(len(self.results)+1)}.pkl"

            if self.fields is not None:
                data_sample = {k: v for k, v in data_sample.items() if k in self.fields}

                if not len(data_sample) == len(self.fields):
                    print_log(
                        "Missing fields in data sample: "
                        f"{[field for field in self.fields if field not in data_sample.keys()]}",
                        logger="current",
                        level=logging.WARNING,
                    )

            if not data_sample:
                print_log(
                    "No data to save, got an empty data sample in DumpSamples. "
                    "Check configuration.",
                    logger="current",
                    level=logging.WARNING,
                )
                continue

            dump(_to_cpu(data_sample), output_path)
            self.results.append(output_path)
