"""
Base class for direction estimation decoder heads.

This module provides a BaseDirectionDecodeHead class, that is a direction estimation counterpart
of mmseg's BaseDecoderHead.
"""

from abc import ABCMeta, abstractmethod
from typing import Sequence

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from mmseg.models.builder import build_loss
from mmseg.models.utils import resize


class BaseDirectionDecodeHead(BaseModule, metaclass=ABCMeta):
    """
    Base class for direction estimation decoder heads.

    This class is a direction estimation counterpart of mmseg's BaseDecoderHead.

    From BaseDecoderHead:

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict segmentation results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_dir.
        dir_classes (int|Sequence[int]): Index or sequence of indexes of classes for which
            direction estimation is performed. If not provided, it assumes binary segmentation and
            positive class as linear. Default: None.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers. Default: dict(type='ReLU').
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resized to the same size as the first
                one and then concat together.
            'multiple_select': Multiple feature maps will be bundled into a list and passed into
                decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of the corresponding loss function which could be shown
            in the training log. If you want this loss item to be included in the backward graph,
            `loss_` must be the prefix of the name. Defaults to 'loss_dir'.
            e.g. dict(type='DirectionalLoss'),
            [dict(type='DirectionalLoss', loss_name='loss_dir'),
            dict(type='SmoothnessLoss', loss_name='loss_smt')]
            Default: dict(type='DirectionalLoss').
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate. Default: False.
        pre_norm_vectors (bool): Whether apply L2 normalization to directional vectors before
            vector filed resize. Default: False.
        post_norm_vectors (bool): Whether apply L2 normalization to directional vectors in output
            vector filed. Default: False.
        gt_scale_factor (float): Per class sematic segmentation map resize factor for loss
            calculation. Default: 1.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    DEFAULT_ACT_CFG: dict = dict(type='ReLU')
    DEFAULT_LOSS: dict = dict(type='DirectionalLoss')
    DEFAULT_INIT_CFG: dict = dict(
        type='Normal',
        std=0.01,
        override=dict(name='conv_dir'),
    )

    def __init__(
        self,
        in_channels: int,
        channels: int,
        *,
        dir_classes: Sequence[int]=None,
        dropout_ratio: float=0.1,
        conv_cfg: ConfigType=None,
        norm_cfg: ConfigType=None,
        act_cfg: ConfigType=None,
        in_index: int | Sequence[int]=-1,
        input_transform: str=None,
        loss_decode: ConfigType=None,
        ignore_index: int=255,
        sampler: ConfigType=None,
        align_corners: bool=False,
        pre_norm_vectors: bool=False,
        post_norm_vectors: bool=False,
        gt_scale_factor: float=1.,
        init_cfg: ConfigType=None,
    ) -> None:
        """
        Base class for direction estimation decoder heads.

        This class is a direction estimation counterpart of mmseg's BaseDecoderHead.

        From BaseDecoderHead:

        1. The ``init_weights`` method is used to initialize decode_head's
        model parameters. After segmentor initialization, ``init_weights``
        is triggered when ``segmentor.init_weights()`` is called externally.

        2. The ``loss`` method is used to calculate the loss of decode_head,
        which includes two steps: (1) the decode_head model performs forward
        propagation to obtain the feature maps (2) The ``loss_by_feat`` method
        is called based on the feature maps to calculate the loss.

        .. code:: text

        loss(): forward() -> loss_by_feat()

        3. The ``predict`` method is used to predict segmentation results,
        which includes two steps: (1) the decode_head model performs forward
        propagation to obtain the feature maps (2) The ``predict_by_feat`` method
        is called based on the feature maps to predict segmentation results
        including post-processing.

        .. code:: text

        predict(): forward() -> predict_by_feat()

        Args:
            in_channels (int|Sequence[int]): Input channels.
            channels (int): Channels after modules, before conv_dir.
            dir_classes (int|Sequence[int]): Index or sequence of indexes of classes for which
                direction estimation is performed. If not provided, it assumes binary segmentation and
                positive class as linear. Default: None.
            dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
            conv_cfg (dict|None): Config of conv layers. Default: None.
            norm_cfg (dict|None): Config of norm layers. Default: None.
            act_cfg (dict): Config of activation layers. Default: dict(type='ReLU').
            in_index (int|Sequence[int]): Input feature index. Default: -1
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resized to the same size as the first
                    one and then concat together.
                'multiple_select': Multiple feature maps will be bundled into a list and passed into
                    decode head.
                None: Only one select feature map is allowed.
                Default: None.
            loss_decode (dict | Sequence[dict]): Config of decode loss.
                The `loss_name` is property of the corresponding loss function which could be shown
                in the training log. If you want this loss item to be included in the backward graph,
                `loss_` must be the prefix of the name. Defaults to 'loss_dir'.
                e.g. dict(type='DirectionalLoss'),
                [dict(type='DirectionalLoss', loss_name='loss_dir'),
                dict(type='SmoothnessLoss', loss_name='loss_smt')]
                Default: dict(type='DirectionalLoss').
            sampler (dict|None): The config of segmentation map sampler.
                Default: None.
            align_corners (bool): align_corners argument of F.interpolate. Default: False.
            pre_norm_vectors (bool): Whether apply L2 normalization to directional vectors before
                vector filed resize. Default: False.
            post_norm_vectors (bool): Whether apply L2 normalization to directional vectors in output
                vector filed. Default: False.
            gt_scale_factor (float): Per class sematic segmentation map resize factor for loss
                calculation. Default: 1.
            init_cfg (dict or list[dict], optional): Initialization config dict.
        """
        super().__init__(init_cfg or self.DEFAULT_INIT_CFG)
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dir_classes = dir_classes or (1,)
        self.num_classes = len(self.dir_classes)
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg or self.DEFAULT_ACT_CFG
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.pre_norm_vectors = pre_norm_vectors
        self.post_norm_vectors = post_norm_vectors
        self.gt_scale_factor = gt_scale_factor

        # 2 vector components per class
        self.out_channels = 2 * self.num_classes

        loss_decode = loss_decode or self.DEFAULT_LOSS
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        self.conv_dir = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """
        Check and initialize input transforms.

        The in_channels, in_index and input_transform must match. Specifically, when
        input_transform is None, only a single feature map will be selected. So in_channels and
        in_index must be of type int.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resized to the same size as first
                    one and then concat together.
                'multiple_select': Multiple feature maps will be bundled into a list and passed
                    into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """
        Transforms inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def estimate_directions(self, feat: Tensor) -> Tensor:
        """Estimate per pixel directions."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        output = self.conv_dir(feat)
        n, c, h, w = output.shape
        output = output.reshape(n, self.num_classes, 2, h, w)

        if self.pre_norm_vectors:
            output = F.normalize(output, p=2, dim=2)
        return output

    def loss(self, inputs: tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """
        Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg data samples. It usually
                includes information such as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        dir_vectors = self.forward(inputs)
        losses = self.loss_by_feat(dir_vectors, batch_data_samples)
        return losses

    def predict(self, inputs: tuple[Tensor], batch_img_metas: list[dict],
                test_cfg: ConfigType) -> Tensor:
        """
        Forward function for prediction.

        Args:
            inputs (tuple[Tensor]): List of multi-level img features.
            batch_img_metas (list[dict]): List Image info where each dict may also contain: 'img_shape',
                'scale_factor', 'flip', 'img_path', 'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs direction vector field for each class.
        """
        dir_vectors = self.forward(inputs)

        return self.predict_by_feat(dir_vectors, batch_img_metas)

    def _transform_gt_sem_seg(self, gt_sem_seg: Tensor) -> Tensor:
        """
        Transforms gt_sem_seg maps into separate binary maps per class for which directions will
        be estimated.

        Args:
            gt_sem_seg (Tensor): Ground truth semantic segmentation map of shape (N, C, H, W).

        Returns:
            Tensor: Semantic segmentation map of shape (N, K, 1, H, W) where K number of linear
                classes.
        """
        class_maps = []
        for class_index in self.dir_classes:
            class_map = torch.where(gt_sem_seg == class_index, 1, 0)
            class_maps.append(class_map)
        return torch.stack(class_maps, dim=1).float()

    def _resize_per_class_gt_sem_seg(self, per_class_map: Tensor) -> Tensor:
        """
        Transforms gt_sem_seg maps into separate binary maps per class for which directions will
        be estimated.

        Args:
            per_class_map (Tensor): Per class ground truth semantic segmentation map of
                shape (N, K, 1, H, W).

        Returns:
            Tensor: Resized per class ground truth semantic segmentation map of shape (N, K, 1,
                gt_scale_factor * H, gt_scale_factor * W).
        """
        return resize(
            input=per_class_map.squeeze(2),
            scale_factor=self.gt_scale_factor,
            mode='bilinear',
            align_corners=self.align_corners
        ).unsqueeze(2)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        """Stacks sematic segmentation ground truth maps into one batch."""
        gt_semantic_segs: list[Tensor] = [data_sample.gt_sem_seg.data for data_sample in batch_data_samples]
        gt_sem_seg: Tensor = torch.stack(gt_semantic_segs, dim=0)

        per_class_gt_sem_seg: Tensor = self._transform_gt_sem_seg(gt_sem_seg)

        if self.gt_scale_factor != 1.0:
            per_class_gt_sem_seg: Tensor = self._resize_per_class_gt_sem_seg(per_class_gt_sem_seg)

        return per_class_gt_sem_seg

    def loss_by_feat(self, dir_vectors: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """
        Compute loss.

        Args:
            dir_vectors (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg data samples. It usually
                includes information such as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        n, k, c, h1, w1 = dir_vectors.shape
        h2, w2 = seg_label.shape[-2:]
        dir_vectors = resize(
            input=dir_vectors.view(n, k*2, h1, w1),
            size=(h2, w2),
            mode='bilinear',
            align_corners=self.align_corners).reshape(n, k, c, h2, w2)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(dir_vectors, seg_label)
        else:
            seg_weight = None

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    dir_vectors,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    dir_vectors,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        return loss

    def predict_by_feat(self, dir_vectors: Tensor,
                        batch_img_metas: list[dict]) -> Tensor:
        """
        Transforms a batch of output dir_vectors to the input shape.

        Args:
            dir_vectors (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g., image size,
                scaling factor, etc.

        Returns:
            Tensor: Outputs direction vector field for each class.
        """

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        n, k, c, h1, w1 = dir_vectors.shape
        h2, w2 = size
        dir_vectors = resize(
            input=dir_vectors.view(n, k*2, h1, w1),
            size=size,
            mode='bilinear',
            align_corners=self.align_corners).reshape(n, k, c, h2, w2)

        if self.post_norm_vectors:
            dir_vectors = F.normalize(dir_vectors, p=2, dim=2)

        return dir_vectors
