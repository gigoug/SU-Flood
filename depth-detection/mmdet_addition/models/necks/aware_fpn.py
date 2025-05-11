# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models import NECKS
import torch
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, query_dim, kv_dim, num_heads, output_dim=None):
        super(MultiHeadCrossAttention, self).__init__()
        #assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.output_dim = output_dim if output_dim else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(query_dim, embed_dim,bias=False)
        self.k_proj = nn.Linear(kv_dim, embed_dim,bias=False)
        self.v_proj = nn.Linear(kv_dim, embed_dim,bias=False)
        self.out_proj = nn.Linear(embed_dim, output_dim,bias=False)

    def forward(self, query, key, value, mask=None, return_attn=False):
        batch_size = query.size(0)

        # Linear projections
        q = self.q_proj(query) # NLC
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape and transpose for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)


        # Combine heads
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_proj(context)
        if return_attn:
            return output, attn
        return output

class ColumnAttentionMap(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.w_cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim
        )
        self.num_heads = num_heads

    def forward(self, x, return_attn=False,query=None): # x: BCHW
        b,c,h,w=x.shape
        # print(x.shape)
        x= x.permute(0, 2, 3, 1).reshape(-1,w,c)#(B*H,C,W)
        # print(x.shape)
        output, attn_w = self.w_cross_attn(query=x,key=x,value=x,return_attn=True) #(B*H,C,W)
        x=x-output
        x=x.reshape(b,h,w,c).permute(0,3,1,2)
        # print(x.shape)
        return x
@NECKS.register_module()
class AWARE_FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg = None,
        norm_cfg = None,
        act_cfg = None,
        upsample_cfg = dict(mode='nearest'),
        init_cfg = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.column_attns=nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            column_attn=ColumnAttentionMap(
                out_channels,
                4,
                out_channels
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.column_attns.append(column_attn)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.extra_column_attns=nn.ModuleList()
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                column_attn = ColumnAttentionMap(
                    out_channels,
                    4,
                    out_channels
                )
                self.fpn_convs.append(extra_fpn_conv)
                self.extra_column_attns.append(column_attn)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals = [
            column_attn(laterals[i])
            for i, column_attn in enumerate(self.column_attns)
        ]


        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](self.extra_column_attns[i-used_backbone_levels - 1](F.relu(outs[-1]))))
                    else:
                        outs.append(self.fpn_convs[i](self.extra_column_attns[i-used_backbone_levels - 1](outs[-1])))
        return tuple(outs)
