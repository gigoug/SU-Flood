# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import AnchorHead
import torch.nn.functional as F
@HEADS.register_module()
class RegressionHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=8,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 loss_depth=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        # self.act_cfg={'type': ' PReLU'},
        self.norm_cfg = norm_cfg
        super(RegressionHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        self.MSE_loss=build_loss(dict(
                     type='MSELoss',reduction='sum', loss_weight=1.0))

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.water_convs=nn.ModuleList()
        in_channels = self.in_channels
        for i in range(self.stacked_convs):
            self.cls_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    act_cfg=dict(type='LeakyReLU'),
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    act_cfg=dict(type='LeakyReLU'),
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.cls_out_channels = self.cls_out_channels - 1
        self.retina_cls = nn.Conv2d(
            in_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            in_channels, self.num_base_priors*4, 3, padding=1)

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.water_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    bias=False,
                    act_cfg= dict(type='LeakyReLU'),
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.channel_fusion=nn.Conv2d(self.feat_channels*2,self.feat_channels,1,padding=0,bias=True)
        self.mask_conv_up=nn.Conv2d(self.feat_channels,self.feat_channels//2,3,padding=1,bias=True)
        self.mask_conv_down=nn.Conv2d(self.feat_channels,self.feat_channels//2,3, padding=1,bias=True)
        self.retina_depth_reg=nn.Conv2d(int(self.feat_channels),self.num_base_priors,3,padding=1,bias=True)# version 1.0 mean
        self.dropout=nn.Dropout2d(p=0.1)
        # self.depth_cross_attn=AttentionPool3d(embed_dim=int(self.feat_channels),num_heads=int(4),output_dim=int(self.feat_channels//2))
    def forward(self, x,mask=None):
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        # print(x[-1].shape)
        #return self.water_level(x[-1].reshape(x[-1].shape[0],-1))
        if mask is not None:
            return multi_apply(self.forward_single, x,mask)
        else:
            return multi_apply(self.forward_single, x)
    def resize_mask(self,mask,shape):
        ratio=int(mask.shape[1]/shape[2]),int(mask.shape[2]/shape[3])
        avg=nn.AvgPool2d(ratio,stride=ratio,ceil_mode=False)
        res=avg(mask.unsqueeze(1))
        return res
    def forward_single(self, water_feat,mask=None):
        """Forward feature of a single scale level.


        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        # cls_feat = x
        # reg_feat = x
        # for cls_conv in self.cls_convs:
        #     cls_feat = cls_conv(cls_feat)
        # for reg_conv in self.reg_convs:
        #     reg_feat = reg_conv(reg_feat)
        # cls_score = self.retina_cls(cls_feat)
        # bbox_pred = self.retina_reg(reg_feat)
        # water_feat=torch.cat((cls_feat,reg_feat),dim=1)
        # self.dropout(water_feat)
        # print(water_feat.shape)
        water_feat=self.channel_fusion(water_feat)
        for water_conv in self.water_convs:
            water_feat = water_conv(water_feat)
        # self.dropout(water_feat)
        reg_depth=self.retina_depth_reg(water_feat)
        return 0,0,reg_depth

    def extract_feats_single(self,x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return torch.cat((cls_feat,reg_feat),dim=1),(cls_score, bbox_pred)

    # def extract_feat(self, x):
    #     student_bbox_head_feat,cls_bboxs=multi_apply(self.extract_feats_single, x)
    #     cls=[]
    #     bbox=[]
    #     for cls_bbox in cls_bboxs:
    #         cls+=[cls_bbox[0]]
    #         bbox+=[cls_bbox[1]]
    #     return student_bbox_head_feat,(cls,bbox)
    def extract_feat(self, x):
        student_bbox_head_feat, cls_bboxs = multi_apply(self.extract_feats_single, x)
        cls = [cls_bbox[0] for cls_bbox in cls_bboxs] # 假设 cls_bbox[0] 是张量
        bbox = [cls_bbox[1] for cls_bbox in cls_bboxs]  # 假设 cls_bbox[1] 是张量
        return student_bbox_head_feat, (cls, bbox)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        pass

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        pass
    def depth2level(self,tensor):
        #tensor [5,1]
        res=[]
        for idx in range(tensor.shape[0]):
            depth=tensor[idx,0]
            if depth==0:
                res+=[0]
            elif depth<=0.01:
                res+=[1]
            elif depth<=0.1:
                res+=[2]
            elif depth<=0.2125:
                res+=[3]
            elif depth<=0.425:
                res+=[4]
            elif depth<=0.6375:
                res+=[5]
            elif depth<=0.85:
                res+=[6]
            elif depth<=1.0625:
                res+=[7]
            elif depth<=1.275:
                res+=[8]
            elif depth<=1.4875:
                res+=[9]
            else:
                res+=[10]
        res=torch.tensor(res).reshape(-1,1)
        return res
    #tensor([[0.]], device='cuda:1', grad_fn=<ReluBackward0>)
# tensor(1.0000e-08, device='cuda:1', grad_fn=<AddBackward0>)
# tensor([[0.0045]], device='cuda:1', grad_fn=<ReshapeAliasBackward0>)
# tensor([[0.]], device='cuda:0', grad_fn=<ReluBackward0>)
# tensor(1.0000e-08, device='cuda:0', grad_fn=<AddBackward0>)
# tensor([[0.3053]], device='cuda:0', grad_fn=<ReshapeAliasBackward0>)

    # def depth2level(self,tensor):
    #     # tensor [5,1]
    #     res = []
    #     for idx in range(tensor.shape[0]):
    #         depth = tensor[idx,0]
    #         if depth==0:
    #             res+=[0]
    #         elif depth <= 0.03:
    #             res += [1]
    #         elif depth <= 0.15:
    #             res += [2]
    #         elif depth <= 0.27:
    #             res += [3]
    #         elif depth <= 0.4:
    #             res += [4]
    #         elif depth <= 0.6:
    #             res += [5]
    #         else:
    #             res += [6]
    #     res=torch.tensor(res).reshape(-1,1)
    #     return res
    def rank_loss(self,y_pred, y):
        y_pred=y_pred.reshape(-1,1)#.cuda()
        y=y.reshape(-1,1).cuda()
        # print(y_pred)
        # print(y)
        ranking_loss = torch.nn.functional.relu(
            (y_pred - y_pred.t()) * torch.sign((y.t() - y))
        )
        # print( (y_pred - y_pred.t()),torch.sign((y.t() - y)))
        scale = (1e-8)+torch.max(ranking_loss)
        # print(torch.sum(ranking_loss),scale)

        if torch.isnan(torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale):
            print(ranking_loss)
            print(scale)
            print(y_pred)
        del y
        if y_pred.shape[0]==0 or y_pred.shape[0]-1==0:
            return (
                torch.sum(ranking_loss) / scale
        ).float()
        return (
                torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
        ).float()
    def regrssion_loss(self,reg_depth,gt_labels,):
        # reg_depth = reg_depth.mean(dim=(-1, -2, -3))
        # loss=((reg_depth - gt_labels) ** 2) * torch.max((gt_labels - 0.15) ** 2, torch.tensor((0.2 - 0.15) ** 2)) * 1e3
        # loss=loss.sum(dim=0)
        # return loss
        loss=[]
        MSE_loss=[]
        rank_loss=[]
        # print(reg_depth[0])
        depth = reg_depth[-1].mean(dim=(-1, -2, -3))
        return self.MSE_loss(depth.reshape(-1,1).to(torch.float32),gt_labels.reshape(-1,1).to(torch.float32)),5*self.rank_loss(depth.reshape(-1,1).to(torch.float32),self.depth2level(gt_labels.reshape(-1,1).to(torch.float32)))
