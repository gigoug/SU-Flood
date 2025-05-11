from .regression_head import *
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule
from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import AnchorHead
import torch.nn.functional as F
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
        #print(context.shape)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.embed_dim)

        # Final linear projection
        output = self.out_proj(context)
        if return_attn:
            return output, attn
        return output

class LineAttentionMap(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.w_cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim*2
        )
        self.h_cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            output_dim=output_dim*2
        )
        self.num_heads = num_heads
        self.linear_h=nn.Linear(num_heads,num_heads)
        self.linear_w = nn.Linear(num_heads, num_heads)
        # nn.init.zeros_(self.linear_w.weight)  # 初始化权重为 0
        # nn.init.zeros_(self.linear_w.bias)
        self.relu= nn.ReLU()

    def forward(self, x, return_attn=False,query=None): # x: BCHW
        b,c,h,w=x.shape
        x= x.permute(0, 2, 3, 1).reshape(-1,w,c)#(B*H//2,C,W)
        output_w, attn_w = self.w_cross_attn(query=x,key=x,value=x,return_attn=True) #(B*H,W,C),(B*H,head,W,W)
        x=x.reshape(b,h,w,c).permute(0,2,1,3).reshape(-1,h,c)
        output_h,attn_h=self.h_cross_attn(query=x,key=x,value=x,return_attn=True)#(B*W,H,C),(B*H,head,H,H)
        attn_w=attn_w.sum(dim=-2).reshape(b,h,self.num_heads,w).permute(0,2,1,3)#[B,head,H,W]
        attn_h=attn_h.sum(dim=-2).reshape(b,w,self.num_heads,h).permute(0,2,3,1)#[B,head,H,W]
        attn=self.linear_w(attn_w.permute(0,2,3,1))+self.linear_h(attn_h.permute(0,2,3,1))
        attn=self.relu(attn.permute(0,-1,1,2))
        x=x.reshape(b,c,h,w).reshape(b,self.num_heads,c//self.num_heads,h,w)
        # print(attn.shape)
        # print(x.shape)
        return (attn.unsqueeze(2)*x).reshape(b,c,h,w)



@HEADS.register_module()
class RAU_head(RegressionHead):
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
        super(RAU_head,self).__init__(
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
            **kwargs
        )
    def _init_layers(self):
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
            chn = self.in_channels*2 if i == 0 else self.feat_channels
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
        self.seg_attnMap=LineAttentionMap(embed_dim=int(self.feat_channels),num_heads=int(4),output_dim=int(self.feat_channels//2))
        self.retina_depth_reg=nn.Conv2d(int(self.feat_channels),self.num_base_priors,3,padding=1,bias=True)# version 1.0 mean
        #self.depth_cross_attn=AttentionPool3d(embed_dim=int(self.feat_channels),num_heads=int(4),output_dim=int(self.feat_channels//2))
    def forward(self, x,teacher_x=None,mask=None):
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
        water_feat=x[-1]
        water_feat=self.channel_fusion(water_feat)
        mask_attn=self.seg_attnMap(water_feat)
        water_feat=torch.cat((water_feat,mask_attn),dim=1)
        for water_conv in self.water_convs:
            water_feat = water_conv(water_feat)

        reg_depth=self.retina_depth_reg(water_feat)
        return 0,0,[reg_depth]
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

    def extract_feat(self, x):
        student_bbox_head_feat,cls_bboxs=multi_apply(self.extract_feats_single, x)
        cls=[]
        bbox=[]
        for cls_bbox in cls_bboxs:
            cls+=[cls_bbox[0]]
            bbox+=[cls_bbox[1]]
        return student_bbox_head_feat,(cls,bbox)
