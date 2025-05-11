import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mmdet_addition.models.backbone import *
from mmdet_addition.models.dense_heads import *
from mmcv.cnn.utils import constant_init, kaiming_init, normal_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmdet.models import build_loss
import os



@DISTILLER.register_module()
class DetectionDistiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False,
                 bbox=False,
                 line_loss=True,
                 regression=True,
                 backbone=False
                 ):

        super(DetectionDistiller, self).__init__()
        self.index=0
        self.line_loss=line_loss
        self.anns=[]#torch.load("/public/DATA/lxy/workspace/flood_assess/depth2lidar/val_anno.pth")
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.bbox=bbox
        self.regression=regression
        self.backbone=backbone
        if teacher_pretrained is not None:
            self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()
        for key, value in dict(self.teacher.named_children()).items():
                for param in value.parameters():
                    param.requires_grad = False
        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        # ckpt =torch.load('/home/lxy/.cache/torch/hub/checkpoints/resnest50-528c19ca.pth')
        # self.student.backbone.load_state_dict(ckpt,strict=True)
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                #cmt
                # if name.startswith("img_"):
                #     all_name.append((name[4:],v))
                # if name.startswith("backbone."):
                #     continue
                # else:
                all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            # print(state_dict.keys())
            load_state_dict(self.student, state_dict)
            self.init_weights()

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())

        def regitster_hooks(student_module, teacher_module):
            def hook_teacher_forward(module, input, output):
                self.register_buffer(teacher_module, output)

            def hook_student_forward(module, input, output):
                self.register_buffer(student_module, output)

            return hook_teacher_forward, hook_student_forward

        for item_loc in distill_cfg:

            student_module = 'student_' + item_loc.student_module.replace('.', '_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')

            self.register_buffer(student_module, None)
            self.register_buffer(teacher_module, None)

            hook_teacher_forward, hook_student_forward = regitster_hooks(student_module,teacher_module)

            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)

    def init_weights(self, pretrained=None):
        if pretrained is None:
            for m in self.student.bbox_head.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=0.01)
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')



    def get_bbox_loss(self,teacher_bbox,student_bbox,the=0.7,img_height=800,img_width=1344):
        def draw_line(heatmap, x1, y1, x2, y2, value=1.0):
            """在热图上绘制线段"""
            x = torch.linspace(x1, x2, steps=int(max(abs(x2 - x1), abs(y2 - y1)).item()) + 1)
            y = torch.linspace(y1, y2, steps=int(max(abs(x2 - x1), abs(y2 - y1)).item()) + 1)
            x = x.long().clamp(0, img_width - 1)
            y = y.long().clamp(0, img_height - 1)
            heatmap[y, x] = value  # 增加热度值

        def gaussian_blur(heatmap, kernel_size=50, sigma=10.0):
            """对热图应用高斯模糊"""
            # 创建高斯核
            x = torch.arange(kernel_size) - kernel_size // 2
            gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            kernel_1d = gauss / gauss.sum()
            kernel_2d = torch.outer(kernel_1d, kernel_1d)
            kernel_2d = kernel_2d.expand(1, 1, -1, -1)  # 扩展为4D
            heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # 扩展为 (1, 1, H, W)
            kernel_2d=kernel_2d#.cuda()
            heatmap = F.conv2d(heatmap, kernel_2d, padding=kernel_size // 2)
            return heatmap.squeeze()

        loss=torch.tensor(0.0,dtype=torch.float32,device='cuda')
        for t_bbox,s_bbox in zip(teacher_bbox,student_bbox):
            heatmap_t=torch.zeros((img_height, img_width), dtype=torch.float32)#.cuda()
            heatmap_s = torch.zeros((img_height, img_width), dtype=torch.float32)#.cuda()
            mask=t_bbox[0][:,-1]>the
            thr_t_bbox=t_bbox[0][mask]
            # mask=s_bbox[0][:,-1]>the
            thr_s_bbox=s_bbox[0][:10]
            thr_t_line = thr_t_bbox[:, [0, 3, 2, 3]]
            thr_s_line = thr_s_bbox[:, [0, 3, 2, 3]]
            for i in range(thr_t_line.shape[0]):
                draw_line(heatmap_t,thr_t_line[i,0],thr_t_line[i,1],thr_t_line[i,2],thr_t_line[i,3])
            for i in range(thr_s_line.shape[0]):
                draw_line(heatmap_s,thr_s_line[i,0],thr_s_line[i,1],thr_s_line[i,2],thr_s_line[i,3])
            heatmap_t = gaussian_blur(heatmap_t)
            heatmap_s = gaussian_blur(heatmap_s)
            single_loss=(heatmap_t-heatmap_s)**2*0.5
            single_loss=(single_loss/(img_height*img_width)).sum()
            loss=single_loss+loss

        return loss

    def forward_train(self,img, img_metas, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
        #print(img_metas)
        # if 'mask' in kwargs.keys():
        #     mask=kwargs["mask"].permute(0,-1,1,2)
        #     mask=mask.mean(dim=1)
        #     mask[mask!=0]=1
        #     mask=tuple(mask.clone() for _ in range(5))
        # else:
        #     mask=None
        # print("forward")
        # self.index=self.index+1
        # if self.index%10==0:
        #     torch.save(self.student,f"./work_dirs/depth_try/{self.index}.pth")
        student_loss={}
        student_feat=self.student.extract_feat(img.permute(0,-1,1,2))
        student_bbox_head_feat,cls_bbox=self.student.bbox_head.extract_feat(student_feat)
        # water_feat=torch.cat((student_bbox_head_feat[0][-1],student_bbox_head_feat[1][-1]),dim=1)

        t_img=kwargs['t_img'].permute(0,-1,1,2)
        with torch.no_grad():
            self.teacher.eval()
            teacher_feat = self.teacher.extract_feat(t_img)
            # self.teacher.predict_by_feat(teacher_feat[0],teacher_feat[1])
            if self.line_loss:
                teacher_bbox=self.teacher.bbox_head.simple_test(teacher_feat,img_metas,rescale=True)
                student_bbox=self.teacher.bbox_head.get_bboxes(*cls_bbox,img_metas=img_metas,rescale=False)
            else:
                del cls_bbox
        if self.line_loss:
            student_loss["bbox_KD"]=self.get_bbox_loss(teacher_bbox,student_bbox).cuda()

        _,_,student_depth=self.student.bbox_head([student_bbox_head_feat[-1]],None)
        # print(kwargs['gt_labels'])
        reg_loss,rank_loss=self.student.bbox_head.regrssion_loss(student_depth,kwargs['gt_labels'])
        depth_loss=reg_loss.item()+rank_loss.item()
        # for img_metas, pred,gt in zip(img_metas,student_depth[-1].mean(dim=(-1, -2, -3)),kwargs['gt_labels']):
        #     print(f"{img_metas['filename']}: {pred.shape},{pred}, {gt}")
        # print(img_metas)
        # print(f"{kwargs['filename']},{student_depth},{kwargs['gt_labels']},{reg_loss},{rank_loss}")

        if self.bbox:
                buffer_dict = dict(self.named_buffers())
                for item_loc in self.distill_cfg:

                    student_module = 'student_' + item_loc.student_module.replace('.', '_')
                    teacher_module = 'teacher_' + item_loc.teacher_module.replace('.', '_')

                    student_feat = buffer_dict[student_module]
                    teacher_feat = buffer_dict[teacher_module]

                    for item_loss in item_loc.methods:
                        loss_name = item_loss.name
                        distill_loss=self.distill_losses[loss_name](student_feat, teacher_feat, img_metas,kwargs['mask'])
                        #zero_mask=(kwargs['gt_labels']!=0.0)

                        student_loss[loss_name] = (distill_loss/distill_loss.item()*depth_loss)#.cuda()
        if self.regression:
            # print(".")
            student_loss["loss_reg"]=reg_loss#.cuda()
            student_loss["loss_rank"]=rank_loss#.cuda()


        for name, param in self.student.named_parameters():
            if torch.isnan(param).any():
                print(f"Parameter {name} contains NaN")
                break
            if torch.isinf(param).any():
                print(f"Parameter {name} contains Inf")
                break
        # for name, param in self.student.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter {name}:")
        #         print(f"  Shape: {param.shape}, Strides: {param.stride()}")
        #         print(f"  Gradient Shape: {param.grad.shape}, Strides: {param.grad.stride()}")
        # print("================")
        # for name, param in self.teacher.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter {name}:")
        #         print(f"  Shape: {param.shape}, Strides: {param.stride()}")
        #         print(f"  Gradient Shape: {param.grad.shape}, Strides: {param.grad.stride()}")

        return student_loss

    def visualize_attention(self,tensor, image, filename, pred,gt, output_dir):
        self.index+=1
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 将特征张量从形状 [1, 256, 96, 168] 中的注意力图处理为平均特征
        # 首先移除batch维度(即1)，然后对特征维度（256）求均值，得到 [96, 168] 的注意力图
        attention_map = torch.mean(tensor.cpu().squeeze(0), dim=0).detach().numpy()

        # 归一化注意力图到 [0, 255] 范围，并调整为与图像匹配的尺寸
        attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map)) * 255
        attention_map = cv2.resize(attention_map, (image.shape[2], image.shape[1]))  # 调整到 [1600, 900]

        # 将注意力图转换为伪彩色图像
        attention_colormap = cv2.applyColorMap(attention_map.astype(np.uint8), cv2.COLORMAP_JET)

        # 转换原始图像为 NumPy 数组，调整为 OpenCV 格式（[900, 1600, 3]）
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np)) * 255  # 归一化
        image_np = image_np.astype(np.uint8)

        # 叠加注意力图到原始图像上，采用透明度（alpha）控制可视化效果
        overlay = cv2.addWeighted(image_np, 1.0, attention_colormap, 0.9, 0)

        # 在图像上添加 gt 和 pred 标签
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # text = f'pred={pred.item()}\ngt={gt}'
        # cv2.putText(overlay, text, (50, 50), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        # 保存结果图片
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, overlay)
        print(f"Saved: {output_path}")
        # attn=torch.load("attn_visual.pth")
        # print(attn.shape)
        # attn_w=attn[:,:4,...]
        # attn_h=attn[:,4:,...]
        # self.visualize_attention(attn_w, img[0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], torch.tensor(0), 0,"./RAU_attn_show")
        # self.visualize_attention(attn_h, img[0], str(self.index) + "_" + img_metas[0]['filename'].split('/')[-1], torch.tensor(0), 0,
        #                          "./RAU_attn_show")
        # self.visualize_attention(attn, img[0], str(self.index) + "_" + img_metas[0]['filename'].split('/')[-1], torch.tensor(0), 0,
        #                          "./RAU_attn_show")
        # attn=torch.load("attn_visual_2.pth")
        # print(attn.shape)
        # self.visualize_attention(attn, img[0], str(self.index) + "_" + img_metas[0]['filename'].split('/')[-1], torch.tensor(0), 0,
        #                          "./RAU_attn_show")
    def Dualvisualize_attention(self, tensor, image, tensor2, image2, filename, pred, gt, output_dir):
        self.index += 1

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 处理第一个tensor和image
        def process_tensor_image(tensor, image):
            # 将特征张量处理为平均特征
            attention_map = torch.mean(tensor.cpu().squeeze(0), dim=0).detach().numpy()

            # 归一化注意力图到 [0, 255] 范围，并调整为与图像匹配的尺寸
            attention_map = (attention_map - np.min(attention_map)) / (
                        np.max(attention_map) - np.min(attention_map)) * 255
            attention_map = cv2.resize(attention_map, (image.shape[2], image.shape[1]))

            # 将注意力图转换为伪彩色图像
            attention_colormap = cv2.applyColorMap(attention_map.astype(np.uint8), cv2.COLORMAP_JET)

            # 转换原始图像为 NumPy 数组，调整为 OpenCV 格式（[H, W, C]）
            image_np = image.permute(1, 2, 0).detach().cpu().numpy()
            image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np)) * 255  # 归一化
            image_np = image_np.astype(np.uint8)

            # 叠加注意力图到原始图像上，采用透明度（alpha）控制可视化效果
            overlay = cv2.addWeighted(image_np, 1.0, attention_colormap, 0.5, 0)

            return overlay

        # 处理第一个图像和tensor
        overlay1 = process_tensor_image(tensor, image)

        # 处理第二个图像和tensor
        overlay2 = process_tensor_image(tensor2, image2)

        # 将两张结果图像水平拼接
        result_image = cv2.hconcat([overlay1, overlay2])

        # 在图像上添加 gt 和 pred 标签（这里只添加到第一张图上，或者你可以分别添加）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'pred={pred.item()}\ngt={gt}'
        cv2.putText(overlay1, text, (50, 50), font, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        # 保存结果图片
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, result_image)
        print(f"Saved: {output_path}")

    def threshold(self,pred,gt,the):
        if pred.item()>gt:
            return pred-the<gt
        if pred.item()<gt:
            return pred+the>gt
        return True
    def simple_test(self, img, img_metas, **kwargs):#depth
        self.student.eval()
        with torch.no_grad():
            for name, param in self.student.named_parameters():
                if torch.isnan(param).any():
                    print(f"Parameter {name} contains NaN")
                if torch.isinf(param).any():
                    print(f"Parameter {name} contains Inf")

            # print(img)
            feat=self.student.extract_feat(img)
            # print(feat)
            student_bbox_head_feat, cls_bbox = self.student.bbox_head.extract_feat(feat)
            # print(student_bbox_head_feat[-1])
            result = self.student.bbox_head([student_bbox_head_feat[-1]],None)
            # print(result)
            pred=result[2][-1].mean(dim=(-1,-2,-3))
        #self.student.train()

        return pred
    def simple_test_3(self, img, img_metas, **kwargs):#backbone
        if not self.backbone:
            return self.simple_test_3(img, img_metas, **kwargs)
        if 'mask' in kwargs.keys():
            mask=kwargs["mask"][0]#.permute(0,-1,1,2)
            mask=mask.mean(dim=1,keepdim=True)
            mask[mask!=0]=1
        else:
            mask=None
        feat=self.student.extract_feat(img)
        t_img = kwargs['t_img']
        teacher_feat=self.teacher.extract_feat(t_img[0])
        res=[]
        for tf,sf in zip(teacher_feat,feat):
            res+=[(tf-sf).mean(dim=(-1,-2,-3))]
        #print(res)
        diff=[torch.cat(res,dim=0)]#(teacher_feat-feat).mean()
        self.Dualvisualize_attention(feat[0], img[0], teacher_feat[0], t_img[0][0],
                                     str(self.index) + "_" + img_metas[0]['filename'].split('/')[-1], torch.tensor(0),
                                     self.anns[self.index], "./result_show/tmp_itr1000/")
        # print(len(teacher_feat))

        # result = self.student.bbox_head(feat,mask)
        # res=[]
        # for r in result[2]:
        #     res+=[r.mean(dim=(-1,-2,-3))]
        # pred=sum(res)/len(res)

        return diff

    def simple_test_2(self, img, img_metas, **kwargs):
        feat=self.student.extract_feat(img)
        result = self.student.bbox_head(feat,None)
        pred=result[2][-1].mean(dim=(-1,-2,-3))
        gt=kwargs['gt_labels'][0]
        # t_img = kwargs['t_img']
        # teacher_feat=self.teacher.extract_feat(t_img[0])

        if self.threshold(pred,self.anns[self.index],
                          0.02):
            self.visualize_attention(feat[0], img[0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], pred, gt,"./result_show/single_attention_show_thred=2")
        elif self.threshold(pred,gt,0.05):
            self.visualize_attention(feat[0], img[0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], pred, gt,"./result_show/single_attention_show_thred=5")
        elif self.threshold(pred,gt,0.10):
            self.visualize_attention(feat[0], img[0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], pred, gt,"./result_show/single_attention_show_thred=10")
        elif self.threshold(pred,gt,0.20):
            self.visualize_attention(feat[0], img[0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], pred, gt,"./result_show/single_attention_show_thred=20")
        else:
            self.visualize_attention(feat[0], img[0],str(self.index) + "_" + img_metas[0]['filename'].split('/')[-1],
                                     pred, gt, "./result_show/single_attention_show_thred>20")
        return pred
    def simple_test_1(self, img, img_metas, **kwargs):
        feat=self.student.extract_feat(img)
        result = self.student.bbox_head(feat,None)
        pred=result[2][-1].mean(dim=(-1,-2,-3))

        t_img = kwargs['t_img']
        teacher_feat=self.teacher.extract_feat(t_img[0])

        if self.threshold(pred,self.anns[self.index],0.02):
            self.Dualvisualize_attention(feat[0], img[0], teacher_feat[0],t_img[0][0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], pred, self.anns[self.index],"./result_show/attention_show_thred=2")
        elif self.threshold(pred,self.anns[self.index],0.05):
            self.Dualvisualize_attention(feat[0], img[0], teacher_feat[0],t_img[0][0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], pred, self.anns[self.index],"./result_show/attention_show_thred=5")
        elif self.threshold(pred,self.anns[self.index],0.10):
            self.Dualvisualize_attention(feat[0], img[0], teacher_feat[0],t_img[0][0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], pred, self.anns[self.index],"./result_show/attention_show_thred=10")
        elif self.threshold(pred,self.anns[self.index],0.20):
            self.Dualvisualize_attention(feat[0], img[0], teacher_feat[0],t_img[0][0],str(self.index)+"_"+img_metas[0]['filename'].split('/')[-1], pred, self.anns[self.index],"./result_show/attention_show_thred=20")
        else:
            self.Dualvisualize_attention(feat[0], img[0],teacher_feat[0],t_img[0][0], str(self.index) + "_" + img_metas[0]['filename'].split('/')[-1],
                                     pred, self.anns[self.index], "./result_show/attention_show_thred>20")
        return pred
        #return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


