from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from PIL import Image
import os.path as osp
from pyquaternion import Quaternion
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
import os
from copy import deepcopy

class Depth:
    def __init__(self, x):
        self.points = x.transpose(1,0).numpy()

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]


    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])
class NuscDepth:
    '''
    depth:[w,h,1]
    '''
    def __init__(self,nusc,mask=None):
        self.nusc=nusc
        self.mask=None

    def pc_to_depth(self, pc, pointsensor, cam, min_dist=1.0, im=[1600, 900]):
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
        depths = pc.points[2, :]
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im[1] - 1)
        depths = depths[mask]
        return depths,points, mask
    def pc_to_depth_and_get_uv(self, pc, pointsensor, cam, min_dist=1.0, im=[1600, 900]):
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
        depths = pc.points[2, :]
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im[1] - 1)
        depths = depths[mask]
        return depths, points, mask, pc.points[0,:][mask], pc.points[1,:][mask]

    def choose_pc_with_flood_mask(self,pointsensor,cam,pc,im,flood_mask,min_dist=1,visual=True):

        depths,points,mask=self.pc_to_depth(deepcopy(pc), pointsensor, cam)
        points=points[:,mask]
        flood_mask=np.array(flood_mask)[...,0].transpose(1,0)
        sec_mask=flood_mask[points[0,:].astype(int),points[1,:].astype(int)]==0
        randn_mask=torch.randn(sec_mask.shape).numpy()
        randn_mask=randn_mask>0.7
        sec_mask=np.logical_or(sec_mask,randn_mask)
        if visual:
            self.render_pointcloud_in_image(deepcopy(pc),im,pointsensor,cam,sample_token="origin_pc",dot_size=3)
        pc.points=pc.points[:,mask][:,sec_mask]
        self.write_obj(pc,"flood_pc")
        if visual:
            im=Image.open("/public/DATA/lxy/workspace/flood_assess/xl-inpainting/pc_demo_result.jpg")
            self.render_pointcloud_in_image(deepcopy(pc),im,pointsensor,cam,sample_token="selected_pc",dot_size=3)


    def map_pointcloud_to_image(self,
                                pc,
                                im,
                                pointsensor: str,
                                cam
    ) :
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidar intensity instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """
        coloring,points,mask=self.pc_to_depth(pc, pointsensor, cam)
        return points[:,mask], mask, im

    def render_pointcloud_in_image(self,
                                   pc,
                                   im,
                                   pointsensor,
                                   camera,
                                   sample_token: str,
                                   dot_size: int = 5,
                                   camera_channel: str = 'CAM_FRONT',
                                   out_path: str = None,
                                   ax: Axes = None,
                                   verbose: bool = True,
                                   lidarseg_preds_bin_path: str = None,
                                   show_cluster=False,
                                   labels=None,
                                   height_labels=None,
                                   ):
        """
        Scatter-plots a pointcloud on top of image.
        :param sample_token: Sample token.
        :param dot_size: Scatter plot dot size.
        :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param out_path: Optional path to save the rendered figure to disk.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        :param ax: Axes onto which to render.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param verbose: Whether to display the image in a window.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        points, mask, im = self.map_pointcloud_to_image(deepcopy(pc),im,pointsensor, camera)
        coloring=pc.points[2,:][mask]
        # Init axes.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 16))
            if lidarseg_preds_bin_path:
                fig.canvas.manager.set_window_title(sample_token + '(predictions)')
            else:
                fig.canvas.manager.set_window_title(sample_token)
        else:  # Set title on if rendering as part of render_sample.
            ax.set_title(camera_channel)
        cluster_mask=None
        if show_cluster and labels is not None:
            coloring=[]
            for _, label in enumerate(labels):
                coloring+=[label]
            counts = np.bincount(labels)
            label = np.argmax(counts)
            cluster_mask = (labels == label)

        ax.imshow(im)
        if cluster_mask is not None:

            ax.scatter(points[0, :][cluster_mask], points[1, :][cluster_mask], c=
                       label.repeat(points[0, :][cluster_mask].shape), s=dot_size+2,marker='*')
            ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
        else:
            ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size)
        if height_labels is not None:
            for height_label in height_labels[1]:
                ax.scatter(points[0,height_label[0]],points[1,height_label[0]],c="r",s=5)
            ax.set_title(f"the predict flood depth is {height_labels[0]}")
        ax.axis('off')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
        if verbose:
            plt.show()



    def get_tokens(self,sample_token,pointsensor_channel,camera_channel):
        sample_record = self.nusc.get('sample', sample_token)
        pointsensor_token = sample_record['data'][pointsensor_channel]
        camera_token = sample_record['data'][camera_channel]
        return pointsensor_token,camera_token
    def write_obj(self,pc,name):
        filename = name+'.obj'

        # 打开文件准备写入
        with open(filename, 'w') as file:
            for idx in range(pc.points.shape[1]):
                if pc.points[0,idx]!=np.nan and pc.points[0,idx]!=np.inf and pc.points[0,idx]!=-np.inf:
                    file.write(f'v {pc.points[0,idx]} {pc.points[1,idx]} {pc.points[2,idx]}\n')
            # 遍历每个点
            # for point in pc:
            #     # 每个点写入一行，格式为 'v x y z'，x, y, z 是点的坐标
            #     file.write(f'v {point[0]} {point[1]} {point[2]}\n')

    def depth2lidar(self,depth_path,sample_token,pointsensor_channel="LIDAR_TOP",camera_channel="CAM_FRONT"):
        pointsensor_token, camera_token=self.get_tokens(sample_token,pointsensor_channel,camera_channel)
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))
        depth=torch.from_numpy(torch.load(depth_path))#.transpose(1,0)
        origin_depth,origin_mask=self.pc_to_depth(deepcopy(pc),pointsensor,cam)
        depth=(depth)/(depth.max()-depth.min())*(origin_depth.max() - origin_depth.min())+origin_depth.min()-depth.min()
        H,W=depth.shape
        coords_h = torch.arange(H).float().reshape(H,1).repeat(1,W)
        coords_w = torch.arange(W).float().reshape(1,W).repeat(H,1)
        coords = torch.stack([coords_w, coords_h, depth, coords_h.new_ones(coords_h.shape)], dim=-1)
        coords[..., :2] = coords[..., :2] * coords[..., 2:3]
        coords=coords.reshape(-1,4)

        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        intrinsic=cs_record["camera_intrinsic"]
        coords[:,0]=(coords[:,0]-intrinsic[0][2]*coords[:,2])/intrinsic[0][0]
        coords[:, 1] = (coords[:, 1] - intrinsic[1][2] * coords[:, 2]) / intrinsic[1][1]
        #随机筛选一半
        indices_to_select = coords.shape[0] // 10
        selected_indices = torch.randint(0, coords.shape[0], (indices_to_select,))
        coords=coords[selected_indices]
        depth=Depth(coords)

        #1st: camera to ego
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        depth.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        depth.translate(np.array(cs_record['translation']))
        #2ed:ego to global at timestamp of image
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        depth.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        depth.translate(np.array(poserecord['translation']))
        #3th: global to ego at timestamp of pc
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        depth.translate(-np.array(poserecord['translation']))
        depth.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
#       #4th:ego to pc
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        depth.translate(-np.array(cs_record['translation']))
        depth.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)
        new_pc=depth
        pc.points=pc.points[:,origin_mask]
        # new_pc.points=new_pc.points[[2,0,1,3],:]
        self.write_obj(pc,"old_pc")
        self.write_obj(new_pc, "new_absolute_pc")
        return new_pc
    def select(self,sample_token,mask_path,pointsensor_channel="LIDAR_TOP",camera_channel="CAM_FRONT"):
        pointsensor_token, camera_token=self.get_tokens(sample_token,pointsensor_channel,camera_channel)
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))
        mask=torch.load(mask_path)
        self.choose_pc_with_flood_mask(pointsensor,cam,pc,im,mask)



if __name__ == '__main__':
    depth_path="../"
    nuscenes_path='/public/DATA/lxy/dataset/nuscenes'
    nusc = NuScenes(version='v1.0-mini',dataroot=nuscenes_path)
    nusc_depth=NuscDepth(nusc)
    #nusc_depth.depth2lidar(depth_path="../Depth-Anything-V2/absolute_depth_flood.pth",sample_token="3950bd41f74548429c0f7700ff3d8269")
    nusc_depth.select(sample_token="3950bd41f74548429c0f7700ff3d8269",mask_path="/public/DATA/lxy/workspace/flood_assess/xl-inpainting/mask.pth")
