from nuscenes.nuscenes import NuScenes
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
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
import glob
from copy import deepcopy
from depth2lidar import Depth,NuscDepth
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random
from pathlib import Path
root_path='./dataset/lyft/'

mask_folds=['SUFlood-dataset/lyft/part5']

prompt_list=[
             "Big puddle on the street",
             "flood waterlogging on the street",
             ]
class LyftLableDepth(LyftDataset):
    def __init__(self,lyft):
        self.lyft=lyft
    def generate_label(self, name, sample_token, image_filename=None,pointsensor_channel="LIDAR_TOP", camera_channel="CAM_FRONT"):
        #pointsensor_token, camera_token = self.get_tokens(sample_token, pointsensor_channel, camera_channel)
        camera_token=self.lyft.get('sample',sample_token)['data']['CAM_FRONT']
        pointsensor_token=self.lyft.get('sample',sample_token)['data']['LIDAR_TOP']
        cam = self.lyft.get('sample_data', camera_token)
        pointsensor = self.lyft.get('sample_data', pointsensor_token)
        pcl_path = Path(osp.join(root_path, pointsensor['filename']))
        pc = LidarPointCloud.from_file(pcl_path)
        im = Image.open(name)
        maskfold=osp.join(root_path,name.split('/')[-3])
        maskfold=osp.join(maskfold,name.split('/')[-2]+'/post_mask')
        fold = osp.join(maskfold, "mask_"+name.split('/')[-1][:-4]+"jpg")
        if os.path.exists(fold):
            print(f"has {fold}, start to generate label")
            mask = Image.open(fold)
            label=self.generate_label_with_flood_mask(pointsensor, cam, pc, im,flood_mask=mask,image_filename=image_filename, visual=True)
            return label
        else:
            print("error: no found mask for"+name)
    def cal_max_depth(self,pointsensor,pc,cluster_mask=None,m=25):
        if cluster_mask is not None:
            pc.points=pc.points[:,cluster_mask]

        cs_record = self.lyft.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))
        #ego to global
        z_values,x_values,y_values=pc.points[2,:],pc.points[0,:],pc.points[1,:]
        z_mean,x_mean,y_mean = np.mean(pc.points[2,:], axis=0),np.mean(pc.points[0,:], axis=0),np.mean(pc.points[1,:], axis=0)
        z_std,x_std,y_std = np.std(pc.points[2,:], axis=0),np.std(pc.points[0,:], axis=0),np.std(pc.points[1,:], axis=0)
        z_low,x_low,y_low=z_mean-3*z_std,x_mean-3*x_std,y_mean-3*y_std
        z_high,x_high,y_high=z_mean+3*z_std,x_mean+3*x_std,y_mean+3*y_std
        normal_points = pc.points[:,(z_values >= z_low) & (z_values <= z_high)&(x_values >= x_low) & (x_values <= x_high)&(y_values >= y_low) & (y_values <= y_high)]

        # Assume `points` is your [n, 3] array
        x_min, x_max = normal_points[0,:].min(), normal_points[0,:].max()
        y_min, y_max = normal_points[1,:].min(), normal_points[1,:].max()

        x_bins = np.linspace(x_min, x_max, m + 1)
        y_bins = np.linspace(y_min, y_max, m + 1)

        # Assign each point to a cell
        df = pd.DataFrame(normal_points[:3,:].transpose(), columns=['x', 'y', 'z'])
        df['x_bin'] = np.digitize(df['x'], x_bins) - 1
        df['y_bin'] = np.digitize(df['y'], y_bins) - 1

        # Group by cell and calculate min and max z
        result = df.groupby(['x_bin', 'y_bin'])['z'].agg(['min', 'max']).reset_index()
        result['depth']=result['max']-result['min']

        max_depth_row = result.loc[result['depth'].idxmax()]
        z_max = max_depth_row['depth']
        max_cell_value = max_depth_row['max']
        min_cell_value=max_depth_row['min']
        where=(np.where(pc.points[2,:] == max_cell_value),np.where(pc.points[2,:] == min_cell_value))
        return z_max,normal_points,where
    def pc_to_depth(self, pc, pointsensor, cam, min_dist=1.0, im=[1600, 900]):
        cs_record = self.lyft.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.lyft.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.lyft.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.lyft.get('calibrated_sensor', cam['calibrated_sensor_token'])
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
    def pc_to_depth_and_get_uv(self, pc, pointsensor, cam, min_dist=1.0, im=[1224, 1024]):
        cs_record = self.lyft.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.lyft.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.lyft.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.lyft.get('calibrated_sensor', cam['calibrated_sensor_token'])
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
        return depths, points, mask, pc.points[0, :][mask], pc.points[1, :][mask]
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
    def generate_label_with_flood_mask(self,pointsensor,cam,pc,im,flood_mask,image_filename=None,min_dist=1,visual=True):

        depths,points,mask,u,v=self.pc_to_depth_and_get_uv(deepcopy(pc), pointsensor, cam)
        points=points[:,mask]
        flood_mask=np.array(flood_mask).transpose(1,0)
        # #explosion
        # flood_mask[:, :-1] = np.where(flood_mask[:, 1:] > 200, 255, flood_mask[:, :-1])

        sec_mask=flood_mask[points[0,:].astype(int),points[1,:].astype(int)]>200 #pc in water

        pc.points=pc.points[:,mask][:,sec_mask]
        points=points[:,sec_mask]
        depths=depths[sec_mask]

        image_points=np.asarray(im).transpose(1,0, 2)[points[0,:].astype(int),points[1,:].astype(int),:]
        z_max,preprocessed_data_array,where=self.cal_max_depth(deepcopy(pointsensor),deepcopy(pc))
        # self.write_obj(pc,"flood_pc")
        if not osp.exists("./depth_label_show/"+image_filename.split('/')[-3]):
            os.mkdir("./depth_label_show/"+image_filename.split('/')[-3])
        if not osp.exists("./depth_label_show/"+image_filename.split('/')[-3]+"/"+image_filename.split('/')[-2]):
            os.mkdir("./depth_label_show/"+image_filename.split('/')[-3]+"/"+image_filename.split('/')[-2])
        if visual:
            self.render_pointcloud_in_image(deepcopy(pc),im,pointsensor,cam,out_path="./depth_label_show/"+image_filename.split('/')[-3]+"/"+image_filename.split('/')[-2]+"/"+image_filename.split('/')[-1],sample_token="selected_pc",dot_size=3,height_labels=(z_max,where),verbose=False)#,show_cluster=True,labels=kmeans.labels_)

        return z_max
    def from_image_name_to_sample_token(self,camera_channel="CAM_FRONT"):
        sample_tokens={}
        image_filenames=[]
        for maskfold in mask_folds:
            for prompt in prompt_list:
                fold=osp.join(maskfold,prompt)
                filename=glob.glob(os.path.join(fold, '*.jpeg'))
                image_filenames+=filename
        for index, sample in enumerate(self.lyft.sample):
            camera_token = sample['data'][camera_channel]
            image_name = lyft.get('sample_data', camera_token)['filename'].split('/')[-1]
            for maskfold in mask_folds:
                for prompt in prompt_list:
                    fake_name=osp.join(osp.join(maskfold,prompt),image_name)
                    if fake_name in image_filenames:
                        sample_tokens[fake_name]=sample["token"]
        return image_filenames,sample_tokens




if __name__ == '__main__':
    depth_path="../"
    nuscenes_path=''
    lyft = LyftDataset(data_path='', json_path='', verbose=True)
    nusc_depth=LyftLableDepth(lyft)
    image_filenames,sample_tokens=nusc_depth.from_image_name_to_sample_token()
    label_list=[]
    for image_filename in image_filenames:
        try:
            sample_token=sample_tokens[image_filename]
            label=nusc_depth.generate_label(image_filename,sample_token,image_filename)
            label_list+=[label]
        except:
            label_list += ['None']
    with open("lyft_ground_label_2.txt","w") as file:
        for image_filename,label in zip(image_filenames,label_list):
            if label != 'None':
                file.write(f"{image_filename},{label}\n")
