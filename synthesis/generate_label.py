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
import glob
from copy import deepcopy
from depth2lidar import Depth,NuscDepth
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import random
root_path=r''
mask_folds=[
            './SU-Flood/dataset/nuscenes/MINI_CAM_FRONT_part6',
            ]
prompt_list=[
            "waterlogging",
             "Big puddle on the street",
            "flood waterlogging on the street",
             "the floodwater on the road appears to be several inches deep",
             "the floodwater on the road appears to be several inches deep, flood other things on the road"
             ]
class NuscLableDepth(NuscDepth):
    def generate_label(self, name, sample_token, image_filename=None,pointsensor_channel="LIDAR_TOP", camera_channel="CAM_FRONT"):
        pointsensor_token, camera_token = self.get_tokens(sample_token, pointsensor_channel, camera_channel)
        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        im = Image.open(name)
        maskfold=osp.join(root_path,name.split('/')[-3])
        maskfold=osp.join(maskfold,name.split('/')[-2]+'/post_mask')
        fold = osp.join(maskfold, "mask_"+name.split('/')[-1])
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

        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))
        #ego to global
        # poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        # pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        # pc.translate(np.array(poserecord['translation']))
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
        # z_min=min(preprocessed_data_array)
        # z_max=pc.points[2,:].max()
        where=(np.where(pc.points[2,:] == max_cell_value),np.where(pc.points[2,:] == min_cell_value))
        return z_max,normal_points,where
    def generate_label_with_flood_mask(self,pointsensor,cam,pc,im,flood_mask,image_filename=None,min_dist=1,visual=True):

        depths,points,mask,u,v=self.pc_to_depth_and_get_uv(deepcopy(pc), pointsensor, cam)
        points=points[:,mask]
        try:
            flood_mask=np.array(flood_mask)[...,0].transpose(1,0)
        except:
            flood_mask = np.array(flood_mask).transpose(1, 0)
        # #explosion
        # flood_mask[:, :-1] = np.where(flood_mask[:, 1:] > 200, 255, flood_mask[:, :-1])

        sec_mask=flood_mask[points[0,:].astype(int),points[1,:].astype(int)]>200 #pc in water

        pc.points=pc.points[:,mask][:,sec_mask]
        points=points[:,sec_mask]
        depths=depths[sec_mask]

        image_points=np.asarray(im).transpose(1,0, 2)[points[0,:].astype(int),points[1,:].astype(int),:]
        kmeans_feature=np.concatenate((image_points,depths.reshape(-1,1),pc.points[0].reshape(-1,1),pc.points[1].reshape(-1,1)),axis=1)
        kmeans=KMeans(n_clusters=2).fit(kmeans_feature)
        counts = np.bincount(kmeans.labels_)
        label=np.argmax(counts)
        cluster_mask=(kmeans.labels_==label)
        z_max,preprocessed_data_array,where=self.cal_max_depth(deepcopy(pointsensor),deepcopy(pc))
        if not osp.exists("./depth_label_show/" + image_filename.split('/')[-3]):
            os.mkdir("./depth_label_show/" + image_filename.split('/')[-3])
        if not osp.exists("./depth_label_show/" + image_filename.split('/')[-3] + "/" + image_filename.split('/')[-2]):
            os.mkdir("./depth_label_show/" + image_filename.split('/')[-3] + "/" + image_filename.split('/')[-2])
        if visual:
            self.render_pointcloud_in_image(deepcopy(pc), im, pointsensor, cam,
                                            out_path="./depth_label_show/" + image_filename.split('/')[-3] + "/" +
                                                     image_filename.split('/')[-2] + "/" + image_filename.split('/')[
                                                         -1], sample_token="selected_pc", dot_size=3,
                                            height_labels=(z_max, where),
                                            verbose=False)  # ,show_cluster=True,labels=kmeans.labels_)

        # #pc to ego
        # cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        # pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        # pc.translate(np.array(cs_record['translation']))
        # #ego to global
        # poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        # pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        # pc.translate(np.array(poserecord['translation']))
        #
        # z_max=pc.points[2,:].max()
        return z_max
    def from_image_name_to_sample_token(self,camera_channel="CAM_FRONT"):
        sample_tokens={}
        image_filenames=[]
        for maskfold in mask_folds:
            for prompt in prompt_list:
                fold=osp.join(maskfold,prompt)
                filename=glob.glob(os.path.join(fold, '*.jpg'))
                image_filenames+=filename

        for index, sample in enumerate(self.nusc.sample):
            camera_token = sample['data'][camera_channel]
            image_name = nusc.get('sample_data', camera_token)['filename'].split('/')[-1]
            for maskfold in mask_folds:
                for prompt in prompt_list:
                    fake_name=osp.join(osp.join(maskfold,prompt),image_name)
                    if fake_name in image_filenames:
                        sample_tokens[fake_name]=sample["token"]
        return image_filenames,sample_tokens




if __name__ == '__main__':
    depth_path="../"
    nuscenes_path=''
    nusc = NuScenes(version='v1.0-trainval',dataroot=nuscenes_path)
    nusc_depth=NuscLableDepth(nusc)
    image_filenames,sample_tokens=nusc_depth.from_image_name_to_sample_token()
    label_list=[]
    for image_filename in image_filenames:
        sample_token=sample_tokens[image_filename]
        label=nusc_depth.generate_label(image_filename,sample_token,image_filename)
        label_list+=[label]
    with open("nusc_deep.txt","w") as file:
        for image_filename,label in zip(image_filenames,label_list):
            file.write(f"{image_filename},{label}\n")
