import os
import random
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import json

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的 CLIP 模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# 随机抽取图像路径
def sample_images(folder, num_samples=500):
    png_files = []
    for subdir, _, files in os.walk(folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                png_files.append(os.path.join(subdir, file))
    return random.sample(png_files, min(num_samples, len(png_files)))


def sample_images_from_file(num_samples=500):
    json_file_paths = ['final_v_train.json', 'final_v_test.json']
    file_name = []
    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as f:
            eval_data = json.load(f)
        images = eval_data['images']
        for image in images:
            file_name.append(image["file_name"])
    return random.sample(file_name, min(num_samples, len(file_name)))


# 预处理图像
def preprocess_images(image_paths):
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    return processor(images=images, return_tensors="pt", padding=True).to(device)


# 提取图像特征
def extract_image_features(image_paths):
    inputs = preprocess_images(image_paths)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features /= features.norm(dim=-1, keepdim=True)  # 归一化
    return features


# 提取文本特征
def extract_text_features(text):
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    features /= features.norm(dim=-1, keepdim=True)  # 归一化
    return features


def compute_similarity(image_features, text_features, reduction="mean"):
    """
    计算图像和文本的相似度，支持聚合多个描述的相似度。
    :param image_features: 图像特征 (N, D)
    :param text_features: 文本特征 (M, D)
    :param reduction: "mean" 或 "max"，指定如何聚合多个描述的相似度
    :return: 相似度数组 (N,)
    """
    similarity_matrix = image_features @ text_features.T  # (N, M)

    if reduction == "mean":
        return similarity_matrix.mean(dim=-1)  # 平均相似度
    elif reduction == "max":
        #print(similarity_matrix.max(dim=-1).values)
        return similarity_matrix.max(dim=-1).values  # 最大相似度
    else:
        raise ValueError("Unsupported reduction type. Use 'mean' or 'max'.")
# 分析相似度结果
def analyze_similarities(image_paths, similarities):
    similarities=similarities.cpu().numpy()
    average_similarity = np.mean(similarities)
    sorted_indices = np.argsort(similarities)  # 升序排序

    # 最低相似度图像
    lowest_idx = sorted_indices[0]
    lowest_path = image_paths[lowest_idx]
    lowest_similarity = similarities[lowest_idx]

    # 最高相似度图像
    highest_idx = sorted_indices[-1]
    highest_path = image_paths[highest_idx]
    highest_similarity = similarities[highest_idx]

    return average_similarity, (lowest_path, lowest_similarity), (highest_path, highest_similarity)


# 主要数据集路径
data2path = {
    "ITSC": "/public/DATA/lxy/dataset/itsc_flood_dataset/images",
    "EU-FLOOD": "/public/DATA/lxy/dataset/EU-flood/selected",
    "VFLOOD": "/public/DATA/lxy/dataset/WaterDataset/train_images/JPEGImages/flooding_pixabay",
    "FloodIMG": "/public/DATA/lxy/dataset/Flood_Images"
}


text = [
    "The river has overflowed, flooding streets and homes in its wake.",
    "Buildings are partially submerged, with only their rooftops visible above the water.",
    "Fast-moving water rushes through the town, carrying debris and wreckage.",
    "Rescue boats patrol the flooded streets, searching for stranded residents.",
    "The sky is heavy with dark clouds, threatening more rainfall.",
    "Cars are abandoned on the road, some submerged, others barely visible above the water.",
    "People stand on rooftops, waving for help as the floodwaters continue to rise.",
    "Children are seen clinging to trees, trying to escape the rising tide.",
    "The sound of rushing water fills the air, drowning out all other noises.",
    "Utility poles and power lines are knocked down, leaving the area in darkness.",
    "Streets have turned into rivers, with swift currents making navigation impossible.",
    "Floodwaters have submerged parks, turning green spaces into vast lakes.",
    "Mud and debris cover the roads, making rescue efforts difficult.",
    "Emergency shelters are overcrowded as people seek refuge from the rising water.",
    "The floodwaters have washed away entire neighborhoods, leaving a path of destruction.",
    "Evacuation centers are overwhelmed, with volunteers working tirelessly to assist.",
    "The flood has left behind a trail of wrecked homes and ruined possessions.",
    "Animals are stranded in the flood, with some rescued by volunteers.",
    "The flood’s force has torn up streets, uprooting trees and washing away cars.",
    "Rescue teams work tirelessly, braving dangerous waters to save lives."
]

# 提取文本特征
text_features = extract_text_features(text)

# 从 JSON 文件中获取图像并计算相似度
print("Calculating similarity for sampled images from JSON...")
average_similarity_list=[]
lowest=1
for i in range(10):
    images2 = sample_images_from_file(1000)
    image_features = extract_image_features(images2)
    similarity = compute_similarity(image_features,text_features,reduction="max")
    average_similarity, lowest_image, highest_image = analyze_similarities(images2, similarity)

    print("\nJSON Dataset Results:")
    print(f"Average Similarity: {average_similarity:.4f}")
    print(f"Lowest Similarity Image: {lowest_image[0]}, Similarity: {lowest_image[1]:.4f}")
    print(f"Highest Similarity Image: {highest_image[0]}, Similarity: {highest_image[1]:.4f}")
    average_similarity_list+=[average_similarity]
    if lowest>lowest_image[1]:
        lowest=lowest_image[1]
print(f"10 fold Average Similarity: {sum(average_similarity_list)/len(average_similarity_list):.4f}")
print(f"Lowest Similarity: {lowest}")
