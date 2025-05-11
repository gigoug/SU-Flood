import os
from typing import Tuple, Optional

import cv2
import gradio as gr
import numpy as np
import supervision as sv
import torch
import os.path as osp
from PIL import Image
from tqdm import tqdm
from utils.video import generate_unique_name, create_directory, delete_directory

from utils.florence import load_florence_model, run_florence_inference, \
    FLORENCE_DETAILED_CAPTION_TASK, \
    FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, \
    IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
from utils.sam import load_sam_image_model, run_sam_inference, load_sam_video_model

MARKDOWN = """
# Florence2 + SAM2 🔥

<div>
    <a href="https://github.com/facebookresearch/segment-anything-2">
        <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub" style="display:inline-block;">
    </a>
    <a href="https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-segment-images-with-sam-2.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" style="display:inline-block;">
    </a>
    <a href="https://blog.roboflow.com/what-is-segment-anything-2/">
        <img src="https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg" alt="Roboflow" style="display:inline-block;">
    </a>
    <a href="https://www.youtube.com/watch?v=Dv003fTyO-Y">
        <img src="https://badges.aleen42.com/src/youtube.svg" alt="YouTube" style="display:inline-block;">
    </a>
</div>

This demo integrates Florence2 and SAM2 by creating a two-stage inference pipeline. In 
the first stage, Florence2 performs tasks such as object detection, open-vocabulary 
object detection, image captioning, or phrase grounding. In the second stage, SAM2 
performs object segmentation on the image.
"""

IMAGE_PROCESSING_EXAMPLES = [
    [IMAGE_OPEN_VOCABULARY_DETECTION_MODE, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", 'straw, white napkin, black napkin, hair'],
    [IMAGE_OPEN_VOCABULARY_DETECTION_MODE, "https://media.roboflow.com/notebooks/examples/dog-3.jpeg", 'tail'],
    [IMAGE_CAPTION_GROUNDING_MASKS_MODE, "https://media.roboflow.com/notebooks/examples/dog-2.jpeg", None],
    [IMAGE_CAPTION_GROUNDING_MASKS_MODE, "https://media.roboflow.com/notebooks/examples/dog-3.jpeg", None],
]
VIDEO_PROCESSING_EXAMPLES = [
    ["videos/clip-07-camera-1.mp4", "player in white outfit, player in black outfit, ball, rim"],
    ["videos/clip-07-camera-2.mp4", "player in white outfit, player in black outfit, ball, rim"],
    ["videos/clip-07-camera-3.mp4", "player in white outfit, player in black outfit, ball, rim"]
]

VIDEO_SCALE_FACTOR = 0.5
VIDEO_TARGET_DIRECTORY = "tmp"
create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)
SAM_IMAGE_MODEL = load_sam_image_model(device=DEVICE)
SAM_VIDEO_MODEL = load_sam_video_model(device=DEVICE)
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#32CD32', '#8A2BE2']
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
BOX_ANNOTATOR = sv.BoxAnnotator(color=COLOR_PALETTE, color_lookup=sv.ColorLookup.INDEX)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX,
    text_position=sv.Position.CENTER_OF_MASS,
    text_color=sv.Color.from_hex("#000000"),
    border_radius=5
)
MASK_ANNOTATOR = sv.MaskAnnotator(
    color=COLOR_PALETTE,
    color_lookup=sv.ColorLookup.INDEX
)


def annotate_image(image, detections):
    output_image = image.copy()
    output_image = MASK_ANNOTATOR.annotate(output_image, detections)
    output_image = BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections)
    return output_image


def on_mode_dropdown_change(text):
    return [
        gr.Textbox(visible=text == IMAGE_OPEN_VOCABULARY_DETECTION_MODE),
        gr.Textbox(visible=text == IMAGE_CAPTION_GROUNDING_MASKS_MODE),
    ]


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process_image(
    mode_dropdown, image_input, text_input, save_name
) :
    if not image_input:
        gr.Info("Please upload an image.")
        return None, None

    if mode_dropdown == IMAGE_OPEN_VOCABULARY_DETECTION_MODE:
        if not text_input:
            gr.Info("Please enter a text prompt.")
            return None, None

        texts = [prompt.strip() for prompt in text_input.split(",")]
        detections_list = []
        for text in texts:
            _, result = run_florence_inference(
                model=FLORENCE_MODEL,
                processor=FLORENCE_PROCESSOR,
                device=DEVICE,
                image=image_input,
                task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
                text=text
            )
            detections = sv.Detections.from_lmm(
                lmm=sv.LMM.FLORENCE_2,
                result=result,
                resolution_wh=image_input.size
            )
            detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
            detections_list.append(detections)

        detections = sv.Detections.merge(detections_list)
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        mask = np.any(detections.mask, axis=0)
        image = Image.fromarray(mask.astype(np.uint8) * 255,'L')
        image.save(save_name)
        return annotate_image(image_input, detections), None

    if mode_dropdown == IMAGE_CAPTION_GROUNDING_MASKS_MODE:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_DETAILED_CAPTION_TASK
        )
        caption = result[FLORENCE_DETAILED_CAPTION_TASK]
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=image_input,
            task=FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK,
            text=caption
        )
        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=image_input.size
        )
        detections = run_sam_inference(SAM_IMAGE_MODEL, image_input, detections)
        image = Image.fromarray(detections.mask.squeeze().astype(np.uint8) * 255,'L')
        image.save(save_name)
        return None

import glob
def get_jpg_jpeg_files(directory):
    # 确保目录路径是字符串类型，并且适用于当前操作系统
    directory = os.path.abspath(directory)
    if not directory.endswith(os.sep):
        directory += os.sep

        # 使用glob模块搜索.jpg和.jpeg文件
    jpg_files = glob.glob(directory + '*.jpg')
    jpeg_files = glob.glob(directory + '*.jpeg')

    # 合并结果
    all_jpg_jpeg_files = jpg_files + jpeg_files

    return all_jpg_jpeg_files

image_dirs=[
    './SU-Flood/lyft/part5/Big puddle on the street',
    './SU-Flood/lyft/part5/flood waterlogging on the street'
]

text_input="water"
for dir in image_dirs:
    if not osp.exists(osp.join(dir,"post_mask")):
        os.mkdir(osp.join(dir,"post_mask"))
    image_list=get_jpg_jpeg_files(dir)
    for file_path in image_list:
        image_input=Image.open(file_path)
        output_name=osp.join(dir,"post_mask")+"/mask_"+file_path.split('/')[-1].split('.')[0]+".jpg"
        image=process_image(
            IMAGE_OPEN_VOCABULARY_DETECTION_MODE, image_input, text_input, output_name
        )
        if not osp.exists("./mask_show/"+file_path.split('/')[-3]):
            os.mkdir("./mask_show/"+file_path.split('/')[-3])
        if not osp.exists("./mask_show/"+file_path.split('/')[-3]+"/"+file_path.split('/')[-2]):
            os.mkdir("./mask_show/"+file_path.split('/')[-3]+"/"+file_path.split('/')[-2])
        image[0].save("./mask_show/"+file_path.split('/')[-3]+"/"+file_path.split('/')[-2]+"/"+file_path.split('/')[-1].split('.')[0]+".jpg")
