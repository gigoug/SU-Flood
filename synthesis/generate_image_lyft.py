import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from diffusers import AutoPipelineForInpainting, UNet2DConditionModel
import diffusers
from nuscenes.nuscenes import NuScenes
from lyft_dataset_sdk.lyftdataset import LyftDataset
from PIL import Image
import os.path as osp

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)
level5data = LyftDataset(data_path=r'', json_path='', verbose=True)
name_list = []
scheduler = "EulerDiscreteScheduler"
scheduler_class_name = scheduler.split("-")[0]

add_kwargs = {}
if len(scheduler.split("-")) > 1:
    add_kwargs["use_karras"] = True
if len(scheduler.split("-")) > 2:
    add_kwargs["algorithm_type"] = "sde-dpmsolver++"

scheduler = getattr(diffusers, scheduler_class_name)
pipe.scheduler = scheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler",
                                           **add_kwargs)

from nuscenes.nuscenes import NuScenes
dataroot=''
mask_folds=['.SU-Flood/lyft/part5',
            ]
prompt_list=["flood waterlogging on the street","Big puddle on the street"]
def multi_predict(name, negative_prompt="", guidance_scale=8.5, steps=25, strength=0.99):
    if negative_prompt == "":
        negative_prompt = None

    for maskfold in mask_folds:
        mask_path = maskfold
        if not os.path.exists(osp.join(mask_path, "mask_"+name.split('/')[-1])[:-4]+"jpg"):
            continue
        mask = Image.open(osp.join(mask_path, "mask_"+name.split('/')[-1])[:-4]+"jpg").resize((1024, 1024))
        for prompt in prompt_list:
            result_path = osp.join(maskfold, prompt)
            if(os.path.exists(osp.join(result_path,name.split('/')[-1]))):
                continue
            init_image = Image.open(osp.join(dataroot, name))
            init_image = init_image.convert("RGB").resize((1024, 1024))
            output = pipe(prompt = prompt, negative_prompt=negative_prompt, image=init_image, mask_image=mask, guidance_scale=guidance_scale, num_inference_steps=int(steps), strength=strength)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            output.images[0].resize((1224,1024)).save(osp.join(result_path,name.split('/')[-1]))
            print(f"create{name.split('/')[-1]}")
    # num=num+1
    return None

if '__name__==__main__':
    #name=[]
    for index, sample in enumerate(level5data.sample):
        image_token=sample["data"]['CAM_FRONT']
        # image=sample.get(image_token)
        cam = level5data.get('sample_data', image_token)
        try:
            #name+=[cam['filename']]
            multi_predict(cam['filename'],)
        except Exception as e:
            print(e)
    #print(name)
