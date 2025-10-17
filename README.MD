## 🌊 SR-Flood: A Dataset and Benchmark for Road Flood Depth Detection via Autonomous Driving Image Editing
### 📕 [Dataset Link](https://www.kaggle.com/datasets/liangxiaoyupku/su-flood)
### 📁 Data Synthesis
The results of each step can be found in "synthesis/example". In every step, simply adjust the input and output folders to manage what goes in and what comes out. Here’s a detailed breakdown:

1. **🖌️ get user drawn mask**
```
We highly recommend using labeling tools like labelme to create accurate masks.
```
2. **🎨 generate flood image**
```python
python synthesis/generate_image.py # for nuscenes dataset 
python synthesis/generate_image_lyft.py # for lyft L5 dataset
```
3. **🔍 generate water mask**
```python
python synthesis/florence-sam/bash_generate_mask.py
```
4**📊 calculate depth**
```python
python synthesis/generate_label.py #for nuscenes dataset
python synthesis/generate_label_lyft.py #for lyft L5  dataset
```
### 🚀 Water Depth Detection
**🏋️ train**
```python
cd depth-detection
bash tools/dist_train.sh configs/distillers/suflood/neck_RAU_final_stage_1.py 2
bash tools/dist_train.sh configs/distillers/suflood/neck_RAU_final_stage_1.py 2 --ckpt ./work_dirs/stage1/latest.pth
```
**🧪 test**
```python
python pth_transfer.py --fgd_path ./work_dirs/stage2/latest.pth --output_path ./work_dirs/stage2/student_latest.pth
bash tools/dist_test.sh configs/distillers/suflood/neck_RAU_final_stage_1.py ./work_dirs/stage2/student_latest.pth 1

```

