import argparse
import os
from PIL import Image


def get_image_paths(input_folder=None, input_txt=None):
    image_paths = []

    if input_folder:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
        for root, _, files in os.walk(input_folder):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    image_paths.append(os.path.join(root, file))
    elif input_txt:
        with open(input_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    path = line.split(',')[0].strip()
                    image_paths.append(path)

    return image_paths


def calculate_average_resolution(image_paths):
    total_width = 0
    total_height = 0
    count = 0

    for path in image_paths:
        try:
            with Image.open(path) as img:
                width, height = img.size
                total_width += width
                total_height += height
                count += 1
                print(f"已处理: {path} ({width}x{height})")
        except Exception as e:
            print(f"错误: 无法处理 {path} - {str(e)}")

    if count == 0:
        return None, None, 0

    return total_width / count, total_height / count, count


def main():
    parser = argparse.ArgumentParser(description='计算图像平均分辨率')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_folder', help='包含图像的文件夹路径')
    group.add_argument('--input_txt', help='包含图像路径的文本文件路径')
    args = parser.parse_args()

    image_paths = get_image_paths(args.input_folder, args.input_txt)

    if not image_paths:
        print("错误: 未找到任何图像文件")
        return

    avg_width, avg_height, count = calculate_average_resolution(image_paths)

    if count == 0:
        print("错误: 没有成功读取任何图像")
        return

    print(f"\n统计结果:")
    print(f"总处理图像数: {count}")
    print(f"平均分辨率: {avg_width:.1f}x{avg_height:.1f}")
    print(f"最大可能的标准分辨率: {int(avg_width)}x{int(avg_height)}")

# EU-FLOOD 741.4×527.9
#itsc: 512*384.0
# V-FloodNet:657.3*508.9
#Flood_Images: 1417.1*954.7
#OUrs: 1449.9*949.5
if __name__ == "__main__":
    main()