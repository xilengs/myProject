# 使用opencv的Selective Search算法为每张图像生成候选框

import cv2
import os
import json
# 进度条库
from tqdm import tqdm
import argparse

def selective_search(img, mode='fast'):
    """
    img: 输入图像(numpy array)
    mode: fast/quality
    输出：候选区域列表[x1, y1, x2, y2]
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    if mode == 'fast':
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()

    return [[x, y, x+w, y+h] for (x, y, w, h) in rects]

def generate_proposals(img_dir, out_file, max_regions=2000):
    """
    为整个数据集生成 proposals
    :param img_dir: 图片路径
    :param out_file: 输出JSON文件
    :param max_regions: 每张图片保留的最大候选框数量
    """
    results = {}
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')])
    for f in tqdm(files):
        path = os.path.join(img_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue
        rects = selective_search(img, mode='fast')
        results[f] = rects[:max_regions]

    # 保存为JSON， 方便后续使用
    with open(out_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True, help="photo folder path")
    parser.add_argument('--out', required=True, help="output folder path")
    args = parser.parse_args()
    generate_proposals(args.img_dir, args.out)