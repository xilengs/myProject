"""
生成图像列表文件
这个列表在提取proposals和训练时用到
"""
import os
import argparse
from pascal_voc_reader import load_voc_annotations

def write_image_list(voc_root, out_txt):
    img_dir = os.path.join(voc_root, 'JPEGImages')
    ann_dir = os.path.join(voc_root, 'Annotations')
    imgs = []
    for fn in sorted(os.listdir(img_dir)):
        if fn.lower().endswith('.jpg'):
            key = os.path.splitext(fn)[0]
            xml = os.path.join(ann_dir, key + '.xml')
            if not os.path.exists(xml):
                continue
            imgs.append(fn)
        with open(out_txt, 'w') as f:
            for im in imgs:
                f.write(im + '\n')
    print('wrote', len(imgs), 'images to', out_txt)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--voc_root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    write_image_list(args.voc_root, args.out)


