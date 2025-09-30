"""
读取VOC标注文件
"""
import os
import xml.etree.ElementTree as ET

def parse_voc_annotation(xml_file):
    """
    解析单个xml文件
    :param xml_file
    :return:
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        bbox = [int(float(bndbox.find('xmin').text)),
                int(float(bndbox.find('ymin').text)),
                int(float(bndbox.find('xmax').text)),
                int(float(bndbox.find('ymax').text))]

        objs.append({'label': name, 'bbox': bbox})

    return objs

def load_voc_annotations(voc_root, subset=None):
    """
    读取整个VOC数据集
    :param voc_root:
    :param subset: 可选，只加载指定图片列表
    :return: dict[imname.jpg] -> list of annotations
    """

    ann_dir = os.path.join(voc_root, 'Annotations')
    records = {}
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.endswith('.xml'):
            continue
        key = os.path.splitext(fn)[0]
        img_name = key + '.jpg'
        if subset is not None and img_name not in subset:
            continue
        xml_path = os.path.join(ann_dir, fn)
        objs = parse_voc_annotation(xml_path)
        records[img_name] = objs
    return records
