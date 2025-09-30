import json
import cv2
import torch.nn
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pascal_voc_reader import load_voc_annotations
from torchvision import transforms
from utils import iou, bbox_transform


class RCNNDataset(Dataset):
    def __init__(self, voc_root, image_list_file, proposals_file, classes, bg_class_id=0, pos_iou_thresh=0.5, neg_iou_thresh=0.1, im_transform=None, crop_size=224, samples_per_image=64):
        self.voc_root = voc_root
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.classes = classes
        self.num_classes = len(classes)
        self.bg_class_id = bg_class_id
        self.pos_iou_thresh = pos_iou_thresh    # IoU > 0.5 设为正样本
        self.neg_iou_thresh = neg_iou_thresh # IoU < 0.5 and IoU >= 0.1 设为负样本
        self.crop_size = crop_size
        self.samples_per_image = samples_per_image

        with open(image_list_file, 'r') as f:
            self.image_names = [line.strip() for line in f.readlines()]

        with open(proposals_file, 'r') as f:
            self.proposals = json.load(f)

        self.annotations = load_voc_annotations(voc_root, subset=set(self.image_names))

        if im_transform is None:
            self.im_transform = transforms.Compose([
                # transforms.Resize(224, 224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.im_transform = im_transform

        self.class_to_idx = {name: i + 1 for i, name in enumerate(classes)}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        im_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, im_name)

        img = cv2.imread(img_path)
        if img is None:
            return self.__getitem__((idx+1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        proposals = np.array(self.proposals.get(im_name, []))
        gt_objs = self.annotations.get(im_name, [])

        if len(proposals) == 0 or len(gt_objs) == 0:
            return self.__getitem__((idx + 1) % len(self))

        gt_boxes = np.array([obj['bbox'] for obj in gt_objs])
        gt_classes = np.array(self.class_to_idx[obj['label']] for obj in gt_objs)

        ious = np.array([iou(p, g) for p in proposals for g in gt_boxes]).reshape(len(proposals), len(gt_boxes))

        max_ious = ious.max(axis=1)
        gt_assignment = ious.argmax(axis=1)

        labels = np.zeros(len(proposals), dtype=np.int32)
        bbox_targets = np.zeros((len(proposals), 4), dtype=np.float32)

        pos_idx = np.where(max_ious >= self.pos_iou_thresh)[0]
        labels[pos_idx] = gt_classes[gt_assignment[pos_idx]]

        if len(pos_idx) > 0:
            assigned_gt = gt_boxes[gt_assignment[pos_idx]]
            assigned_proposals = proposals[pos_idx]
            bbox_targets[pos_idx, :] = bbox_transform(assigned_proposals, assigned_gt)

        neg_idx = np.where((max_ious < self.pos_iou_thresh) &
                           (max_ious >= self.neg_iou_thresh))[0]



