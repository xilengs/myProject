"""
工具函数集合, 包含IoU计算， bounding box 变换等
IoU 用于判定proposals 是否为正/负样本
bounding box transform 用于训练回归器，使候选框更接近GT
"""
import numpy as np
from torch import dtype


def iou(box1, box2):
    """
    计算两个 box 的 IoU
    :param box1: [x1, y1, x2, y2]
    :param box2: [x1, y1, x2, y2]
    :return: IoU
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1 + 1)
    inter_h = max(0, y2 - y1 + 1)
    inter_area = inter_h * inter_w

    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box1[1] + 1)

    return inter_area / float(area1 + area2 - inter_area + 1e-6)

def bbox_transform(ex_rois, gt_rois):
    """
    bounding box regression
    :param ex_rois: proposals
    :param gt_rois: ground truth
    :return: [dx, dy, dw, dh]
    """
    ex_rois, gt_rois = np.array(ex_rois), np.array(gt_rois)

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def apply_bbox_transform(boxes, deltas):
    """
    反向变换：给定 proposals 和预测的 (dx, dy, dw, dh), 生成 refined boxes
    bounding box regressor 的预测性输出
    :param boxes: proposals
    :param deltas: 预测 (dx, dy, dw, dh)
    :return: 预测框
    """
    boxes = boxes.astype(np.float32)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros(deltas.shape, dtype=np.float32)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w - 1.0
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h - 1.0

    return pred_boxes