import math
import os
from collections import Counter

import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Defining project root in order to avoid relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initializing torch device according to hardware available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

IMG_SIZE = 256

CONFIDECNE_TRESHOLD = 0.5
IOU_TRESHOLD = 0.5


def collate_fn(batch):
    """
    Function to combine images, boxes and labels
    :param batch: an iterable of N sets from __getitem__() of CustomDataset
    :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
    """
    images = list()
    boxes = list()
    labels = list()
    mask_coords = list()
    objectness_mask = list()
    boxes_mask = list()
    labels_mask = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1]["boxes"])
        labels.append(b[1]["labels"])
        mask_coords.append(b[1]["objectness"]["coords"])
        objectness_mask.append(b[1]["objectness"]["matrix"])
        boxes_mask.append(b[1]["boxes_mask"])
        labels_mask.append(b[1]["labels_mask"])

    images = torch.stack(images)
    objectness_mask = torch.stack(objectness_mask)
    boxes_mask = torch.stack(boxes_mask)
    labels_mask = torch.stack(labels_mask)

    return images, (boxes, labels, mask_coords, objectness_mask, boxes_mask, labels_mask)


def with_bounding_box(image, target):
    """
    Returns an image with bounding boxes and labels
    :param image: image as Tensor
    :param target: dict representing containing the bounding boxes
    """
    tensor_image = torchvision.utils.draw_bounding_boxes(transforms.PILToTensor()(transforms.ToPILImage()(image)), target['boxes'], target['categories'], colors="red", width=2)
    return transforms.ToPILImage()(tensor_image)


def plot_aspect_ratio_distribution(dataset):
    """
    Returns the aspect ratio distribution of a CustomDataset
    :param dataset: the dataset of type CustomDataset
    """
    aspect_ratios = np.empty(len(dataset), dtype=float)
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        sizes = img.size
        aspect_ratios = np.append(aspect_ratios, sizes[0] / sizes[1])

    plt.bar(*np.unique(aspect_ratios, return_counts=True))
    return plt


def to_center_coords(boxes):
    new_boxes = []
    for box in boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = math.ceil(box[0] + w/2)
        y = math.ceil(box[1] + h/2)
        new_boxes.append([x, y, w, h])
    return new_boxes


def i_over_u(batched_predicted_boxes: torch.Tensor, batched_target_boxes: torch.Tensor):
    """
    Compute intersection over union of batched Tensors
    """

    pred_x1 = batched_predicted_boxes[..., 0:1] - batched_predicted_boxes[..., 2:3] / 2
    pred_y1 = batched_predicted_boxes[..., 1:2] - batched_predicted_boxes[..., 3:4] / 2
    pred_x2 = batched_predicted_boxes[..., 0:1] + batched_predicted_boxes[..., 2:3] / 2
    pred_y2 = batched_predicted_boxes[..., 1:2] + batched_predicted_boxes[..., 3:4] / 2

    target_x1 = batched_target_boxes[..., 0:1] - batched_target_boxes[..., 2:3] / 2
    target_y1 = batched_target_boxes[..., 1:2] - batched_target_boxes[..., 3:4] / 2
    target_x2 = batched_target_boxes[..., 0:1] + batched_target_boxes[..., 2:3] / 2
    target_y2 = batched_target_boxes[..., 1:2] + batched_target_boxes[..., 3:4] / 2

    intersection_area = (torch.min(pred_x2, target_x2) - torch.max(pred_x1, target_x1)).clamp(0) * (torch.min(pred_y2, target_y2) - torch.max(pred_y1, target_y1)).clamp(0)

    pred_area = torch.abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    target_area = torch.abs((target_x2 - target_x1) * (target_y2 - target_y1))

    union_area = pred_area + target_area - intersection_area

    return intersection_area/(union_area + 1e-8)


def non_max_suppression(bounding_boxes: list):
    """
    Perform Non Max Suppression to find which predicted bounding box is the best
    :param bounding_boxes: list of lists of the type [class, confidence, x, y, h, w]
    :return:
    """
    # Purge boxes which confidence is too low
    bounding_boxes = [box for box in bounding_boxes if box[1] < CONFIDECNE_TRESHOLD]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True)
    boxes = []
    while bounding_boxes:
        max_confidence_box = bounding_boxes.pop(0)
        boxes.append(max_confidence_box)

        for index, box in enumerate(bounding_boxes):
            iou = i_over_u(torch.tensor(max_confidence_box[2:], device=DEVICE), torch.tensor(box[2:], device=DEVICE))
            if box[0] == max_confidence_box[0] and iou >= IOU_TRESHOLD:
                bounding_boxes.pop(index)
    return boxes


def mean_average_precision(predictions, targets):
    """
    Computes mAP to evaluate model
    :param predictions: list of lists of the type [image_index, class, confidence, x, y, w, h]
    :param targets: as the previous but with only ture boxes
    :return: mAP score
    """
    average_precisions = []

    for cla in range(13):
        ground_truths = [ground_truth for ground_truth in targets if ground_truth[1] == cla]
        total_true_bounding_boxes = len(ground_truths)

        if total_true_bounding_boxes == 0:
            continue

        detections = [detection for detection in predictions if detection[1] == cla]
        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections), device=DEVICE)
        FP = torch.zeros(len(detections), device=DEVICE)

        amount_of_bounding_boxes = Counter([ground_truth[0] for ground_truth in ground_truths])
        for key, val in amount_of_bounding_boxes.items():
            amount_of_bounding_boxes[key] = torch.zeros(val, device=DEVICE)

        for detection_id, detection in enumerate(detections):
            filtered_ground_truths = [box for box in ground_truths if box[0] == detection[0]]
            best_iou = 0

            for ground_truth_id, ground_truth in enumerate(filtered_ground_truths):
                iou = i_over_u(
                    torch.tensor(detection[3:], device=DEVICE),
                    torch.tensor(ground_truth[3:], device=DEVICE)
                )

                if iou > best_iou:
                    best_iou = iou
                    best_ground_truth_id = ground_truth_id

            if best_iou > IOU_TRESHOLD and amount_of_bounding_boxes[detection[0]][best_ground_truth_id] == 0:
                    TP[detection_id] = 1
                    amount_of_bounding_boxes[detection[0]][best_ground_truth_id] = 1
            else:
                FP[detection_id] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bounding_boxes + 1e-6)
        recalls = torch.cat((torch.tensor([0], device=DEVICE), recalls))
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))
        precisions = torch.cat((torch.tensor([1], device=DEVICE), precisions))
        average_precisions.append(torch.trapz(precisions, recalls))
        return sum(average_precisions) / len(average_precisions), recalls, precisions
