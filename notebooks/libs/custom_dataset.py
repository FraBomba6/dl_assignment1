import json
import math
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import utils as custom_utils


class CustomDataset(Dataset):
    """
    Class that represents a dataset object to use as input on a CNN
    """
    def __init__(self, root):
        """
        Default initializer
        :param root: path to dataset root
        """
        self.root = root
        self.size = custom_utils.IMG_SIZE

        # Load images filelist
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        # Load annotations filelist
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, index):
        """
        Default getter for dataset objects
        :param index: i of the wanted image + annotation
        :return: image as PIL Image and target dictionary
        """
        img = self.__load_image(index)
        target = self.__generate_target(index)
        if self.size is not None:
            img, target = self.__apply_transform(img, target)

        target["objectness"] = self.__compute_objectness(target['boxes'])
        target["boxes_mask"] = self.__build_target_bb_mask(target['boxes'])
        target["labels_mask"] = self.__build_target_labels_mask(target['objectness']['coords'], target['labels'])

        return img, target

    def __apply_transform(self, img, target):
        """
        Apply a resize transformation to an image and its target
        :param img: image as PIL Image
        :param target: dict representing the bounding boxes
        """
        target["boxes"] = self.__resize_boxes(target["boxes"], img.size)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.size, self.size))])
        img = transform(img)
        return img, target

    def __resize_boxes(self, boxes, img_size):
        """
        Apply to bounding boxes the same resize as the corresponding image
        :param boxes: tensor containing the coordinates of the bounding boxes
        :param img_size: size of the original image
        """
        x_scale = self.size/img_size[0]
        y_scale = self.size/img_size[1]

        scaled_boxes = []
        for box in boxes:
            box = box.tolist()
            x = int(np.round(box[0] * x_scale))
            y = int(np.round(box[1] * y_scale))
            x_max = int(np.round(box[2] * x_scale))
            y_max = int(np.round(box[3] * y_scale))
            scaled_boxes.append([x, y, x_max, y_max])
        return torch.as_tensor(scaled_boxes, dtype=torch.float32, device=custom_utils.DEVICE)

    def __load_image(self, index):
        """
        Load an image from the list of available images
        :param index: i of the wanted image
        :return: the image as a PIL.Image object
        """
        image_path = os.path.join(self.root, "images", self.images[index])
        return Image.open(image_path).convert("RGB")

    def __load_annotation(self, index):
        """
        Load image annotations from the list of available annotations files
        :param index: i of the wanted image
        :return: the annotations as a dict
        """
        annotation_path = os.path.join(self.root, "annotations", self.annotations[index])
        with open(annotation_path, "r") as fp:
            annotation_json = json.load(fp)
        return [value for key, value in annotation_json.items() if "item" in key]

    def __compute_objectness(self, boxes):
        target_matrix = np.zeros(49, dtype=np.float32).reshape(7, 7)
        coords = []
        square_length = np.round(self.size/7, 1)

        for box in boxes:
            box = box.tolist()
            box_center_x, box_center_y = np.round((box[2] - box[0]) / 2 + box[0], 1), np.round((box[3] - box[1]) / 2 + box[1], 1)
            box_center_x, box_center_y = math.floor(box_center_x / square_length), math.floor(box_center_y / square_length)
            target_matrix[box_center_y, box_center_x] = 1.0
            coords.append((box_center_x, box_center_y))

        return {"matrix": torch.as_tensor(target_matrix, dtype=torch.float32, device=custom_utils.DEVICE), "coords": coords}

    def __build_target_bb_mask(self, boxes):
        target_matrix = np.zeros(49*4, dtype=np.float32).reshape((7, 7, 4))
        square_length = np.round(self.size/7, 1)

        for box in boxes:
            box = box.tolist()
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            box_center_x = np.round(box[0] + box_w / 2, 1)
            box_center_y = np.round(box[1] + box_h / 2, 1)
            square_x, square_y = math.floor(box_center_x / square_length), math.floor(box_center_y / square_length)
            square_corner_x, square_corner_y = square_x * square_length, square_y * square_length
            box_center_x = (box_center_x - square_corner_x) / square_length
            box_center_y = (box_center_y - square_corner_y) / square_length
            box_w = box_w / self.size
            box_h = box_h / self.size
            target_matrix[square_y, square_x, 0] = box_center_x
            target_matrix[square_y, square_x, 1] = box_center_y
            target_matrix[square_y, square_x, 2] = box_w
            target_matrix[square_y, square_x, 3] = box_h
        return torch.as_tensor(target_matrix, dtype=torch.float32, device=custom_utils.DEVICE)

    def __build_target_labels_mask(self, coords, labels):
        target_matrix = np.zeros(49 * 13, dtype=np.float32).reshape((7, 7, 13))

        labels = labels.tolist()
        for index, label in enumerate(labels):
            target_matrix[coords[index][1], coords[index][0], label-1] = 1.

        return torch.as_tensor(target_matrix, dtype=torch.float32, device=custom_utils.DEVICE)

    def __generate_target(self, index):
        """
        Generate the target dict according to Torch specification
        :param index: i of the wanted annotations
        :return: target dict
        """
        annotations = self.__load_annotation(index)
        boxes = []
        labels = []
        categories = []

        for annotation in annotations:
            boxes.append(annotation["bounding_box"])
            labels.append(annotation["category_id"])
            categories.append(annotation['category_name'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=custom_utils.DEVICE)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=custom_utils.DEVICE)

        return {
            "boxes": boxes,
            "labels": labels,
            "categories": categories,
            "image_id": torch.tensor([index], device=custom_utils.DEVICE)
        }

    def __len__(self):
        return len(self.images)
