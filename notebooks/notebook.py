# %%
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from PIL import Image  # module
from PIL.Image import Image as PilImage  # object
from random import randint
import matplotlib.pyplot as plt
import numpy as np

# Defining project root in order to avoid relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initializing torch device according to hardware available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
class CustomDataset(Dataset):
    """
    Class that represents a dataset object to use as input on a CNN
    """
    def __init__(self, root, transformation=None):
        """
        Default initializer
        :param root: path to dataset root
        :param transformation: optional list of transforms
        """
        self.root = root

        if transformation is not None:
            self.transforms = transforms.Compose(transformation)
        else:
            self.transforms = None

        # Load images filelist
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        # Load annotations filelist
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))

    def __getitem__(self, index):
        """
        Default getter for dataset objects
        :param index: index of the wanted image + annotation
        :return: image as PIL Image and target dictionary
        """
        img = self.__load_image(index)
        target = self.__generate_target(index)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __load_image(self, index):
        """
        Load an image from the list of available images
        :param index: index of the wanted image
        :return: the image as a PIL.Image object
        """
        image_path = os.path.join(self.root, "images", self.images[index])
        return Image.open(image_path).convert("RGB")

    def __load_annotation(self, index):
        """
        Load image annotations from the list of available annotations files
        :param index: index of the wanted image
        :return: the annotations as a dict
        """
        annotation_path = os.path.join(self.root, "annotations", self.annotations[index])
        with open(annotation_path, "r") as fp:
            annotation_json = json.load(fp)
        return [value for key, value in annotation_json.items() if "item" in key]

    def __generate_target(self, index):
        """
        Generate the target dict according to Torch specification
        :param index: index of the wanted annotations
        :return: target dict
        """
        annotations = self.__load_annotation(index)
        boxes = []
        labels = []
        isCrowd = torch.zeros((len(annotations),), dtype=torch.int64)
        for annotation in annotations:
            boxes.append(annotation["bounding_box"])
            labels.append(annotation["category_id"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=DEVICE)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=DEVICE)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], device=DEVICE),
            "area": area,
            "isCrowd": isCrowd
        }

    def __len__(self):
        return len(self.images)


# %%
t = [transforms.Resize((225, 225))]
dataset = CustomDataset(os.path.join(PROJECT_ROOT, "data", "assignment_1", "train"), t)

image, target = dataset[randint(0, len(dataset))]

image.show()

# %%
dataset = CustomDataset(os.path.join(PROJECT_ROOT, "data", "assignment_1", "train"))
aspect_ratios = []
for i in range(len(dataset)):
    img, target = dataset[i]
    sizes = img.size
    aspect_ratios.append(sizes[0]/sizes[1])

# %%
plt.bar(*np.unique(aspect_ratios, return_counts=True))
plt.show()