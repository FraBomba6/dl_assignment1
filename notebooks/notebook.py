import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from PIL import Image

# %%
# Initializing torch device according to hardware available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# %%
class CustomDataset(Dataset):
    """
    Class that represents a dataset object to use as input on a CNN
    """
    def __init__(self, root, transforms=None):
        """
        Default initializer
        :param root: path to dataset root
        :param transforms: optional list of transforms
        """
        self.root = root

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
        return self.__load_image(index), self.__generate_target(index)

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
            fp.close()
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
        boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=device)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], device=device),
            "area": area,
            "isCrowd": isCrowd
        }

    def __len__(self):
        return len(self.images)


#%%
dataset = CustomDataset("../data/assignment_1/train")
