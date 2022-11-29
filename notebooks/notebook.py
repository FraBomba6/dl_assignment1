# %%
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from PIL import Image # module
from PIL.Image import Image as PilImage # object
from random import randint

# Defining project root in order to avoid relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initializing torch device according to hardware available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
class ITransform(ABC):

    @abstractmethod
    def __call__(self, input: PilImage) -> PilImage:
        pass     
    

class CustomDataset(Dataset):
    """
    Class that represents a dataset object to use as input on a CNN
    """
    def __init__(self, root, transforms: list[ITransform] = []):
        """
        Default initializer
        :param root: path to dataset root
        :param transforms: optional list of transforms
        """
        self.root = root

        self.transforms = torchvision.transforms.Compose(transforms)

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
            img, target = self.transforms(img, target)
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

        
class TestTransform(ITransform):

    def __init__(self, size: tuple[int, int] | int) -> None:
        self.size = size 
    
    def __call__(self, input: PilImage | torch.Tensor) -> PilImage | torch.Tensor:
        return TF.resize(input, size=self.size)


# %%
dataset = CustomDataset(os.path.join(PROJECT_ROOT, "data", "assignment_1", "train"))

# testing
scale = TestTransform((200,200))
image, label = dataset[randint(0, len(dataset))]
scale(image).show()
