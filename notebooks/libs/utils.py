import os
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

__device = None
__project_root = None

def __get_device():
    global __device
    if not __device:
        __device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return __device

def __get_proj_root():
    global __project_root
    if not __project_root:
        __project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return __project_root

# Defining project root in order to avoid relative paths
PROJECT_ROOT = __get_proj_root()

# Initializing torch device according to hardware available
DEVICE =  __get_device()

# Size for all the images
IMG_SIZE = 128 

NUM_CATEGORIES = 13

CATEGORY_MAPPING = {
    "short sleeve top":      0,
    "long sleeve top":       1,
    "short sleeve outwear":  2,
    "long sleeve outwear":   3,
    "vest":                  4,
    "sling":                 5,
    "shorts":                6,
    "trousers":              7,
    "skirt":                 8,
    "short sleeve dress":    9,
    "long sleeve dress":     10,
    "vest dress":            11, 
    "sling dress":           12
}



def collate_fn(batch):
    """
    Function to combine images, boxes and labels
    :param batch: an iterable of N sets from __getitem__() of CustomDataset
    :return: a tensor of images, lists of varying-size tensors of bounding boxes and labels
    """

    targets = list()

    images = list()
    boxes = list()
    labels = list()
    objectness = list()

    for b in batch:
        images.append(b[0])
        boxes.append(b[1]["boxes"])
        labels.append(b[1]["labels"])
        objectness.append(b[1]["objectness"])

    images = torch.stack(images, dim=0)

    return images, boxes, labels, objectness


def with_bounding_box(image, target):
    """
    Returns an image with bounding boxes and labels
    :param image: image as Tensor
    :param target: dict representing containing the bounding boxes
    """
    tensor_image = torchvision.utils.draw_bounding_boxes(transforms.PILToTensor()(transforms.ToPILImage()(image)), target['bounding_box'], target['category_name'], colors="red", width=2)
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
