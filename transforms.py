import torch
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        return image, target


class ToDtype:
    def __call__(self, image, target):
        image = F.convert_image_dtype(image, dtype=torch.float)
        return image, target
