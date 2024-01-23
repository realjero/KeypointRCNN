import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F

from dataset import CocoKeypoint
from utils import flip_coco_person_keypoints, plot_keypoints


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


class RandomHorizontalFlip:
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F.get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


if __name__ == '__main__':
    transform = Compose([
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        ToDtype(),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CocoKeypoint(root="./coco/val2017",
                                 annFile="./coco/annotations/person_keypoints_val2017.json",
                                 min_keypoints_per_image=11,
                                 transform=transform)
    image, targets = train_dataset[0]

    plt.imshow(F.to_pil_image(image))
    for box in targets["boxes"]:
        plt.scatter(box[0], box[1], color='blue', marker='o', s=4)
        plt.scatter(box[2], box[3], color='blue', marker='o', s=4)
    plot_keypoints(targets["keypoints"])
    plt.show()
