import matplotlib.pyplot as plt
import torch
from torchvision.transforms import functional as F

from utils.coco_utils import CocoKeypoint
from utils.utils import plot_keypoints


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class ToTensor:
    def __call__(self, image, target=None):
        image = F.pil_to_tensor(image)
        return image, target


class ToDtype:
    def __call__(self, image, target=None):
        image = F.convert_image_dtype(image, dtype=torch.float)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, image, target=None):
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


def flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


transform = Compose([
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    ToDtype(),
])

transform_val = Compose([
    ToTensor(),
    ToDtype(),
])

if __name__ == '__main__':
    train_dataset = CocoKeypoint(root="../coco/val2017",
                                 annFile="../coco/annotations/person_keypoints_val2017.json",
                                 min_keypoints_per_image=11,
                                 transform=transform)
    image, targets = train_dataset[0]

    plt.imshow(F.to_pil_image(image))
    for box in targets["boxes"]:
        plt.scatter(box[0], box[1], color='blue', marker='o', s=4)
        plt.scatter(box[2], box[3], color='blue', marker='o', s=4)
    plot_keypoints(targets["keypoints"])
    plt.show()
