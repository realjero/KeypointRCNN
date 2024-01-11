import os
from typing import List, Any, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


class CocoKeypoint:
    def __init__(self, root, annFile, transform=None, target_transform=None):
        # TODO: Filter without bbox, less than x keypoints or none
        self.dataset = CocoDetection(root=root, annFile=annFile)

        # Create augmentations for image and targets

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, targets = self.dataset[index]

        boxes = []
        labels = []
        keypoints = []
        for target in targets:
            # Transform XYWH to XYXY
            bbox = target["bbox"]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

            labels.append(target['category_id'])

            flat_kpts = target["keypoints"]
            keypoints.append([[flat_kpts[i], flat_kpts[i + 1], flat_kpts[i + 2]] for i in range(0, len(flat_kpts), 3)])

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        keypoints = torch.tensor(keypoints, dtype=torch.float)

        targets = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints
        }

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return image, targets


class CocoWholeBody(Dataset):
    def __init__(self, root: str, annFile: str, transform=None, target_transform=None) -> None:
        from xtcocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.root = root

        # TODO: Create augmentations for image and targets
        self.transform = transform
        self.target_transform = target_transform

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        def _kpts_to_matrix(kpts):
            return [[kpts[i], kpts[i + 1], kpts[i + 2]] for i in range(0, len(kpts), 3)]

        image_id = self.ids[index]
        image = self._load_image(image_id)
        targets = self._load_target(image_id)

        boxes = []
        labels = []
        keypoints = []
        for target in targets:
            # Transform XYWH to XYXY
            bbox = target["bbox"]
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

            labels.append(target['category_id'])

            body_kpts = target["keypoints"]
            foot_kpts = target["foot_kpts"]
            face_kpts = target["face_kpts"]
            lefthand_kpts = target["lefthand_kpts"]
            righthand_kpts = target['righthand_kpts']

            kpts = []
            kpts.extend(body_kpts)
            kpts.extend(foot_kpts)
            kpts.extend(face_kpts)
            kpts.extend(lefthand_kpts)
            kpts.extend(righthand_kpts)

            keypoints.append(_kpts_to_matrix(kpts))

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        keypoints = torch.tensor(keypoints, dtype=torch.float)

        targets = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints
        }

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return image, targets

    def __len__(self) -> int:
        return len(self.ids)
