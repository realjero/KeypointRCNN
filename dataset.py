import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

from utils import has_valid_annotation


class CocoKeypoint(Dataset):
    def __init__(self, root: str, annFile: str, min_keypoints_per_image=None, transform=None):
        from pycocotools.coco import COCO
        self.dataset = CocoDetection(root=root, annFile=annFile)
        self.transform = transform
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        if min_keypoints_per_image is not None:
            print("Removing targets without annotations...")
            filtered_ids = []
            for img_id in self.ids:
                annotations = self._load_target(img_id)
                if has_valid_annotation(annotations, min_keypoints_per_image):
                    filtered_ids.append(img_id)
            self.ids = filtered_ids
            print(f"{len(self.ids)} targets remain")

        self.transform = transform

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        def _kpts_to_matrix(kpts):
            return [[kpts[i], kpts[i + 1], kpts[i + 2]] for i in range(0, len(kpts), 3)]

        image_id = self.ids[index]
        image = self._load_image(image_id)
        targets = self._load_target(image_id)

        boxes = []
        labels = []
        keypoints = []
        for target in targets:
            bbox = target["bbox"]
            label = target["category_id"]
            body_kpts = target["keypoints"]

            # XYWH to XYXY
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(label)
            keypoints.append(_kpts_to_matrix(body_kpts))

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        keypoints = torch.tensor(keypoints, dtype=torch.float)

        targets = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints
        }

        if self.transform is not None:
            image, targets = self.transform(image, targets)

        return image, targets

    def __len__(self) -> int:
        return len(self.ids)
