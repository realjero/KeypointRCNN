import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


class CocoKeypoint(Dataset):
    def __init__(self, root: str, annFile: str, min_keypoints_per_image=None, train=False, transform=None):
        from pycocotools.coco import COCO
        self.dataset = CocoDetection(root=root, annFile=annFile)
        self.transform = transform
        self.root = root
        self.coco = COCO(annFile)
        self.train = train
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

    def _load_image(self, idx: int):
        path = self.coco.loadImgs(idx)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, idx: int):
        return self.coco.loadAnns(self.coco.getAnnIds(idx))

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
        image_id = torch.tensor(image_id, dtype=torch.int)

        targets = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints
        }

        if not self.train:
            targets["image_id"] = image_id

        if self.transform is not None:
            image, targets = self.transform(image, targets)

        return image, targets

    def __len__(self) -> int:
        return len(self.ids)


class CocoEvaluator:
    def __init__(self, cocoGt, iou_types):
        self.cocoGt = cocoGt
        self.iou_types = iou_types
        self.results = []
        self.imgIds = []

    def add(self, prediction, image_id):
        self.imgIds.append(image_id)

        boxes = prediction["boxes"].to("cpu")
        scores = prediction["scores"].to("cpu")
        keypoints = prediction["keypoints"].to("cpu")

        for box, score, kpts in zip(boxes, scores, keypoints):
            self.results.append({
                "image_id": image_id,
                "category_id": 1,
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "keypoints": kpts.reshape(-1),
                "score": score
            })

    def evaluate(self):
        cocoDt = COCO.loadRes(self.cocoGt, self.results)
        for iou_type in self.iou_types:
            evalCoco = COCOeval(self.cocoGt, cocoDt, iou_type)
            evalCoco.params.imgIds = np.unique(self.imgIds)
            evalCoco.evaluate()
            evalCoco.accumulate()
            evalCoco.summarize()


def has_valid_annotation(anno, min_keypoints_per_image):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False
