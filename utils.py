from matplotlib import pyplot as plt
from torch.utils.data import Subset
from tqdm import tqdm


def plot_keypoints(kpts):
    for p in kpts:
        for kpt in p:
            x, y, v = kpt
            if v >= 0.9:
                plt.scatter(x, y, color='blue', marker='o', s=1)


def _coco_remove_images_without_annotations(dataset):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj[2:]) for obj in anno["boxes"])

    def _count_visible_keypoints(anno):
        return sum(sum(1 for x, y, v in ann if v > 0) for ann in anno["keypoints"])

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        # if "keypoints" not in anno:
        #     return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    print("Removing targets without annotations...")
    status_data = tqdm(enumerate(dataset), desc="Processing", total=len(dataset), unit="targets")
    ids = []
    for i, (image, targets) in status_data:
        if _has_valid_annotation(targets):
            ids.append(i)

    dataset = Subset(dataset, ids)
    print(f"{len(dataset)} targets remain")
    return dataset
