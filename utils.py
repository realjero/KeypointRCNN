from matplotlib import pyplot as plt


def plot_keypoints(kpts):
    for p in kpts:
        for kpt in p:
            x, y, v = kpt
            if v >= 0.9:
                plt.scatter(x, y, color='green', marker='o', s=0.1)


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
