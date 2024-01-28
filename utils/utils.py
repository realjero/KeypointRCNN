import cv2
import torch
from matplotlib import pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def plot_keypoints(kpts):
    for p in kpts:
        for kpt in p:
            x, y, v = kpt
            if v >= 0.9:
                plt.scatter(x, y, color='green', marker='o', s=1)


def keypoints_on_cv2(frame, output, connections, SCORE_THRESHOLD, KEYPOINT_THRESHOLD):
    keypoints = output["keypoints"]
    keypoints_scores = output["keypoints_scores"]
    boxes = output["boxes"]
    scores = output["scores"]

    # Iterate through detected objects and draw bounding boxes and keypoints
    for box, score, kpts, kpts_scores in zip(boxes, scores, keypoints, keypoints_scores):
        if score > SCORE_THRESHOLD:
            # Draw a bounding box around the detected object
            # box_color = (255, 0, 0)
            # cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 1)

            # Draw lines connecting keypoints if their scores are above the threshold
            for p1, p2 in connections:
                if kpts_scores[p1] > KEYPOINT_THRESHOLD and kpts_scores[p2] > KEYPOINT_THRESHOLD:
                    # Generate a random color for each line
                    color = (0, 255, 0)
                    x1, y1, _ = kpts[p1]
                    x2, y2, _ = kpts[p2]
                    cv2.circle(frame, (int(x1), int(y1)), 2, color, -1)
                    cv2.circle(frame, (int(x2), int(y2)), 2, color, -1)
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    return frame
