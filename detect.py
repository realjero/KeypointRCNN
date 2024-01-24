import argparse
import random

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib import patches
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from utils.transforms import transform_val
from utils.utils import device, plot_keypoints

SCORE_THRESHOLD = 0.8
KEYPOINT_THRESHOLD = 0.8
connections = [(0, 1), (0, 2), (1, 3), (2, 4), (6, 5), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (12, 11),
                   (11, 13), (13, 15), (12, 14), (14, 16)]
def webcam(model):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor, target = transform_val(pil_image)

        output = model(image_tensor.to(device).unsqueeze(0))[0]
        output = {k: v.to("cpu") for k, v in output.items()}

        keypoints = output["keypoints"]
        keypoints_scores = output["keypoints_scores"]
        boxes = output["boxes"]
        scores = output["scores"]

        for box, score, kpts, kpts_scores in zip(boxes, scores, keypoints, keypoints_scores):
            if score > SCORE_THRESHOLD:
                # Draw rectangle
                box_color = (255, 0, 0)  # Red color
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 1)

                # Draw keypoints and connections
                for p1, p2 in connections:
                    if kpts_scores[p1] > KEYPOINT_THRESHOLD and kpts_scores[p2] > KEYPOINT_THRESHOLD:
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        x1, y1, _ = kpts[p1]
                        x2, y2, _ = kpts[p2]
                        cv2.circle(frame, (int(x1), int(y1)), 2, color, -1)
                        cv2.circle(frame, (int(x2), int(y2)), 2, color, -1)
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

        cv2.imshow('Webcam', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def one_image(model, file):
    image = Image.open(file)
    image_tensor, _ = transform_val(image)

    output = model(image_tensor.to(device).unsqueeze(0))[0]
    output = {k: v.to("cpu") for k, v in output.items()}

    boxes = output["boxes"]
    scores = output["scores"]
    keypoints = output["keypoints"]
    keypoints_scores = output["keypoints_scores"]

    fig, ax = plt.subplots()
    ax.imshow(image)
    for box, score, keypoints, kpts_scores in zip(boxes, scores, keypoints, keypoints_scores):
        if score > SCORE_THRESHOLD:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            for p1, p2 in connections:
                if kpts_scores[p1] > KEYPOINT_THRESHOLD and kpts_scores[p2] > KEYPOINT_THRESHOLD:
                    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                    x1, y1, _ = keypoints[p1]
                    x2, y2, _ = keypoints[p2]
                    ax.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color=color, linewidth=1, markersize=2)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument("--file", required=False, help="Path to the input file.")
    args = parser.parse_args()

    model = keypointrcnn_resnet50_fpn(weights="DEFAULT").to(device)  # TODO: SET TO CORRECT MODEL
    model.eval()
    # model.load_state_dict(torch.load("checkpoint_10.pth"))

    with torch.no_grad():
        if not args.file:
            print("Press q to close webcam")
            webcam(model)
        else:
            one_image(model, args.file)