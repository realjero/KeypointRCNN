import argparse
import random

import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
from matplotlib import patches
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from utils.transforms import transform_val
from utils.utils import device

SCORE_THRESHOLD = 0.9
KEYPOINT_THRESHOLD = 0.9
connections = [(0, 1), (0, 2), (1, 3), (2, 4), (6, 5), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (12, 11),
               (11, 13), (13, 15), (12, 14), (14, 16)]


def keypoints_on_webcam(model):
    cap = cv2.VideoCapture(0)

    # Set the desired frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Continuously capture frames from the webcam and process them
    while True:
        ret, frame = cap.read()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform_val(pil_image)

        # Move the image tensor to the specified device and add a batch dimension
        output = model(image_tensor.to(device).unsqueeze(0))[0]
        output = {k: v.to("cpu") for k, v in output.items()}

        keypoints = output["keypoints"]
        keypoints_scores = output["keypoints_scores"]
        boxes = output["boxes"]
        scores = output["scores"]

        # Iterate through detected objects and draw bounding boxes and keypoints
        for box, score, kpts, kpts_scores in zip(boxes, scores, keypoints, keypoints_scores):
            if score > SCORE_THRESHOLD:
                # Draw a bounding box around the detected object
                box_color = (255, 0, 0)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 1)

                # Draw lines connecting keypoints if their scores are above the threshold
                for p1, p2 in connections:
                    if kpts_scores[p1] > KEYPOINT_THRESHOLD and kpts_scores[p2] > KEYPOINT_THRESHOLD:
                        # Generate a random color for each line
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


def keypoints_on_image(model, file):
    image = Image.open(file)
    image_tensor, _ = transform_val(image)

    # Move the image tensor to the specified device and add a batch dimension
    output = model(image_tensor.to(device).unsqueeze(0))[0]
    output = {k: v.to("cpu") for k, v in output.items()}

    # Extract relevant information from the output
    boxes = output["boxes"]
    scores = output["scores"]
    keypoints = output["keypoints"]
    keypoints_scores = output["keypoints_scores"]

    # Create a subplot for displaying the image
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Iterate through detected objects and draw bounding boxes and keypoints
    for box, score, keypoints, kpts_scores in zip(boxes, scores, keypoints, keypoints_scores):
        if score > SCORE_THRESHOLD:
            # Draw a bounding box around the detected object
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Draw lines connecting keypoints if their scores are above the threshold
            for p1, p2 in connections:
                if kpts_scores[p1] > KEYPOINT_THRESHOLD and kpts_scores[p2] > KEYPOINT_THRESHOLD:
                    # Generate a random color for each line
                    color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                    x1, y1, _ = keypoints[p1]
                    x2, y2, _ = keypoints[p2]
                    ax.plot([x1, x2], [y1, y2], marker='o', linestyle='-', color=color, linewidth=1, markersize=2)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a file.")
    parser.add_argument("--file", required=False, help="Path to the input file.")
    args = parser.parse_args()

    model = keypointrcnn_resnet50_fpn().to(device)
    model.load_state_dict(torch.load("checkpoint_33.pth"))
    model.eval()

    with torch.no_grad():
        if not args.file:
            print("Press q to close webcam")
            keypoints_on_webcam(model)
        else:
            keypoints_on_image(model, args.file)
