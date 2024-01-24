import cv2
from PIL import Image
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

from utils.transforms import transform_val
from utils.utils import device

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT

    model = keypointrcnn_resnet50_fpn(weights="DEFAULT").to(device)  # TODO: SET TO CORRECT MODEL
    model.eval()
    # model.load_state_dict(torch.load("checkpoint_10.pth"))

    while True:
        ret, frame = cap.read()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor, target = transform_val(pil_image)

        pred = model(image_tensor.to(device).unsqueeze(0))

        keypoints = pred[0]["keypoints"].cpu().detach().numpy()
        keypoints_scores = pred[0]["keypoints_scores"].cpu().detach().numpy()
        boxes = pred[0]["boxes"].cpu().detach().numpy()
        scores = pred[0]["scores"].cpu().detach().numpy()

        for box, score, kpts, kpts_scores in zip(boxes, scores, keypoints, keypoints_scores):
            if score > 0.5:
                for point, k_score in zip(kpts, kpts_scores):
                    x, y, v = point
                    if k_score > 0.5:
                        cv2.circle(frame, (x.astype(int), y.astype(int)), 1, (0, 255, 0), - 1)

        cv2.imshow('Webcam', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
