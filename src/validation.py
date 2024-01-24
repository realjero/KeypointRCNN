import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from tqdm import tqdm

from train import collate_fn
from utils.coco_utils import CocoKeypoint, CocoEvaluator
from utils.transforms import transform_val
from utils.utils import device

if __name__ == '__main__':
    val_dataset = CocoKeypoint(root="../coco/val2017",
                               annFile="../coco/annotations/person_keypoints_val2017.json",
                               transform=transform_val)

    val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

    model = keypointrcnn_resnet50_fpn(weights="DEFAULT").to(device)
    model.eval()

    evaluator = CocoEvaluator(val_dataset.coco, ["bbox", "keypoints"])

    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = list(image.to(device) for image in images)
            output = model(images)

            for output, target in zip(output, targets):
                evaluator.add(output, target["image_id"])

        evaluator.evaluate()
