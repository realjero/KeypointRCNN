import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from dataset import CocoWholeBody
from engine import train_one_epoch

EPOCHS = 42
BATCH_SIZE = 8
NUM_KEYPOINTS = 133

# OPTIMIZER
LEARN_RATE = 0.02
MOMENTUM = 0.9

# SCHEDULAR
WEIGHT_DECAY = 1e-4
MILESTONES = (36, 43)
GAMMA = 0.1


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    # Check if CUDA (GPU) is available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Torch device={device}")

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    transform = weights.transforms()

    dataset = CocoWholeBody(root="./coco/train2017",
                            annFile="./coco/annotations/coco_wholebody_train_v1.0.json",
                            min_keypoints_per_image=55,
                            transform=transform)

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = keypointrcnn_resnet50_fpn(num_keypoints=NUM_KEYPOINTS).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=LEARN_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    status_bar = tqdm(range(0, EPOCHS))

    for epoch in status_bar:
        train_one_epoch(model=model,
                        optimizer=optimizer,
                        data_loader=data_loader,
                        device=device,
                        epoch=epoch,
                        status_bar=status_bar,
                        print_freq=20)
        lr_scheduler.step()

        # TODO: Evaluate
        # https://github.com/alexppppp/keypoint_rcnn_training_pytorch/blob/main/train.py

    # save model
    torch.save(model.state_dict(), f'e{EPOCHS}_b{BATCH_SIZE}_lr{LEARN_RATE}_m{MOMENTUM}.pth')
