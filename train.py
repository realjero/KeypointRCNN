import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from tqdm import tqdm

from utils.coco_utils import CocoKeypoint
from utils.engine import train_one_epoch
from utils.transforms import transform, transform_val
from utils.utils import device

EPOCHS = 42
BATCH_SIZE = 8
NUM_KEYPOINTS = 17

# OPTIMIZER
LEARN_RATE = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# SCHEDULAR
MILESTONES = (36, 43)
GAMMA = 0.1


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    train_dataset = CocoKeypoint(root="./coco/train2017",
                                 annFile="./coco/annotations/person_keypoints_train2017.json",
                                 min_keypoints_per_image=11,
                                 train=True,
                                 transform=transform)

    # This was a dumb idea train/val/test
    val_dataset = CocoKeypoint(root="./coco/val2017",
                               annFile="./coco/annotations/person_keypoints_val2017.json",
                               min_keypoints_per_image=1,
                               transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = keypointrcnn_resnet50_fpn(num_keypoints=NUM_KEYPOINTS).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=LEARN_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_scheduler = MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

    status_bar = tqdm(range(0, EPOCHS))

    train_loss = []
    val_loss = []

    for epoch in status_bar:
        train, val = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     status_bar=status_bar,
                                     print_freq=20)
        lr_scheduler.step()

        train_loss.append(train)
        val_loss.append(val)

        torch.save(model.state_dict(), f'./temp/checkpoint_{epoch}.pth')

    # save model
    torch.save(model.state_dict(), f'./e{EPOCHS}_b{BATCH_SIZE}_lr{LEARN_RATE}_m{MOMENTUM}.pth')

    plt.plot(range(0, EPOCHS), train_loss, label='Train Loss')
    plt.plot(range(0, EPOCHS), val_loss, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"e{EPOCHS}_b{BATCH_SIZE}_lr{LEARN_RATE}_m{MOMENTUM}.png")
    plt.show()
