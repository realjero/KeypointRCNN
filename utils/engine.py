import math
import sys

import torch
from torch.optim.lr_scheduler import LinearLR


def train_one_epoch(model, optimizer, train_loader, val_loader, device, epoch, status_bar, print_freq=10, sched=None):
    model.train()

    train_loss = 0.0
    val_loss = 0.0

    # if epoch == 0:
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(train_loader) - 1)
    #
    #     sched = LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    #
    # for i, (images, targets) in enumerate(train_loader):
    #     images = list(image.to(device) for image in images)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    #
    #     loss_dict = model(images, targets)
    #     losses = sum(loss for loss in loss_dict.values())
    #     train_loss += losses
    #
    #     if not math.isfinite(losses):
    #         print(f"Loss is {losses}, stopping training")
    #         sys.exit(1)
    #
    #     optimizer.zero_grad()
    #     losses.backward()
    #     optimizer.step()
    #
    #     if i % print_freq == 0:
    #         loss_classifier = loss_dict["loss_classifier"]
    #         loss_box_reg = loss_dict["loss_box_reg"]
    #         loss_keypoint = loss_dict["loss_keypoint"]
    #         loss_objectness = loss_dict["loss_objectness"]
    #         loss_rpn_box_reg = loss_dict["loss_rpn_box_reg"]
    #
    #         status_bar.set_description(f"{losses=:.5f} "
    #                                    f"{loss_classifier=:.5f} "
    #                                    f"{loss_box_reg=:.5f} "
    #                                    f"{loss_keypoint=:.5f} "
    #                                    f"{loss_objectness=:.5f} "
    #                                    f"{loss_rpn_box_reg=:.5f}")
    #
    #     if sched is not None:
    #         sched.step()

    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses

    train_loss = train_loss.cpu().detach().numpy() / len(train_loader)
    val_loss = val_loss.cpu().detach().numpy() / len(val_loader)

    return train_loss, val_loss
