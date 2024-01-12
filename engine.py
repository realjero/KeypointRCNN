import math
import sys

from torch.optim.lr_scheduler import LinearLR


def train_one_epoch(model, optimizer, data_loader, device, epoch, status_bar, print_freq=10, sched=None):
    model.train()

    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        sched = LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

    # TODO: Create Intercept to log metrics
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not math.isfinite(losses):
            print(f"Loss is {losses}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        status_bar.set_description(f"Loss: {losses:.5f}")

        if sched is not None:
            sched.step()
