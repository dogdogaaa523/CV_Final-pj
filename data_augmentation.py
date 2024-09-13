import numpy as np
import torch

# CUTOUT 实现


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)  # 返回随机数/数组(整数)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)  # 截取函数
            y2 = np.clip(y + self.length // 2, 0, h)  # 用于截取数组中小于或者大于某值的部分，
            x1 = np.clip(x - self.length // 2, 0, w)  # 并使得被截取的部分等于固定的值
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)  # 数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变
        mask = mask.expand_as(img)  # 把一个tensor变成和函数括号内一样形状的tensor
        img = img * mask

        return img


# CUTMIX 实现


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""
    assert alpha > 0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# 在baseline的train部分做如下修改

for i, data in enumerate(trainloader, 0):
    length = len(trainloader)
    input, target = data
    # measure data loading time
    input = input.cuda()
    target = target.cuda()

    r = np.random.rand(1)
    if r < 0.5:
        # 1.设定lambda的值，服从beta分布
        lam = np.random.beta(1, 1)
        # 2.找到两个随机样本
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        # 3.生成裁剪区域B
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        # 4.将原有的样本A中的B区域，替换成样本B中的B区域
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # 5.根据裁剪区域坐标框的值调整lambda的值
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2])
        )
        # 6.将生成的新的训练样本丢到模型中进行训练
        output = net(input)
        optimizer.zero_grad()
        # 7.按lambda值分配权重
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (
            1.0 - lam
        )
        target = target.to(device)
    else:
        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()

        # forward + backward
        output = net(input)
        loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# MIXUP 实现


def mixup_data(x, y, alpha=1, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


use_cuda = True

# 在baseline的train部分做如下修改


for i, data in enumerate(trainloader, 0):
    length = len(trainloader)
    inputs, targets = data
    if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
    optimizer.zero_grad()
    inputs, targets_a, targets_b = map(Variable, [inputs, targets_a, targets_b])
    outputs = net(inputs)

    loss_func = mixup_criterion(targets_a, targets_b, lam)
    loss = loss_func(criterion, outputs)
    loss.backward()
    optimizer.step()
