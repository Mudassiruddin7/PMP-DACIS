"""ResNet-18 with block-internal channel pruning.

Only the *mid* channels of each BasicBlock are pruned (conv1 outputs, bn1, conv2 inputs).
The residual stream keeps its width (64/128/256/512), so shortcuts never need surgery. This is
the scheme He et al. (ICCV 2017) use for residual branches. Module names match torchvision's
ResNet-18, so ImageNet weights load directly.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE_WIDTHS = (64, 128, 256, 512)


class Block(nn.Module):
    def __init__(self, cin, cmid, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cmid, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cmid)
        self.conv2 = nn.Conv2d(cmid, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.downsample = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.capture = False   # when True, keep the post-ReLU mid activation (and its grad)
        self.mid = None
        self.mask = None       # optional [cmid] 0/1 tensor, used only by the surgery unit test

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        h = F.relu(self.bn1(self.conv1(x)))
        if self.mask is not None:
            h = h * self.mask.view(1, -1, 1, 1).to(h.dtype)
        if self.capture:
            self.mid = h
            if h.requires_grad:
                h.retain_grad()
        out = self.bn2(self.conv2(h))
        return F.relu(out + idt)


class PResNet18(nn.Module):
    def __init__(self, in_ch=3, mids=None, num_classes=0):
        super().__init__()
        self.in_ch = in_ch
        full = [64, 64, 128, 128, 256, 256, 512, 512]
        mids = list(mids) if mids is not None else full
        self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        layers, cin, b = [], 64, 0
        for s, w in enumerate(STAGE_WIDTHS):
            blks = []
            for j in range(2):
                stride = 2 if (s > 0 and j == 0) else 1
                blks.append(Block(cin, mids[b], w, stride))
                cin = w
                b += 1
            layers.append(nn.Sequential(*blks))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(512, num_classes) if num_classes else None
        self.log_tau = nn.Parameter(torch.tensor(2.302585))  # log(10); temperature of the cosine classifier

    @property
    def blocks(self):
        return [blk for layer in (self.layer1, self.layer2, self.layer3, self.layer4) for blk in layer]

    def features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        f = self.features(x)
        return self.fc(f) if self.fc is not None else f

    def mids(self):
        return [b.conv1.out_channels for b in self.blocks]


def load_imagenet_weights(model: PResNet18):
    """Load torchvision ImageNet-1k (V1) weights. For in_ch != 3, repeat the RGB filters cyclically
    and scale by 3/in_ch (standard recipe for multispectral input)."""
    import torchvision
    tv = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    sd = {k: v for k, v in tv.state_dict().items() if not k.startswith("fc.")}
    if model.in_ch != 3:
        w = sd["conv1.weight"]                                   # [64,3,7,7]
        reps = [w[:, i % 3] for i in range(model.in_ch)]
        sd["conv1.weight"] = torch.stack(reps, 1) * (3.0 / model.in_ch)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not (k.startswith("fc.") or k == "log_tau")]
    assert not bad and not unexpected, (bad, unexpected)
    return model


def make_model(in_ch=3, pretrained=True):
    m = PResNet18(in_ch=in_ch)
    return load_imagenet_weights(m) if pretrained else m


def count_params(model):
    """Parameters of the feature extractor (classifier head and temperature excluded)."""
    return sum(p.numel() for n, p in model.named_parameters() if not (n.startswith("fc.") or n == "log_tau"))


def block_specs(model):
    """[(cin, cout, mid)] for every block, read from the live model."""
    return [(b.conv1.in_channels, b.conv2.out_channels, b.conv1.out_channels) for b in model.blocks]


def params_for_mids(model_template: PResNet18, mids):
    """Exact parameter count if block b had mids[b] mid channels (analytic, no model copy)."""
    total = count_params(model_template)
    for blk, new_m in zip(model_template.blocks, mids):
        cin, cout, m = blk.conv1.in_channels, blk.conv2.out_channels, blk.conv1.out_channels
        per = lambda mm: cin * mm * 9 + 2 * mm + mm * cout * 9
        total += per(new_m) - per(m)
    return total


@torch.no_grad()
def prune_model(model: PResNet18, keep):
    """Return a new model whose block b keeps the mid channels `keep[b]` (sorted LongTensor)."""
    new = copy.deepcopy(model)
    for blk, idx in zip(new.blocks, keep):
        idx = idx.to(blk.conv1.weight.device)
        if len(idx) == blk.conv1.out_channels:
            continue
        dev, k = blk.conv1.weight.device, len(idx)
        c1 = nn.Conv2d(blk.conv1.in_channels, k, 3, blk.conv1.stride, 1, bias=False).to(dev)
        c1.weight.copy_(blk.conv1.weight[idx])
        bn = nn.BatchNorm2d(k).to(dev)
        for n in ("weight", "bias", "running_mean", "running_var"):
            getattr(bn, n).copy_(getattr(blk.bn1, n)[idx])
        bn.num_batches_tracked.copy_(blk.bn1.num_batches_tracked)
        c2 = nn.Conv2d(k, blk.conv2.out_channels, 3, 1, 1, bias=False).to(dev)
        c2.weight.copy_(blk.conv2.weight[:, idx])
        blk.conv1, blk.bn1, blk.conv2 = c1, bn, c2
    return new


@torch.no_grad()
def count_macs(model, res, in_ch=None):
    """Multiply-accumulates of one forward pass at batch size 1 (convs + fc), counted with hooks."""
    in_ch = in_ch or model.in_ch
    total = [0]

    def hook(m, i, o):
        if isinstance(m, nn.Conv2d):
            total[0] += o.shape[2] * o.shape[3] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    model.features(torch.zeros(1, in_ch, res, res, device=dev))
    model.train(was_training)
    for h in hs:
        h.remove()
    return total[0]


class CompactNet(nn.Module):
    """Off-the-shelf small ImageNet network (torchvision) used as a baseline: trunk features, global average pool,
    cosine-prototype evaluation like everything else. It has no prunable blocks."""

    def __init__(self, arch):
        super().__init__()
        import torchvision
        weights = torchvision.models.get_model_weights(arch).IMAGENET1K_V1
        tv = getattr(torchvision.models, arch)(weights=weights)
        self.body = tv.features
        self.feat_dim = {"mobilenet_v3_small": 576, "mobilenet_v2": 1280, "efficientnet_b0": 1280}[arch]
        self.in_ch = 3
        self.fc = None
        self.log_tau = nn.Parameter(torch.tensor(2.302585))

    @property
    def blocks(self):
        return []

    def features(self, x):
        return F.adaptive_avg_pool2d(self.body(x), 1).flatten(1)

    def forward(self, x):
        f = self.features(x)
        return self.fc(f) if self.fc is not None else f

    def mids(self):
        return []


def make_compact(arch):
    return CompactNet(arch)
