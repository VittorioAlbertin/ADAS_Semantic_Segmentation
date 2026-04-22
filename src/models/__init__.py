from .unet import UNet
from .deeplab import DeepLabV3Plus
from .segformer import SegFormer

def get_model(name, num_classes=19):
    if name == 'unet':
        return UNet(n_classes=num_classes)
    elif name == 'deeplab':
        return DeepLabV3Plus(num_classes=num_classes, pretrained_backbone=True)
    elif name == 'segformer':
        return SegFormer(num_classes=num_classes, pretrained_backbone=True)
    else:
        raise ValueError(f"Model {name} not implemented yet.")
