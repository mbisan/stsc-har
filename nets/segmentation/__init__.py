from nets.segmentation.deeplabv3p1d import get_model
from nets.segmentation.unet import UNET
from nets.segmentation.utime import UTime

segmentation_dict = {
    "dlv3": get_model,
    "unet": UNET,
    "utime": UTime
}