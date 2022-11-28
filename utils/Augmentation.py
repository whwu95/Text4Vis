from datasets.transforms import *
from RandAugment import RandAugment

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group], label

def train_augmentation(input_size, flip=True):
    if flip:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            GroupRandomHorizontalFlip(is_flow=False)])
    else:
        return torchvision.transforms.Compose([
            GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip_sth()])


def get_augmentation(training, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 256 if config.data.input_size == 224 else config.data.input_size

    normalize = GroupNormalize(input_mean, input_std)
    if 'something' in config.data.dataset:
        groupscale = GroupScale((240, 320))
    else:
        groupscale = GroupScale(int(scale_size))

    if training:
        train_aug = train_augmentation(
            config.data.input_size,
            flip=False if 'something' in config.data.dataset else True)

        unique = torchvision.transforms.Compose([
            groupscale,
            train_aug,
            GroupRandomGrayscale(p=0.2),
        ])
    else:
        unique = torchvision.transforms.Compose([
            groupscale,
            GroupCenterCrop(config.data.input_size)])

    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize])
    return torchvision.transforms.Compose([unique, common])




def randAugment(transform_train, config):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train
