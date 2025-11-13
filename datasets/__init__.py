import torch.utils.data
import torchvision

from .SHA import build as build_sha

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
}

def build_dataset(image_set, args):
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
