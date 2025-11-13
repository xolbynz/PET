import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import torchvision.transforms as standard_transforms
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

def draw_sample(img, points, path=None, title="Check target", show=False):
    """
    img: torch.Tensor (3, H, W)
    points: numpy array (N, 2) or torch.Tensor (N, 2)
    path: save path. If None and show==True, show by plt.show().
    """
    # Make sure we're working on cpu & numpy
    if hasattr(img, "cpu"):
        img = img.cpu()
    if hasattr(img, "detach"):
        img = img.detach()
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.shape[0] == 3:
        img_vis = np.transpose(img, (1,2,0))
    else:
        img_vis = img
    # Undo normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_vis = img_vis * std + mean
    img_vis = np.clip(img_vis, 0, 1)
    img_vis = (img_vis*255).astype(np.uint8)

    # Points to numpy
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    try:
        points = np.asarray(points)
        if points.shape[0] == 0:
            points = np.zeros((0,2))
    except Exception:
        points = np.zeros((0,2))
    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(img_vis)
    if points.shape[0]>0:
        ax.scatter(points[:,1], points[:,0], c='r', s=10)
    ax.set_title(title + f" (N={points.shape[0]})")
    if path is not None:
        plt.savefig(path)
        plt.close()
    elif show:
        plt.show()
        plt.close()
    else:
        plt.close()

class SHA(Dataset):
    def __init__(self, data_root, transform=None, train=False, flip=False, draw_debug_first=False):
        self.root_path = data_root
        # .list 파일을 읽어서 1번이 img_path 2번이 gt_path이 되게 수정
        list_file = "train.list" if train else "test.list"
        list_path = os.path.join(data_root, list_file)
        self.img_list = []
        self.gt_list = {}
        with open(list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items = line.split()
                img_path = os.path.join(data_root, items[0].strip())
                gt_path = os.path.join(data_root, items[1].strip())
                self.img_list.append(img_path)
                self.gt_list[img_path] = gt_path
        self.img_list = sorted(self.img_list)
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = 256

        # Draw check option
        self.draw_debug_first = draw_debug_first
        self.sampled = False

    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:,1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # load image and gt points
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        img, points = load_data((img_path, gt_path), self.train)
        points = points.astype(float)
        points = to_points_array(points)

        # image transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        # random scale
        if self.train:
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            # interpolation
            if scale * min_size > self.patch_size:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                if points.shape[0] > 0:
                    points *= scale
        else:
            # ----------------- 🔸 val 전용 리사이즈 -----------------
            max_long = 1024   # 필요시 1024/1280/1920 등으로 조정
            C, H, W = img.shape
            long_side = max(H, W)
            if long_side > max_long:
                scale = max_long / long_side
                newH, newW = int(round(H * scale)), int(round(W * scale))
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(newH, newW), mode='bilinear', align_corners=False
                ).squeeze(0)
                if points.shape[0] > 0:
                    points *= scale  # (y, x) 모두 동일 배율 적용
            # -------------------------------------------------------

        # random crop patch
        if self.train:
            img, points = random_crop(img, points, patch_size=self.patch_size)

        # random flip
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            if points.shape[0] > 0:
                points[:, 1] = self.patch_size - points[:, 1]

        # target
        target = {}
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = img_path

        # Draw debug image for sanity check (only first sample)
        if self.draw_debug_first and not self.sampled and points.shape[0] < 1500:
            try:
                debug_dir = "./debug_sha_samples"
                os.makedirs(debug_dir, exist_ok=True)
                pth = os.path.join(debug_dir, f"sha_{'train' if self.train else 'val'}_{index}_check.png")
                draw_sample(img, points, path=pth, title=f"{'Train' if self.train else 'Val'} idx={index} N={points.shape[0]}")
                print(f"[SHA] Draw sample: {pth} (img shape {img.shape}, #points {points.shape[0]})")
            except Exception as e:
                print(f"[SHA] draw_sample failed: {e}")
            self.sampled = True

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = []
    with open(gt_path) as f_label:
        for line in f_label:
            x = float(line.strip().split(' ')[1])
            y = float(line.strip().split(' ')[0])
            points.append([x, y])
    return img, np.array(points)

def to_points_array(points):
    import numpy as np
    # torch.Tensor도 처리
    if points is None:
        return np.zeros((0, 2), dtype=np.float32)

    if hasattr(points, 'detach'):  # torch.Tensor
        points = points.detach().cpu().numpy()

    points = np.asarray(points, dtype=np.float32)

    # 1D (예: [x, y]) 또는 빈 배열 (0,) 케이스 방지
    if points.ndim == 1:
        if points.size == 0:
            points = points.reshape(0, 2)
        else:
            # 길이가 2의 배수인 경우만 허용
            if points.size % 2 != 0:
                raise ValueError(f"points length {points.size} is not even")
            points = points.reshape(-1, 2)

    # 최종 차원 검증
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points shape must be (N,2), got {points.shape}")

    return points
def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size
    
    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    points = to_points_array(points)
    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx].copy()  # copy needed to avoid assignment altering original points for next try
    if result_points.shape[0] > 0:
        result_points[:, 0] -= start_h
        result_points[:, 1] -= start_w

        # resize to patchsize
        imgH, imgW = result_img.shape[-2:]
        fH, fW = patch_h / imgH, patch_w / imgW
        result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
        result_points[:, 0] *= fH
        result_points[:, 1] *= fW
    else:
        # 여전히 리사이즈는 수행 (아마 empty points가 됨)
        imgH, imgW = result_img.shape[-2:]
        result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)

    return result_img, result_points


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    
    data_root = args.data_path
    # Allow passing "draw_debug_first" as an argument attribute or set here.
    draw_debug_first = getattr(args, "draw_debug_first", False)
    if image_set == 'train':
        train_set = SHA(data_root, train=True, transform=transform, flip=True, draw_debug_first=draw_debug_first)
        return train_set
    elif image_set == 'val':
        val_set = SHA(data_root, train=False, transform=transform, draw_debug_first=draw_debug_first)
        return val_set
