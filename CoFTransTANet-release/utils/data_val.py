"""Dataset loaders for binary camouflaged-person segmentation."""

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch.utils.data as data
import torchvision.transforms as transforms


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
MASK_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


def _paired_files(image_root, mask_root):
    images = {p.stem: p for p in Path(image_root).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    masks = {p.stem: p for p in Path(mask_root).iterdir() if p.suffix.lower() in MASK_SUFFIXES}
    common = sorted(images.keys() & masks.keys())
    if not common:
        raise RuntimeError(f'No matching image/mask pairs found in {image_root} and {mask_root}.')
    missing_masks = sorted(images.keys() - masks.keys())
    if missing_masks:
        raise RuntimeError(f'Missing masks for {len(missing_masks)} images; first item: {missing_masks[0]}')
    return [(str(images[name]), str(masks[name])) for name in common]


def _random_flip(image, mask):
    if random.randint(0, 1):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask


def _random_crop(image, mask, border=30):
    width, height = image.size
    crop_width = np.random.randint(max(1, width - border), width + 1)
    crop_height = np.random.randint(max(1, height - border), height + 1)
    box = ((width - crop_width) // 2, (height - crop_height) // 2,
           (width + crop_width) // 2, (height + crop_height) // 2)
    return image.crop(box), mask.crop(box)


def _random_rotation(image, mask):
    if random.random() > 0.8:
        angle = np.random.randint(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        mask = mask.rotate(angle, Image.NEAREST)
    return image, mask


def _enhance_colour(image):
    image = ImageEnhance.Brightness(image).enhance(random.randint(5, 15) / 10.0)
    image = ImageEnhance.Contrast(image).enhance(random.randint(5, 15) / 10.0)
    image = ImageEnhance.Color(image).enhance(random.randint(0, 20) / 10.0)
    return ImageEnhance.Sharpness(image).enhance(random.randint(0, 30) / 10.0)


class SegmentationDataset(data.Dataset):
    def __init__(self, image_root, mask_root, image_size):
        self.pairs = _paired_files(image_root, mask_root)
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if image.size != mask.size:
            raise ValueError(f'Image and mask sizes differ: {image_path}, {mask_path}')
        image, mask = _random_flip(image, mask)
        image, mask = _random_crop(image, mask)
        image, mask = _random_rotation(image, mask)
        image = _enhance_colour(image)
        return self.image_transform(image), self.mask_transform(mask)

    def __len__(self):
        return len(self.pairs)


def get_loader(image_root, gt_root, batchsize, trainsize,
               shuffle=True, num_workers=8, pin_memory=True):
    dataset = SegmentationDataset(image_root, gt_root, trainsize)
    return data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle,
                           num_workers=num_workers, pin_memory=pin_memory)


class test_dataset:
    """Compatibility loader used by the original training and testing scripts."""

    def __init__(self, image_root, gt_root, testsize):
        self.pairs = _paired_files(image_root, gt_root)
        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.size = len(self.pairs)
        self.index = 0

    def load_data(self):
        image_path, mask_path = self.pairs[self.index]
        original = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image = self.transform(original).unsqueeze(0)
        name = Path(image_path).stem + '.png'
        original = original.resize(mask.size)
        self.index = (self.index + 1) % self.size
        return image, mask, name, '', np.array(original)

    def __len__(self):
        return self.size
