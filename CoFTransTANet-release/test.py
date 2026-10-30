import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from lib.TANet_FCT import TANet as Network
from utils.data_val import test_dataset
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--checkpoint', '--pth_path', dest='checkpoint', type=str, required=True,
                    help='path to a trained CoFTransTANet checkpoint')
parser.add_argument('--data_root', type=str, required=True,
                    help='test-set directory containing imgs/ and masks/')
parser.add_argument('--save_root', type=str, default='./results', help='prediction output directory')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU index; ignored when CUDA is unavailable')
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_name = os.path.basename(os.path.normpath(opt.data_root))
save_path = os.path.join(opt.save_root, data_name)
model = Network(imagenet_pretrained=False)
model.load_state_dict(torch.load(opt.checkpoint, map_location=device))
model.to(device)
model.eval()

os.makedirs(save_path, exist_ok=True)
image_root = os.path.join(opt.data_root, 'imgs')
gt_root = os.path.join(opt.data_root, 'masks')
test_loader = test_dataset(image_root, gt_root, opt.testsize)

with torch.no_grad():
    for _ in range(test_loader.size):
        image, gt, name, _, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)

        res = model(image)
        # res = res2
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(data_name, name))
        imageio.imwrite(os.path.join(save_path, name), (res * 255).round().astype(np.uint8))
