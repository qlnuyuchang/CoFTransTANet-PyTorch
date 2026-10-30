# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
# from lib.tanet_dense import TANet as Network
from lib.TANet_FCT import TANet as Network
# from lib.TANet import TANet as Network
from lib.losses import affinity

# from lib.FCT import FCT as Network
from utils.data_val import get_loader, test_dataset
from utils.utils import adjust_lr

import logging
import torch.backends.cudnn as cudnn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train(train_loader, model, optimizer, epoch, save_path):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            pre40, pre30, pre20, pre10, mean1, mean2, mean3, std1, std2, std3 = model(images)

            loss_seg = structure_loss(pre40, gts) + structure_loss(pre30, gts) + structure_loss(pre20, gts) + structure_loss(pre10, gts)
            cb_texloss = affinity(mean1, gts, [12]) + affinity(mean2, gts, [6]) + affinity(mean3, gts, [6])
            std_texloss = affinity(std1, gts, [12]) + affinity(std2, gts, [6]) + affinity(std3, gts, [6])

            # pre = model(images)
            # loss_seg = structure_loss(pre, gts)

            (loss_seg + 0.1*cb_texloss + 0.1*std_texloss).backward()

            optimizer.step()

            step += 1
            epoch_step += 1

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], seg_loss: {:.4f} macro_loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss_seg.data, (cb_texloss+std_texloss).data))

        loss_all /= epoch_step
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch + 1)))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, video_name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device, non_blocking=True)

            res = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_best.pth'))
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=300, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU index; ignored when CUDA is unavailable')
    parser.add_argument('--train_root', type=str, default='./Dataset/train/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./Dataset/test/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/tanet_fct_1e-5/',
                        help='the path to save model and log')
    opt = parser.parse_args()

    # set the device for training
    # if opt.gpu_id == '0':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #     print('USE GPU 0')
    # elif opt.gpu_id == '1':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #     print('USE GPU 1')
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = torch.cuda.is_available()

    # build the model
    model = Network().to(device)

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load, map_location=device))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=os.path.join(opt.train_root, 'imgs'),
                              gt_root=os.path.join(opt.train_root, 'masks'),
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    val_loader = test_dataset(image_root=os.path.join(opt.val_root, 'imgs'),
                              gt_root=os.path.join(opt.val_root, 'masks'),
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=os.path.join(save_path, 'log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0

    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train(train_loader, model, optimizer, epoch, save_path,)
        val(val_loader, model, epoch, save_path)
