from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import skimage.io as io
from pathlib import Path
import yaml
import torch.nn as nn
from collections import OrderedDict
import random


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):

    if len(input_image.shape)<3: return None
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.data[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy[image_numpy<0] = 0
    image_numpy[image_numpy>255] = 255
    return image_numpy.astype(imtype)

# Converts a Tensor into an mask array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2m(input_mask, imtype=np.uint8):
    input_mask = (input_mask+1)/2  # add
    input_mask = torch.sigmoid(input_mask)

    if len(input_mask.shape)<3: return None
    if isinstance(input_mask, torch.Tensor):
        image_tensor = input_mask.data
    else:
        return input_mask
    image_numpy = image_tensor.data[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0
    image_numpy[image_numpy<0] = 0
    image_numpy[image_numpy>255] = 255
    return image_numpy.astype(imtype)

# def save_image(opt, epoch, iter, s, m, sf, sf_pred, path):
#     s = tensor2im(s)
#     m = tensor2im(m)
#     sf = tensor2im(sf)
#     sf_pred = tensor2im(sf_pred)
#     img = np.concatenate((s,m,sf,sf_pred), axis=1)
#     img_file = os.path.join(opt['image_dir'], "epoch%d_iter%d_%s"%(epoch, iter, path[0].split('/')[-1]))
#     io.imsave(img_file, img)

def save_image(opt, epoch, iter, s, m, sf, m_pred, chro_pred, sf_pred, path):
    s = tensor2im(s)
    m = tensor2im(m)
    sf = tensor2im(sf)
    m_pred = tensor2m(m_pred)
    chro_pred = tensor2im(chro_pred)
    sf_pred = tensor2im(sf_pred)
    img = np.concatenate((s,m,sf,m_pred,chro_pred,sf_pred), axis=1)
    img_file = os.path.join(opt['image_dir'], "epoch%d_iter%d_%s"%(epoch, iter, path[0].split('/')[-1]))
    io.imsave(img_file, img)
    # self.logger.info("save: %s" % (img_file))


def save_networks(epoch, net, path):
    save_filename = '%s_net.pth' % (epoch)
    save_path = os.path.join(path, save_filename)
    torch.save(net.state_dict(), save_path)


def load_config(file_name):
    path = Path(__file__).parent.parent/"cfg"/file_name
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)

class MyWcploss(nn.Module):
    def __init__(self):
        super(MyWcploss, self).__init__()

    def forward(self, pred, gt):
        pred = (pred+1)/2
        gt = (gt+1)/2
        eposion = 1e-10
        # sigmoid_pred = torch.sigmoid(pred)
        count_pos = torch.sum(gt)*1.0+eposion
        count_neg = torch.sum(1.-gt)*1.0
        beta = count_neg/count_pos
        beta_back = count_pos / (count_pos + count_neg)
        bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
        loss = beta_back*bce1(pred, gt)
        return loss

# class MyWcploss(nn.Module):
#     def __init__(self):
#         super(MyWcploss, self).__init__()

#     def forward(self, pred, gt):
#         eposion = 1e-10
#         # sigmoid_pred = torch.sigmoid(pred)
#         count_pos = torch.sum(gt)*1.0+eposion
#         count_neg = torch.sum(1.-gt)*1.0
#         beta = count_neg/count_pos
#         beta_back = count_pos / (count_pos + count_neg)
#         bce1 = nn.BCEWithLogitsLoss(pos_weight=beta)
#         loss = beta_back*bce1(pred, gt)
#         return loss


'''  initial seed  '''
def init_seeds(seed=0):
    random.seed(seed)  # seed for module random
    np.random.seed(seed)  # seed for numpy
    torch.manual_seed(seed)  # seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # seed for current PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # seed for all PyTorch GPUs
    if seed == 0:
        # if True, causes cuDNN to only use deterministic convolution algorithms.
        torch.backends.cudnn.deterministic = True
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    import os.path as osp
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)




















def sdmkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)



def tensor2im_logc(image_tensor, imtype=np.uint8,scale=255):
    image_numpy = image_tensor.data[0].cpu().double().numpy()
    image_numpy = np.transpose(image_numpy,(1,2,0))
    image_numpy = (image_numpy+1) /2.0  
    image_numpy = image_numpy * (np.log(scale+1)) 
    image_numpy = np.exp(image_numpy) -1
    image_numpy = image_numpy.astype(np.uint8)

    return image_numpy.astype(np.uint8)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)



def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)



# def load_checkpoints(path):
#     try:
#         obj = torch.load(path, map_location='cpu')
#     except FileNotFoundError:
#         return print("File Not Found")
#     return obj


def load_checkpoints(model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint)
    except:
        state_dict = checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)