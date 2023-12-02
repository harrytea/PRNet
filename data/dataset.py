import os.path
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image,ImageChops
import torch
import random
import numpy as np


class ShadowDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.root = opt['dataroot']
        self.dir_A = os.path.join(opt['dataroot'], 'train_A')
        self.dir_B = os.path.join(opt['dataroot'], 'train_B')
        self.dir_C = os.path.join(opt['dataroot'], opt['sf_fixed'])

        self.imname  = sorted(os.listdir(self.dir_A))
        self.A_paths = [self.dir_A+'/'+img for img in sorted(os.listdir(self.dir_A))]
        self.B_paths = [self.dir_B+'/'+img for img in sorted(os.listdir(self.dir_B))]
        self.C_paths = [self.dir_C+'/'+img for img in sorted(os.listdir(self.dir_C))]

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.A_size = len(self.A_paths)

    def __len__(self):
        return self.A_size

    def __getitem__(self, index):
        birdy = {}
        imname = self.imname[index % self.A_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        C_path = self.C_paths[index % self.A_size]

        birdy['A'] = Image.open(A_path).convert('RGB')
        birdy['B'] = Image.open(B_path)
        birdy['C'] = Image.open(C_path).convert('RGB')

        ow = birdy['A'].size[0]
        oh = birdy['A'].size[1]
        w, h = birdy['A'].size

        loadSize = self.opt['loadSize']  # load_img_size: 256
        loadSize = np.random.randint(loadSize + 1, loadSize * 1.3, 1)[0]
        if w>h: # keep ratio
            ratio = np.float(loadSize)/np.float(h)
            neww = np.int(w*ratio)
            newh = loadSize
        else:
            ratio = np.float(loadSize)/np.float(w)
            neww = loadSize
            newh = np.int(h*ratio)


        # 1. flip and rotate
        t = [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90,]
        for _ in range(0, 4):
            c = np.random.randint(0, 2, 1, dtype=np.int)[0]
            if c==2: continue
            for i in ['A','B','C']:
                birdy[i]=birdy[i].transpose(t[c])

        # 2. rotate and resize
        degree = np.random.randint(-20, 20, 1)[0]
        for i in ['A','B','C']:
            birdy[i]=birdy[i].rotate(degree)
        for k, im in birdy.items():
            birdy[k] = im.resize((neww, newh), Image.NEAREST)


        w, h = birdy['A'].size
        for k, im in birdy.items():
            birdy[k] = self.transform(im)
        for i in ['A','B','C']:
            birdy[i] = (birdy[i] - 0.5)*2

        if not self.opt['no_crop']:
            w_offset = random.randint(0, max(0, w-self.opt['fineSize']-1))
            h_offset = random.randint(0, max(0, h-self.opt['fineSize']-1))
            for k,im in birdy.items():
                birdy[k] = im[:, h_offset:h_offset+self.opt['fineSize'], w_offset:w_offset+self.opt['fineSize']]

        if (not self.opt['no_flip']) and random.random() < 0.5:
            idx = [i for i in range(birdy['A'].size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            for k, im in birdy.items():
                birdy[k] = im.index_select(2, idx)

        for k,im in birdy.items():
            birdy[k] = im.type(torch.FloatTensor)

        birdy['w'] = ow
        birdy['h'] = oh
        birdy['imname'] = imname
        birdy['A_paths'] = A_path
        birdy['B_baths'] = B_path
        birdy['C_baths'] = C_path
        return birdy


    def name(self):
        return 'ShadowDataset'