import os.path
from PIL import ImageFilter
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as data
import numpy as np
import torch

class SingleDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.dir_A = opt['test_image']
        self.dir_B = opt['test_mask']

        self.imname  = os.listdir(self.dir_A)
        self.A_paths = [self.dir_A+'/'+img for img in sorted(os.listdir(self.dir_A))]
        self.B_paths = [self.dir_B+'/'+img for img in sorted(os.listdir(self.dir_B))]

        self.transformB = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        imname = self.imname[index]
        A_path= os.path.join(self.dir_A,imname)
        B_path= os.path.join(self.dir_B,imname)
        A_img = Image.open(A_path).convert('RGB')
        if not os.path.isfile(B_path):
            B_path=B_path[:-4]+'.png'
        B_img = Image.open(B_path).convert('L')

        ow = A_img.size[0]
        oh = A_img.size[1]
        loadsize = self.opt.fineSize if hasattr(self.opt,'fineSize') else 256
        A_img_ori = A_img
        A_img = A_img.resize((loadsize,loadsize))
        B_img = B_img.resize((loadsize,loadsize))
        A_img = torch.from_numpy(np.asarray(A_img,np.float32).transpose(2,0,1)).div(255)
        A_img_ori = torch.from_numpy(np.asarray(A_img_ori,np.float32).transpose(2,0,1)).div(255)
        B_img = self.transformB(B_img)
        B_img = B_img*2-1
        A_img = A_img*2-1
        A_img_ori = A_img_ori*2-1
        A_img = A_img.unsqueeze(0)
        A_img_ori = A_img_ori.unsqueeze(0)
        B_img = B_img.unsqueeze(0)
        B_img = (B_img>0.2).type(torch.float)*2-1

        return {'A':A_img, 'B':B_img, 'A_ori':A_img_ori, 'A_paths':A_path, 'imname':imname, 'w':ow, 'h':oh}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
