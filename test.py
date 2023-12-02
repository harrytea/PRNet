import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import torch
import logging
import argparse
from tqdm import tqdm
from PIL import Image
from model.prnet import PRNet
from utils.util import load_checkpoints, tensor2im, tensor2m
from utils.logger import get_root_logger
from data.test_data import SingleDataset
import mmcv
from utils import util

opt = util.load_config("1.yaml")
class Predict():
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger
        self.model = PRNet(opt).cuda()

    def ISTD_test(self):
        opt = self.opt
        dataset = SingleDataset(opt)
        self.logger.info("{}: size {}".format(dataset.name(), len(dataset)))

        for epoch in range(310, 411, 10):
            opt['results_img'] ='./results/' +"istd/" +str(epoch)+'img'
            self.logger.info(opt['results_img'])
            load_path = './checkpoints/save/{}_net.pth'.format(epoch)
            load_checkpoints(self.model, load_path)
            self.model.cuda()
            self.model.eval()
            self.eval_backend_output_only(dataset, opt)


    def eval_backend_output_only(self, dataset, opt):
        util.mkdir_or_exist(opt['results_img'])
        for _, data in enumerate(tqdm(dataset)):
            with torch.no_grad():
                s, m = data['A'], data['B']
                s, m = s.cuda(), m.cuda()
                input = torch.cat([s, m], dim=1)

                output = self.model(input, s, m, iters=opt['iters'])
                save_img = tensor2im(output[-1])
                im = Image.fromarray(save_img)
                im.save(os.path.join(opt['results_img'], data['imname']))



if __name__=='__main__':
    logger = get_root_logger(name='Test', log_file='./checkpoints/test.log', log_level=logging.INFO)
    predict = Predict(opt, logger)
    predict.ISTD_test()
