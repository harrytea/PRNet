import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'


import time
import logging
from utils.logger import get_root_logger
from data.dataset import ShadowDataset
from torch.utils.data import DataLoader
from model.networks import *
from model.prnet import PRNet
from torch.optim import lr_scheduler
from utils import util
import numpy as np


def sequence_loss(flow_preds, flow_gt, loss_rec, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = loss_rec(flow_preds[i], flow_gt)
        flow_loss = flow_loss + i_weight * i_loss

    return flow_loss

opt = util.load_config("1.yaml")
def main():
    # 1. logger
    util.mkdir_or_exist(opt['check_dir'])
    util.mkdir_or_exist(opt['image_dir'])
    util.mkdir_or_exist(opt['save_dir'])

    logger = get_root_logger(name=opt['exp_name'], log_file=os.path.join(opt['check_dir'], 'train.log'), log_level=logging.INFO)
    logger.info(opt)


    # 2. data
    train_dataset = ShadowDataset(opt)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=opt['batch_size'], shuffle=True, drop_last=True, num_workers=16, pin_memory=True)


    # 3. model
    model = PRNet(opt)
    model = nn.DataParallel(model)
    model.cuda()
    logger.info(model)
    logger.info("model parameters: {}".format(sum(param.numel() for param in model.parameters())/1e6))


    # 4. loss optimizer
    L1Loss = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], betas=(opt['beta1'], 0.999), weight_decay=1e-5)
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt['epoch_count'] - opt['niter']) / float(opt['niter_decay'] + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)



    for epoch in range(opt['epoch_count'], opt['niter']+opt['niter_decay']+1):
        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            s, m, sf = data['A'], data['B'], data['C']
            s, m, sf = s.cuda(), m.cuda(), sf.cuda()
            m = (m>0.9).type(torch.float)*2-1
            s_m = torch.cat([s, m], dim=1)

            output = model(s_m, s, m, iters=opt['iters'])
            optimizer.zero_grad()
            loss = sequence_loss(output, sf, L1Loss, opt['gamma'])
            loss.backward()
            optimizer.step()

            logger.info("epoch: %d step: %d loss: %.2f" % (epoch, i, loss.item()))
            if i%40 == 0:
                util.save_image(opt, epoch, i, s, m, sf, output[0], output[-2], output[-1], data['A_paths'])

        if epoch%10 == 0:
            logger.info('saving the model at the end of epoch {}, iters {}'.format(epoch, i))
            util.save_networks(epoch, model, opt['save_dir'])
        logger.info('End of epoch {} / {} \t Time Taken: {} sec'.format(epoch, opt['niter']+opt['niter_decay'], time.time()-epoch_start_time))
        scheduler.step()


if __name__ == '__main__':
    seed = np.random.randint(10000)
    util.init_seeds(seed)
    main()
