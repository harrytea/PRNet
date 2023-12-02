import torch
import torch.nn as nn

from .update import BasicUpdateBlock
from .extractor import BasicEncoder

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *opt):
            pass


class PRNet(nn.Module):
    def __init__(self, opt):
        super(PRNet, self).__init__()
        self.opt = opt

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        if opt['dropout']==False:
            self.use_drop = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(input_dim=opt['in_channel_m'], output_dim=128, norm_fn='instance', dropout=self.use_drop)
        # self.cnet = BasicEncoder(input_dim=opt['in_channel_m'], output_dim=hdim+cdim, norm_fn='batch', dropout=self.use_drop)
        self.update_block = BasicUpdateBlock(self.opt, hidden_dim=hdim)


    def forward(self, img_m, img, m, iters=6, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # hdim = self.hidden_dim
        # cdim = self.context_dim

        # run the feature network
        proc_img = img
        feat = self.fnet(img_m)

        feat = torch.tanh(feat) # add

        sf_pred = []
        for itr in range(iters):
            feat, refine_img = self.update_block(feat, m, torch.cat([proc_img, m], dim=1))
            proc_img = refine_img
            sf_pred.append(proc_img)

        if test_mode:
            return proc_img

        return sf_pred
