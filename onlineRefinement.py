from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
from onlineTrainer import OnlineTrainer

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    onlineTrainer = OnlineTrainer(opts)
    onlineTrainer.train()
