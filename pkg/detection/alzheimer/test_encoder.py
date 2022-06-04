# Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import logging

import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torchvision

from autoencoder import AutoEncoder 
from AD_3DRandomPatch import AD_3DRandomPatch

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="Starter code for AutoEncoder")

parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--batch_size", default=1, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")
parser.add_argument("--num_classes", default=2, type=int,
                    help="Number of classes.")
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")  

def main(options):

    if options.num_classes == 2:
        TRAINING_PATH = 'train_2classes.txt'
    else:
        TRAINING_PATH = 'train.txt'
    IMG_PATH = './Image'

    dset_train = AD_3DRandomPatch(IMG_PATH, TRAINING_PATH)

    train_loader = DataLoader(dset_train,
                              batch_size = options.batch_size,
                              shuffle = True,
                              num_workers = 4,
                              drop_last = True
                              )

    sparsity = 0.05
    beta = 0.5

    mean_square_loss = nn.MSELoss()
    kl_div_loss = nn.KLDivLoss(reduce=False)

    use_gpu = len(options.gpuid)>=1
    autoencoder = AutoEncoder()

    if(use_gpu):
        autoencoder = autoencoder.cuda()
    else:
        autoencoder = autoencoder.cpu()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)

    train_loss = 0.
    for epoch in range(options.epochs):
        print("At {0}-th epoch.".format(epoch))
        for i, patches in enumerate(train_loader):
            print i
            print len(patches)
            # for batch in patches:
            #     batch = Variable(batch).cuda()
            #     output, mean_activitaion = autoencoder(batch)
            #     loss = mean_square_loss(batch, output) + kl_div_loss(mean_activitaion, sparsity)
            #     train_loss += loss
            #     logging.info("batch {0} training loss is : {1:.5f}".format(i, loss.data[0]))
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
    #     train_avg_loss = train_loss/len(train_loader*1000)
    #     print("Average training loss is {0:.5f} at the end of epoch {1}".format(train_avg_loss.data[0], epoch))
    # torch.save(model.state_dict(), open("autoencoder_model", 'wb'))

if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
  main(options)
