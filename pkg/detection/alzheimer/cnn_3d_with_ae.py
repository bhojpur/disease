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

import torch
from autoencoder import AutoEncoder 
import torch.nn as nn
import math

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 410, kernel_size=7, stride=7, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=7,stride=7)
        # self.conv2 = nn.Conv3d(410, 200, kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool3d(kernel_size=3, stride=3)
        # self.fc1 = nn.Linear(5*5*5*200, 800)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2*3*2*410, 80)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(80, num_classes)
        self.softmax = nn.LogSoftmax()
        self.parameter_initialization()

    def forward(self, out):
        out = self.pool1(self.relu1(self.conv1(out)))
        out = self.dropout1(out)
        # out = self.pool2(self.relu2(self.conv2(out)))
        # out = out.view(-1,5*5*5*200)
        out = out.view(-1, 2*3*2*410)
        out = self.fc1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

    def parameter_initialization(self):
        stdv = 1.0 / math.sqrt(410)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
