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

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv = nn.Conv3d(1, 410, kernel_size=7, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=7,stride=7)
        self.fc1 = nn.Linear(15*15*15, 800)
        self.fc2 = nn.Linear(800, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, out):
        out = self.conv(out)
        out = self.pool(out)
        out = out.view(1,15*15*15)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

    def load_ae(self, ae):
        cnn.state_dict()['conv.weight'] = ae.state_dict()['encoder.weight'].view(410,1,7,7,7)
        cnn.state_dict()['conv.bias'] = ae.state_dict()['encoder.bias']
        return cnn
