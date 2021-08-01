#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (t-sigai at microsoft dot com)                 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, densenet121
#from torchsummary import summary

class model_rnn(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(model_rnn, self).__init__()

        self.n_neurons = n_neurons
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)

        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self,) :
        r = torch.zeros(1, self.batch_size, self.n_neurons)
        return r.cuda()

    def forward(self, X) :
        print("X shape : {}".format(X.shape))
        X_ = []
        for x in X : 
            #print(x.shape)
            x = transforms.functional.to_grayscale(transforms.ToPILImage()(x), num_output_channels=1)
            #print("2 {}".format((transforms.ToTensor()(x)).shape))
            x = transforms.ToTensor()(x)
            x = x.tolist()
            X_.append(x)
        X_ = torch.tensor(X_)
        #print("X_ shape : {}".format(X_.shape))
        X_ = F.interpolate(X_, self.n_inputs)
        #print("X_ shape : {}".format(X_.shape))
        X__ = X_.view(-1, 192, 192)
        X__ = X__.permute(1, 0, 2)
        #print(X__.shape)
        self.batch_size = X__.size(1)
        #print("Input batch size : {}".format(self.batch_size))
        self.hidden = self.init_hidden()

        X__ = X__.cuda()
        lstm_out, self.hidden = self.basic_rnn(X__, self.hidden)
        out = self.FC(self.hidden)
        #print("rnn out : {}".format((out.view(-1, self.n_outputs)).shape))

        return out.view(-1, self.n_outputs)
        #return out

