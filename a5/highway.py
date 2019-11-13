#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn
import torch
import torch.functional as F

class Highway(nn.Module):
    """ Highway networks. see https://arxiv.org/abs/1505.00387
        y = h(x)t(x) + x(1-t(x))
    """
    def __init__(self, input_size, output_size):
        """ Init highway network

        :param input_size: last dimension size of input tensor
        :param output_size: last dimension size of output tensor, must be equal to input_size
        """
        assert output_size == input_size, "output size of highway network must be equal to input size"
        super(Highway, self).__init__()
        self.h = nn.Linear(input_size, output_size, bias=True)
        self.t = nn.Linear(input_size, output_size, bias=True)

    def forward(self, source:torch.Tensor) -> torch.Tensor:
        """
        
        :param x: input tensor
        :return: tensor with shape equal to x
        """
        h_x = torch.relu(self.h(source))
        t = torch.sigmoid(self.t(source))
        y = h_x * t + source * (1-t)
        return y
### END YOUR CODE

