#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """ Character Convolution for a batch of word
    """

    def __init__(self, max_char_num_per_word, char_embed_size, word_embed_size, kernel_size):
        """

        :param max_char_num_per_word: number of characters in a padded word
        :param char_embed_size: character embedding size
        :param word_embed_size: word embedding size
        :param kernel_size: kernel size of convolution layer
        """

        super(CNN, self).__init__()
        self.conv = nn.Conv1d(char_embed_size, word_embed_size, kernel_size)
        pooling_kernel_size = max_char_num_per_word - kernel_size + 1
        self.pooling = nn.MaxPool1d(pooling_kernel_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """

        :param x: shape (batch_size, char_embed_size, char_num)
        :return: tensor of shape (word_num, word_embed_size)
        """
        x_conv = self.conv(x) # shape (batch_size, word_embed_size, L)
        x_max_pool = torch.squeeze(torch.relu(self.pooling(x_conv)), -1) # shape (batch_size, word_embed_size)
        return x_max_pool

### END YOUR CODE

