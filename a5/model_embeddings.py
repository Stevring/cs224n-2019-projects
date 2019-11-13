#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
from vocab import VocabEntry

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab:VocabEntry):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab['<pad>']
        char_embed_size = 50
        self.embeddings = nn.Embedding(len(vocab.char2id), char_embed_size, padding_idx=pad_token_idx)
        self.cnn = CNN(21, char_embed_size, embed_size, 5)
        self.highway = Highway(embed_size, embed_size)
        self.embed_size = embed_size
        ### END YOUR CODE

    def forward(self, input:torch.Tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        word_embeddings = []
        for word_batch in torch.unbind(input, 0):
            word_batch_embedding = torch.transpose(self.embeddings(word_batch), 1, 2) # (batch_size, char_embed_size, max_word_length)
            word_batch_conved = self.cnn(word_batch_embedding) # (batch_size, embed_size)
            word_batch_highway = self.highway(word_batch_conved) # (batch_size, embed_size)
            word_embeddings.append(word_batch_highway)
        word_embeddings_tensor = torch.stack(word_embeddings) # (sequence_length, batch_size, embed_size)
        return word_embeddings_tensor
        ### END YOUR CODE

