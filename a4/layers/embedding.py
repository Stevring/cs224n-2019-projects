import torch.nn as nn
import torch


class FeatureRichEmbedding(nn.Module):
    def __init__(self, word_vocab_size, word_embed_size,
                       bio_vocab_size, bio_embed_size,
                       feat_vocab_size, feat_embed_size):
        super(FeatureRichEmbedding, self).__init__()
        self.word_embed_size = word_embed_size
        self.bio_embed_size = bio_embed_size
        self.feat_embed_size = feat_embed_size
        self.word_embeddings = nn.Embedding(word_vocab_size, word_embed_size)
        self.bio_embeddings = nn.Embedding(bio_vocab_size, bio_embed_size)
        self.feat_embeddings = nn.Embedding(feat_vocab_size, feat_embed_size)

    def forward(self, word_index, bio_index, *feat_index):
        word = self.word_embeddings(word_index)
        bio = self.bio_embeddings(bio_index)
        feat = torch.cat([self.feat_embeddings(index) for index in feat_index], dim=-1)
        return torch.cat([word, bio, feat], dim=-1)
