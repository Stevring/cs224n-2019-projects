from pathlib import Path
from typing import List
import torch.nn as nn

import torch
from layers.decoder import Decoder
from layers.encoder import Encoder
from model_embeddings import ModelEmbeddings
import torch.nn.functional as F
from nmt_model import Hypothesis


class QGModel(nn.Module):
    def __init__(self, vocab, embed_size, hidden_size, enc_bidir, attn_size, dropout=0.2):
        super(QGModel, self).__init__()
        self.vocab = vocab
        self.args = {
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'dropout': dropout,
            'enc_bidir': enc_bidir,
            'attn_size': attn_size
        }
        self.embeddings = ModelEmbeddings(embed_size, vocab)
        self.encoder = Encoder(embed_size, hidden_size, dropout, enc_bidir)
        self.decoder_init_hidden_proj = nn.Linear(self.encoder.hidden_size, hidden_size)
        self.decoder = Decoder(embed_size, hidden_size, attn_size, len(vocab.tgt), dropout)

    def batch_to_tensor(self, source, target):
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)
        source_mask = self.generate_mask(source_lengths, source_padded.shape[0])
        return source_padded, target_padded, source_lengths, source_mask

    def forward(self, source: List[List[str]], target: List[List[str]]):
        source_padded, target_padded, source_lengths, source_mask = self.batch_to_tensor(source, target)

        source_embedding = self.embeddings.source(source_padded)  # (src_len, b, embed_size)
        target_embedding = self.embeddings.target(target_padded)  # (tgt_len, B, embed_size)
        memory, last_hidden = self.encoder(source_embedding, source_lengths)
        # last_hidden: (B, hidden)
        memory = memory.transpose(0, 1)  # memory: (B, src_len, hidden)
        dec_init_hidden = torch.tanh(self.decoder_init_hidden_proj(last_hidden))
        gen_output = self.decoder(memory, source_mask, target_embedding, dec_init_hidden)
        # (tgt_len - 1, B, word_vocab_size), not probability
        P = F.log_softmax(gen_output, dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def generate_mask(self, length, max_length):
        mask = torch.zeros(len(length), max_length, dtype=torch.int, device=self.device)
        for i, x in enumerate(length):
            mask[i, x:] = 1
        return mask

    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70):
        """
        :param batch: batch size is 1
        :param beam_size:
        :return:
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)
        src_len = torch.tensor([len(src_sent)], dtype=torch.int, device=self.device)
        source_embedding = self.embeddings.source(src_sents_var)  # (src_len, b, embed_size)

        memory, last_hidden = self.encoder(source_embedding, src_len)
        # last_hidden: (B, hidden)
        memory = memory.transpose(0, 1)  # memory: (B, src_len, hidden)
        dec_init_hidden = torch.tanh(self.decoder_init_hidden_proj(last_hidden))  # (B, hidden)
        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []
        t = 0
        ctxt_tm1 = torch.zeros(len(hypotheses), self.args['hidden_size'], device=self.device)
        dec_hidden_tm1 = dec_init_hidden
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)
            prev_word = torch.tensor([self.vocab.tgt[x[-1]] for x in hypotheses], dtype=torch.long, device=self.device)
            tgt_tm1 = self.embeddings.target(prev_word)  # (B, word_embed_size)

            memory_tm1 = memory.expand((hyp_num, *memory.shape[1:]))
            gen_t,  dec_hidden_t, ctxt_t = self.decoder.decode_step(tgt_tm1, ctxt_tm1, dec_hidden_tm1, memory_tm1)
            gen_t = torch.log_softmax(gen_t, dim=-1) # (B, vocab)
            live_hyp_num = beam_size - len(completed_hypotheses)
            continuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(gen_t) + gen_t).view(-1)  # (hyp_num * V)
            top_candi_scores, top_candi_position = torch.topk(continuating_hyp_scores, k=live_hyp_num)
            prev_hyp_indexes = top_candi_position / len(self.vocab.tgt)
            hyp_word_indexes = top_candi_position % len(self.vocab.tgt)

            new_hypothesis = []
            live_hyp_index = []
            new_hyp_scores = []
            num_unk = 0
            for prev_hyp_index, hyp_word_index, new_hyp_score in zip(prev_hyp_indexes, hyp_word_indexes, top_candi_scores):
                prev_hyp_index = prev_hyp_index.item()
                hyp_word_index = hyp_word_index.item()
                new_hyp_score = new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_index]
                new_hypo = hypotheses[prev_hyp_index] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hypo[1:-1],
                                                           score=new_hyp_score))
                else:
                    new_hypothesis.append(new_hypo)
                    live_hyp_index.append(prev_hyp_index)
                    new_hyp_scores.append(new_hyp_score)
            if len(completed_hypotheses) == beam_size:
                break
            live_hyp_index = torch.tensor(live_hyp_index, dtype=torch.long, device=self.device)
            dec_hidden_tm1 = dec_hidden_tm1[live_hyp_index]
            ctxt_tm1 = ctxt_t[live_hyp_index]

            hypotheses = new_hypothesis
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))
        completed_hypotheses.sort(key=lambda x: x.score, reverse=True)
        return completed_hypotheses



    @property
    def device(self):
        return self.decoder_init_hidden_proj.weight.device

    def save(self, path):
        path = path + ".qg"
        dir = Path(path).parent
        dir.mkdir(parents=True, exist_ok=True)
        state_dict = {}
        state_dict['vocab'] = self.vocab
        state_dict['args'] = self.args
        state_dict['model_state'] = self.state_dict()
        torch.save(state_dict, path)


    @staticmethod
    def load(path, device):
        params = torch.load(path, map_location=device)

        model = QGModel(vocab=params['vocab'],  **params['args'])  # type:nn.Module
        model.load_state_dict(params['model_state'])
        return model.to(device)