import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, attn_size, vocab_size, dropout):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.vocab_size = vocab_size
        self.memory_attn_proj = nn.Linear(hidden_size, attn_size)
        self.gru_hidden_attn_proj = nn.Linear(hidden_size, attn_size, bias=False)
        self.engy_attn_proj = nn.Linear(attn_size, 1, bias=False)
        self.gru_input_proj = nn.Linear(embed_size + hidden_size, hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.readout = nn.Linear(embed_size + hidden_size + hidden_size, hidden_size)
        self.readout_pooling = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.gen_proj = nn.Linear(hidden_size // 2, vocab_size)

    def concat_attn(self, query, key, value, mask):
        """
        :param query: (B, query_size)
        :param key: (B, src_len, attn_size)
        :param value: (B, src_len, value_size)
        :param mask: (B, src_len)
        :return:
        """
        query = self.gru_hidden_attn_proj(query).unsqueeze(1)  # (B, 1, attn_size)
        tmp = torch.add(key, query)  # (B, src_len, attn_size)
        e = self.engy_attn_proj(torch.tanh(tmp)).squeeze(2)  # (B, src_len)
        e = e * (1 - mask) + mask * (-1000000)
        score = torch.softmax(e, dim=1)  # (B, src_len)
        ctxt = torch.bmm(score.unsqueeze(1), value).squeeze(1)  # (B, 1, value_size)
        return ctxt, score

    def forward(self, memory, memory_mask, tgt, dec_init_hidden):
        """
        :param memory: (B, seq_len, hidden)
        :param memory_mask: (B, seq_len)
        :param tgt: (tgt_len, B, embed)
        :param dec_init_hidden: (B, hidden)
        :return:
        """
        B, src_len, _ = memory.size()
        tgt = tgt[:-1]
        memory_for_attn = self.memory_attn_proj(memory)
        dec_hidden_tm1 = dec_init_hidden
        ctxt_tm1 = torch.zeros(B, self.hidden_size, device=memory.device)
        gen_probs = []
        for tgt_tm1 in torch.split(tgt, 1, dim=0):
            tgt_tm1 = tgt_tm1.squeeze(0)
            gen_prob_t, dec_hidden_tm1, ctxt_tm1 = self.decode_step(tgt_tm1,
                                                                    ctxt_tm1,
                                                                    dec_hidden_tm1,
                                                                    memory_for_attn,
                                                                    memory,
                                                                    memory_mask)
            gen_probs.append(gen_prob_t)
        return torch.stack(gen_probs)  # (tgt_len, B, vocab_size)

    def decode_step(self, tgt_tm1, ctxt_tm1, dec_hidden_tm1, memory_for_attn, memory, memory_mask):
        """
        :param tgt_tm1:
        :param ctxt_tm1:
        :param dec_hidden_tm1:
        :param memory_for_attn:
        :param memory:
        :param memory_mask:
        :return:
        """
        gru_input_tm1 = self.gru_input_proj(torch.cat([tgt_tm1, ctxt_tm1], dim=1))
        dec_hidden_t = self.gru_cell(gru_input_tm1, dec_hidden_tm1)
        ctxt_t, attn_score_t = self.concat_attn(dec_hidden_t, memory_for_attn, memory, memory_mask)
        r_t = self.readout(torch.cat([tgt_tm1, ctxt_t, dec_hidden_t], dim=-1))
        m_t = self.readout_pooling(r_t.unsqueeze(1)).squeeze(1)  # (B, hidden // 2)
        m_t = self.dropout(m_t)
        gen_t = self.gen_proj(m_t) # (B, vocab_size)
        return gen_t, dec_hidden_t, ctxt_t
