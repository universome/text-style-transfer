import torch.nn as nn

from src.models.layers import Dropword


class RNNDecoder(nn.Module):
    def __init__(self, emb_size, hid_size, vocab, dropword_p=0):
        super(RNNDecoder, self).__init__()

        self.hid_size = hid_size
        self.embed = nn.Embedding(len(vocab), emb_size, padding_idx=vocab.stoi['<pad>'])
        self.dropword = Dropword(dropword_p)
        self.gru = nn.GRU(emb_size, hid_size, batch_first=True)
        self.z_to_logits = nn.Linear(hid_size, len(vocab))
        # self.z_to_logits.weight = self.embed.weight # Sharing weights

    def forward(self, z, sentences):
        embs = self.embed(sentences)
        embs = self.dropword(embs)
        self.gru.flatten_parameters()
        hid_states, _ = self.gru(embs, z.unsqueeze(0))
        logits = self.z_to_logits(hid_states)

        return logits
