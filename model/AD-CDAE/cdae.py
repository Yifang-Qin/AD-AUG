import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class CDAE(nn.Module):
    def __init__(self, arg, num_users, num_items, device, aug=False):
        super(CDAE, self).__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.arg = arg
        self.device = device
        self.aug = aug

        self.init_weight(arg.dfac)

    def init_weight(self, dfac):
        self.user_embedding = nn.Embedding(self.n_users, dfac)
        self.encoder = nn.Linear(self.n_items, dfac)
        self.decoder = nn.Linear(dfac, self.n_items)

        proj_hid = self.arg.proj_hid

        if not self.aug:
            self.proj_head = nn.Sequential(
                nn.Linear(dfac, proj_hid),
                nn.ReLU(),
                # nn.Tanh(),
                nn.Linear(proj_hid, proj_hid)
                # nn.BatchNorm1d(kfac)
            )

            for m in self.proj_head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)

        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.trunc_normal_(self.encoder.bias, std=0.001)
        nn.init.trunc_normal_(self.decoder.bias, std=0.001)

        # self.enc_dims = [self.n_items, dfac, dfac]
        # self.dec_dims = [dfac, dfac, self.n_items]
        # self.encoder, self.decoder = [], []

        # for i, (enc_in, enc_out, dec_in, dec_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:], self.dec_dims[:-1], self.dec_dims[1:])):
        #     if i == len(self.enc_dims[:-1]) - 1:
        #         enc_out *= 1  # mu & var

            # self.encoder.append(nn.Linear(enc_in, enc_out))
            # self.decoder.append(nn.Linear(dec_in, dec_out))

            # nn.init.xavier_uniform_(self.encoder[-1].weight)
            # nn.init.xavier_uniform_(self.decoder[-1].weight)
            # nn.init.trunc_normal_(self.encoder[-1].bias, std=0.001)
            # nn.init.trunc_normal_(self.decoder[-1].bias, std=0.001)
    
            # self.encoder.append(nn.Tanh())
            # self.decoder.append(nn.Tanh())

        # self.encoder = nn.Sequential(*self.encoder[:-1])
        # self.decoder = nn.Sequential(*self.decoder[:-1])

    def encode(self, user_id, input, is_train):
        h = F.dropout(input, p=1 - self.arg.keep, training=is_train)
        h = self.encoder(h) + self.user_embedding(user_id)

        z_proj = self.proj_head(h)
        return z_proj

    def forward(self, user_id, input, output, is_train=False):
        # h = F.normalize(input, dim=1)
        h = F.dropout(input, p=1 - self.arg.keep, training=is_train)
        h = self.encoder(h) + self.user_embedding(user_id)
        probs = self.decoder(h)

        # logits = F.log_softmax(probs, dim=1)
        # recon_loss = torch.sum(-logits * input, dim=-1).mean()
        logits = torch.sigmoid(probs)

        if self.aug:
            return logits

        recon_loss = F.binary_cross_entropy(logits, output, reduction='none').sum(dim=1).mean()

        z_proj = self.proj_head(h)
        # return probs, recon_loss
        return logits, recon_loss, z_proj