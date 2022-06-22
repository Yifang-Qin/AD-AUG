import time

import torch, torch.nn as nn

from vae import MultVAE
from dae import MultDAE
import torch.nn.functional as F

class ADV(nn.Module):
    def __init__(self, n_item, arg, device):
        super(ADV, self).__init__()

        self.n_item = n_item
        self.arg = arg
        self.device = device

        if arg.model == 'multvae':
            self.G = MultVAE(arg, n_item, device, aug=True).to(device)
            self.D = MultVAE(arg, n_item, device).to(device)
        elif arg.model == 'multdae':
            self.G = MultDAE(arg, n_item, device, aug=True).to(device)
            self.D = MultDAE(arg, n_item, device).to(device)

    def forward(self, x, anneal, opt, opt_aug, is_train=False):
        # Optimize augmenter
        # x_drop = F.dropout(x, p=1 - self.arg.keep, training=is_train)

        self.G.train()
        self.G.zero_grad()
        self.D.eval()
        # aug = self.G(x_drop, x_drop, is_train=is_train)
        aug = self.G(x, x, is_train=is_train)

        gumbel = self.GumbelMax(aug)
        # gumbel = aug

        # aug_graph = torch.zeros_like(x_drop).to(self.device)
        aug_graph = torch.zeros_like(x).to(self.device)

        # aug_graph[x_drop == 0] = (1 - (1 - x_drop) * gumbel)[x_drop == 0]
        # aug_graph[(x_drop == 0).logical_not()] = (x_drop * gumbel)[(x_drop == 0).logical_not()]
        aug_graph[x == 0] = (1 - (1 - x) * gumbel)[x == 0]
        aug_graph[x == 1] = (x * gumbel)[x == 1]
        # aug_graph[x == 0] = gumbel[x == 0]
        # aug_graph[x == 1] = (1 - gumbel)[x == 1]

        z_aug = self.D.encode(aug_graph, is_train=is_train)
        # z_ori = self.D.encode(x_drop, is_train=is_train)
        z_ori = self.D.encode(x, is_train=is_train)
        mi_aug = self.calc_I(z_ori, z_aug)

        # add = torch.norm(aug_graph[x_drop == 0], p=1) / (x_drop == 0).sum()
        # drop = torch.norm(1 - aug_graph[(x_drop == 0).logical_not()], p=1) / ((x_drop == 0).logical_not()).sum()
        add = torch.norm(aug_graph[x == 0], p=1) / (x == 0).sum()
        drop = torch.norm(1 - aug_graph[x == 1], p=1) / (x == 1).sum()
        reg_aug = add + drop

        # print(((x_drop == 0).logical_not()).sum())
        # time.sleep(10)
        # reg_aug = drop

        aug_loss = self.arg.alpha_aug * mi_aug + self.arg.rg_aug * reg_aug

        aug_loss.backward()
        opt_aug.step()

        # Optimize recommender
        self.D.train()
        self.D.zero_grad()
        self.G.eval()

        # aug = self.G(x_drop, x_drop, is_train=is_train)
        aug = self.G(x, x, is_train=is_train)
        gumbel = self.GumbelMax(aug)
        # gumbel = aug
        # aug_graph = torch.zeros_like(x_drop).to(self.device)
        aug_graph = torch.zeros_like(x).to(self.device)

        # aug_graph[x_drop == 0] = (1 - (1 - x_drop) * gumbel)[x_drop == 0]
        # aug_graph[(x_drop == 0).logical_not()] = (x_drop * gumbel)[(x_drop == 0).logical_not()]
        aug_graph[x == 0] = (1 - (1 - x) * gumbel)[x == 0]
        aug_graph[x == 1] = (x * gumbel)[x == 1]
        # aug_graph[x == 0] = gumbel[x == 0]
        # aug_graph[x == 1] = (1 - gumbel)[x == 1]

        z_aug = self.D.encode(aug_graph, is_train=is_train)

        # z_ori = self.D.encode(x, is_train=is_train)

        if self.arg.model == 'multvae':
            # _, recon, kl, z_ori = self.D(x_drop, x, is_train=is_train)
            _, recon, kl, z_ori = self.D(x, x, is_train=is_train)
            mi_rec = self.calc_I(z_ori, z_aug)
            rec_loss = recon + anneal * kl - self.arg.alpha * mi_rec
            # rec_loss = recon + anneal * kl - (1-anneal) * mi_rec
        elif self.arg.model == 'multdae':
            # _, recon, z_ori = self.D(x_drop, x, is_train=is_train)
            # z_ori = self.D.encode(x, is_train=is_train)
            _, recon, z_ori = self.D(x, x, is_train=is_train)
            mi_rec = self.calc_I(z_ori, z_aug)
            rec_loss = recon - self.arg.alpha * mi_rec
            kl = torch.zeros(1).to(self.device)


        # rec_loss = recon + recon_aug + anneal * (kl+kl_aug) - self.arg.alpha * mi_rec
        # rec_loss = recon - self.arg.alpha * mi_rec

        rec_loss.backward()
        opt.step()

        return rec_loss, aug_loss, mi_rec, mi_aug, recon, kl, reg_aug, drop, add  # torch.Tensor([0.]), reg_aug

    def GumbelMax(self, prob):
        bias = 0.0001
        delta = ((bias - (1 - bias)) * torch.rand(prob.size()) + (1 - bias)).to(self.device)
        p_k = torch.log(delta) - torch.log(1 - delta) + prob
        p_k = torch.sigmoid(p_k / self.arg.tau_aug)
        return p_k

    # def calc_I(self, x1, x2, temperature=0.2):
    #     # Shape of x1 & x2: K * batch_size * proj_hid
    #     K, batch_size, _ = x1.size()
    #     x1_abs = x1.norm(dim=2, p=1)
    #     x2_abs = x2.norm(dim=2, p=1)
    #
    #     sim_matrix = torch.einsum('mil,njl->mnij', x1, x2) / torch.einsum('mi,nj->mnij', x1_abs, x2_abs)
    #     sim_matrix = torch.exp(sim_matrix / temperature)
    #
    #     # Shape: K * K * batch_size
    #     pos_sim = sim_matrix[:, :, range(batch_size), range(batch_size)]
    #     pos_sim = pos_sim[range(K), range(K), :]
    #
    #     # Shape: K * batch_size
    #     loss_0 = pos_sim / (sim_matrix.sum(dim=0).sum(dim=-2) - pos_sim)
    #     loss_1 = pos_sim / (sim_matrix.sum(dim=1).sum(dim=-1) - pos_sim)
    #
    #     loss_0 = torch.log(loss_0).sum(dim=0).mean()
    #     loss_1 = torch.log(loss_1).sum(dim=0).mean()
    #     loss = (loss_0 + loss_1) / 2.0
    #     return loss

    def calc_I(self, x1, x2, temperature=0.2):
        # Shape of x1 & x2: K * batch_size * proj_hid
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('il,jl->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)

        # sim_matrix1 = torch.einsum('il,jl->ij', x1, x1) / torch.einsum('i,j->ij', x1_abs, x1_abs)
        # sim_matrix1 = torch.exp(sim_matrix1 / temperature)
        # #
        # sim_matrix2 = torch.einsum('il,jl->ij', x2, x2) / torch.einsum('i,j->ij', x2_abs, x2_abs)
        # sim_matrix2 = torch.exp(sim_matrix2 / temperature)
        #
        # # Shape: K * K * batch_size
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        #
        # pos_sim1 = sim_matrix1[range(batch_size), range(batch_size)]
        # #
        # pos_sim2 = sim_matrix2[range(batch_size), range(batch_size)]

        # Shape: K * batch_size
        # loss_0 = pos_sim / (sim_matrix.sum(dim=0).sum(dim=-1) + sim_matrix1.sum(dim=0).sum(dim=-1) - pos_sim - pos_sim1)
        # loss_1 = pos_sim / (sim_matrix.sum(dim=1).sum(dim=-2) + sim_matrix2.sum(dim=0).sum(dim=-1) - pos_sim - pos_sim2)

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # loss_0 = pos_sim / (sim_matrix1.sum(dim=1) - pos_sim1)
        # loss_1 = pos_sim / (sim_matrix2.sum(dim=1) - pos_sim2)
        # loss_0 = pos_sim / sim_matrix1.sum(dim=0).sum(dim=-1)
        # loss_1 = pos_sim / sim_matrix2.sum(dim=0).sum(dim=-1)


        loss_0 = torch.log(loss_0).mean()
        loss_1 = torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
        return loss

    # def calc_I(self, x1, x2, temperature=0.2):
    #     # Shape of x1 & x2: K * batch_size * proj_hid
    #     # print(x1.shape)
    #     # print(x2.shape)
    #     # time.sleep(100)
    #     K, batch_size, _ = x1.size()
    #     x1_abs = x1.norm(dim=2, p=1)
    #     x2_abs = x2.norm(dim=2, p=1)
    #
    #     sim_matrix = torch.einsum('mil,mjl->mij', x1, x2) / torch.einsum('mi,mj->mij', x1_abs, x2_abs)
    #     sim_matrix = torch.exp(sim_matrix / temperature)
    #
    #     sim_matrix1 = torch.einsum('mil,mjl->mij', x1, x1) / torch.einsum('mi,mj->mij', x1_abs, x1_abs)
    #     sim_matrix1 = torch.exp(sim_matrix1 / temperature)
    #
    #     sim_matrix2 = torch.einsum('mil,mjl->mij', x2, x2) / torch.einsum('mi,mj->mij', x2_abs, x2_abs)
    #     sim_matrix2 = torch.exp(sim_matrix2 / temperature)
    #
    #     # Shape: K * K * batch_size
    #     pos_sim = sim_matrix[:, range(batch_size), range(batch_size)]
    #     pos_sim = pos_sim[range(K), :]
    #
    #     pos_sim1 = sim_matrix1[:, range(batch_size), range(batch_size)]
    #     pos_sim1 = pos_sim1[range(K), :]
    #
    #     pos_sim2 = sim_matrix2[:, range(batch_size), range(batch_size)]
    #     pos_sim2 = pos_sim2[range(K), :]
    #
    #     # Shape: K * batch_size
    #     # loss_0 = pos_sim / (sim_matrix.sum(dim=-1) + sim_matrix1.sum(dim=-1) - pos_sim - pos_sim1)
    #     # loss_1 = pos_sim / (sim_matrix.sum(dim=-2) + sim_matrix2.sum(dim=-2) - pos_sim - pos_sim2)
    #     loss_0 = pos_sim / (sim_matrix1.sum(dim=-1) - pos_sim1)
    #     loss_1 = pos_sim / (sim_matrix2.sum(dim=-2) - pos_sim2)
    #
    #     loss_0 = torch.log(loss_0).sum(dim=0).mean()
    #     loss_1 = torch.log(loss_1).sum(dim=0).mean()
    #     loss = (loss_0 + loss_1) / 2.0
    #     return loss