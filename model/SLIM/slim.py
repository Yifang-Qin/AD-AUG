import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from sklearn.linear_model import ElasticNet

class SLIM(nn.Module):
    def __init__(self, arg, num_users, num_items, device, aug=False):
        super(SLIM, self).__init__()
        self.n_users = num_users
        self.n_items = num_items
        self.arg = arg
        self.device = device
        self.aug = aug
        self.l1_reg = arg.l1_reg
        self.l2_reg = arg.l2_reg
        self.topk = arg.topk
        alpha = self.l1_reg + self.l2_reg
        l1_ratio = self.l1_reg / alpha
        self.slim = ElasticNet(alpha=alpha,
                               l1_ratio=l1_ratio,
                               positive=True,
                               fit_intercept=False,
                               copy_X=False,
                               precompute=True,
                               selection='random',
                               max_iter=300,
                               tol=1e-3)

    def fit_slim(self, train_matrix, num_blocks=10000000):
        num_items = train_matrix.shape[1]

        # Use array as it reduces memory requirements compared to lists
        rows = np.zeros(num_blocks, dtype=np.int32)
        cols = np.zeros(num_blocks, dtype=np.int32)
        values = np.zeros(num_blocks, dtype=np.float32)

        numCells = 0
        tqdm_iterator = tqdm(range(num_items), desc='# items covered', total=num_items)
        for item in tqdm_iterator:
            y = train_matrix[:, item].toarray()

            # set the j-th column of X to zero
            start_pos = train_matrix.indptr[item]
            end_pos = train_matrix.indptr[item + 1]

            current_item_data_backup = train_matrix.data[start_pos: end_pos].copy()
            train_matrix.data[start_pos: end_pos] = 0.0

            self.slim.fit(train_matrix, y)

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            nonzero_model_coef_index = self.slim.sparse_coef_.indices
            nonzero_model_coef_value = self.slim.sparse_coef_.data

            local_topK = min(len(nonzero_model_coef_value) - 1, self.topk)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(num_blocks, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(num_blocks, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(num_blocks, dtype=np.float32)))

                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = item
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            train_matrix.data[start_pos:end_pos] = current_item_data_backup

        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                      shape=(num_items, num_items), dtype=np.float32)

    def forward(self, train_matrix):
        train_matrix = train_matrix.tocsc()
        self.fit_slim(train_matrix)
        output = train_matrix.tocsr() @ self.W_sparse
        loss = F.binary_cross_entropy(torch.tensor(train_matrix.toarray()), torch.tensor(output.toarray()),
                                      reduction='none').sum(dim=1).mean()
        return loss

    def predict(self, eval_pos, test_batch_size):
        input_matrix = eval_pos.toarray()
        preds = np.zeros_like(input_matrix)

        num_data = input_matrix.shape[0]
        num_batches = int(np.ceil(num_data / test_batch_size))
        perm = list(range(num_data))
        for b in range(num_batches):
            if (b + 1) * test_batch_size >= num_data:
                batch_idx = perm[b * test_batch_size:]
            else:
                batch_idx = perm[b * test_batch_size: (b + 1) * test_batch_size]
            test_batch_matrix = input_matrix[batch_idx]
            batch_pred_matrix = (test_batch_matrix @ self.W_sparse)
            preds[batch_idx] = batch_pred_matrix

        preds[eval_pos.nonzero()] = float('-inf')

        return preds




        # self.init_weight(arg.dfac)

    # def init_weight(self, dfac):
    #     self.user_embedding = nn.Embedding(self.n_users, dfac)
    #     self.encoder = nn.Linear(self.n_items, dfac)
    #     self.decoder = nn.Linear(dfac, self.n_items)
    #
    #     nn.init.xavier_uniform_(self.encoder.weight)
    #     nn.init.xavier_uniform_(self.decoder.weight)
    #     nn.init.trunc_normal_(self.encoder.bias, std=0.001)
    #     nn.init.trunc_normal_(self.decoder.bias, std=0.001)

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

    # def forward(self, user_id, input, is_train=False):
    #     # h = F.normalize(input, dim=1)
    #     h = F.dropout(input, p=1 - self.arg.keep, training=is_train)
    #     h = self.encoder(h) + self.user_embedding(user_id)
    #     probs = self.decoder(h)
    #
    #     # logits = F.log_softmax(probs, dim=1)
    #     # recon_loss = torch.sum(-logits * input, dim=-1).mean()
    #     logits = torch.sigmoid(probs)
    #     recon_loss = F.binary_cross_entropy(logits, input, reduction='none').sum(dim=1).mean()
    #
    #     # return probs, recon_loss
    #     return logits, recon_loss