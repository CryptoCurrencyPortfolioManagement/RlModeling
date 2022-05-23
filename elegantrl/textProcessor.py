import math
import random

import torch
from torch import nn
import numpy as np

from transformers import BertTokenizer, BertModel

from gensim.models import KeyedVectors
fasttext_model = KeyedVectors.load("./other/crawl-300d-2M.model")


def _reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape)))).contiguous()

# layer which allows to randomly drop certain elements from defined indeces of the input tensor
class Dropout_partial_binary(nn.Module):
    def __init__(self, include_index: list, p: float = 0.5):
        super().__init__()
        self.include_index = include_index
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, X):
        X_to_modify = X[:, :, :, self.include_index]

        if self.training:
            X_power = torch.sum(X_to_modify, dim= -1)
            X_to_modify[torch.rand(X_to_modify.shape)<self.p] = 0
            X_to_modify = torch.add(X_to_modify, _reshape_fortran(((X_power - torch.sum(X_to_modify, dim= -1))/X_to_modify.shape[-1]).repeat(1, 1, X_to_modify.shape[-1]), X_to_modify.shape))

        X[:, :, :, self.include_index] = X_to_modify

        return X

# implements the attention layer as proposed by Vaswani et al. (2017)
class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, batch_first):
        super(Attention, self).__init__()
        self.scale = 1. / math.sqrt(query_dim)
        self.batch_first = batch_first

    def forward(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)
        if self.batch_first:
            keys = torch.moveaxis(keys, 0, 1)
            values = torch.moveaxis(values, 0, 1)

        query = query.unsqueeze(1)  # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0, 1).transpose(1, 2)  # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys)  # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = torch.nn.functional.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize

        values = values.transpose(0, 1)  # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1)  # [Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination

# model to combine aggregated w2v embedding over time
class w2vSumTextProcessor(nn.Module):
    def __init__(self, env, input_dim, output_dim, use_price, use_attention, rnn_layers, linear_layers, dropout_prop):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.assets = env.labels
        self.embed_size = env.embed_dim
        self.num_assets = len(self.assets)
        self.window_size = env.window_size
        self.embed_size_with_cc = self.embed_size + self.num_assets
        self.size_lag = self.embed_size * self.num_assets + self.num_assets

        self.use_price = use_price
        self.use_attention = use_attention
        self.rnn_layers = rnn_layers
        self.linear_layers = linear_layers
        self.dropout_prop = dropout_prop

        self.partial_dropout = Dropout_partial_binary(list(range(self.num_assets)), self.dropout_prop)

        if self.use_price:
            self.gru = nn.GRU(self.embed_size_with_cc + 2, self.embed_size_with_cc, self.rnn_layers, bidirectional=False, batch_first=True)
        else:
            self.gru = nn.GRU(self.embed_size_with_cc + 1, self.embed_size_with_cc, self.rnn_layers, bidirectional=False, batch_first=True)

        if self.use_attention:
            self.attention = Attention(self.embed_size_with_cc, self.embed_size_with_cc, self.embed_size_with_cc, batch_first=True)

        self.out = nn.ModuleList()
        for i in range(self.linear_layers - 1):
            self.out.append(nn.Linear(self.embed_size_with_cc * self.num_assets, self.embed_size_with_cc * self.num_assets))
        self.out.append(nn.Linear(self.embed_size_with_cc * self.num_assets, self.output_dim))

    def forward(self, state):
        device = state.device
        batch_size = state.shape[0]
        state = state.reshape((state.shape[0], self.window_size, -1)).cpu().numpy()

        state_positions = state[:, :, :self.num_assets]
        state_positions = np.swapaxes(state_positions, 1, 2)
        state_positions = torch.tensor(state_positions).to(device, dtype=torch.float)

        state = state[:, :, self.num_assets:]
        state = np.reshape(state, (state.shape[0], -1), order="C")
        lag = int(state.shape[1] / self.size_lag)
        # seperate price and text
        text_ind = np.array([])
        for i in range(lag):
            text_ind = np.append(text_ind, np.r_[(self.size_lag * i) + self.num_assets: (self.size_lag * i) + self.num_assets + (self.num_assets * self.embed_size)])
        text_ind = text_ind.astype(int)
        state_text = state[:, text_ind]
        temp = []
        #seperate w2v vectors
        for i in range(self.num_assets):
            temp_ind = np.array([])
            for j in range(lag):
                #temp_ind = np.append(temp_ind, np.r_[(self.size_lag - self.num_assets) * j + (self.embed_size * i): (self.size_lag - self.num_assets) * j + (self.embed_size * i) + self.embed_size])
                temp_ind = np.append(temp_ind, np.r_[(self.size_lag - self.num_assets) * j + i: (self.size_lag - self.num_assets) * (j + 1): self.num_assets])
            temp_ind = temp_ind.astype(int)
            temp.append(state_text[:, temp_ind])
        state_text = np.array(temp)
        del temp
        state_text = np.moveaxis(state_text, 0, 1)
        #state_text = np.split(state_text, np.cumsum([int(state_text.shape[2] / self.embed_size)] * self.embed_size),axis=2)[:-1]
        state_text = np.split(state_text, np.cumsum([self.embed_size] * int(state_text.shape[-1] / self.embed_size)), axis=2)[:-1]
        state_text = np.array(state_text)
        state_text = np.moveaxis(state_text, 0, 1)
        temp = []
        for i, cc in enumerate(self.assets):
            X_text_new_temp = []
            for j in range(self.window_size):
                X_text_new_temp.append(np.apply_along_axis(self._create_w2v_input, 1, state_text[:, j, i, :], cc))
            temp.append(np.stack(X_text_new_temp, axis=1))
        state_text = np.stack(temp, axis=1)
        del temp
        state_text[np.isnan(state_text)] = 0
        state_text = torch.tensor(state_text).to(device, dtype=torch.float)
        state_text = self.partial_dropout(state_text)

        if self.use_price:
            price_ind = np.array([])
            for i in range(lag):
                price_ind = np.append(price_ind, np.r_[(self.size_lag * i): (self.size_lag * i) + self.num_assets])
            price_ind = price_ind.astype(int)
            state_price = state[:, price_ind]
            state_price = np.reshape(state_price, [*state_price.shape[:-1], int(self.num_assets), int(lag)])
            #state_price = state_price.reshape([-1, *state_price.shape[2:]], order="F")
            #state_price = np.swapaxes(state_price, 1, 2)
            state_price = torch.tensor(state_price).to(device, dtype=torch.float)
            combined = torch.cat((state_text, torch.unsqueeze(state_price, -1), torch.unsqueeze(state_positions, -1)), -1)
        else:
            combined = torch.cat((state_text, torch.unsqueeze(state_positions, -1)), -1)

        combined = _reshape_fortran(combined, [-1, *combined.shape[2:]])

        self.gru.flatten_parameters()
        outputs, hidden = self.gru(combined)
        if isinstance(hidden, tuple):
            hidden = hidden[1]
        hidden = hidden[-1]

        if self.use_attention:
            energy, hidden = self.attention(hidden, outputs, outputs)

        hidden = _reshape_fortran(hidden, [batch_size, -1, *hidden.shape[1:]])
        hidden = hidden.flatten(start_dim= 1)
        for layer in self.out:
            hidden = layer(hidden)

        return hidden

    def _create_w2v_input(self, data, cc):
        cc_array = np.zeros(self.num_assets)
        if np.sum((~np.isnan(data)).astype(int)) == 0:
            cc_array[:] = np.nan
        else:
            cc_ind = self.assets.index(cc)
            cc_array[cc_ind] = 1
        return np.concatenate((cc_array, data))

# create model to combine hidden states via gru layers (bert model for hidden states is part of this model, however will not be trained)
class berRnnSeperateTextProcessor(nn.Module):
    def __init__(self, env, input_dim, output_dim, embed_model_dim, embed_dim, bert_model, use_price, use_attention, rnn_layers, linear_layers, dropout_prop):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_model_dim = embed_model_dim
        self.embed_dim = embed_dim
        self.assets = env.labels
        self.num_assets = len(self.assets)
        self.window_size = env.window_size
        self.num_texts_per_day = env.texts_per_day
        self.num_words_per_text = env.words_per_text
        self.use_price = use_price
        self.size_lag = env.observation_shape[1] - self.num_assets
        self.embed_size = (self.size_lag - self.num_assets) / self.num_assets
        self.embed_size_with_cc = self.embed_size + self.num_assets
        self.use_attention = use_attention
        self.rnn_layers = rnn_layers
        self.linear_layers = linear_layers
        self.dropout_prop = dropout_prop

        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.tokenizer.add_tokens(self.assets)
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.bert_model.resize_token_embeddings(len(self.tokenizer))

        self.embed_gru = nn.GRU(self.embed_model_dim, self.embed_dim, self.rnn_layers, batch_first=True)

        if self.use_price:
            self.gru = nn.GRU(self.embed_dim + 2, self.embed_dim, self.rnn_layers, bidirectional=False, batch_first=True)
        else:
            self.gru = nn.GRU(self.embed_dim + 1, self.embed_dim, self.rnn_layers, bidirectional=False, batch_first=True)

        if self.use_attention:
            self.embed_attention = Attention(self.embed_dim + 1, self.embed_dim + 1, self.embed_dim + 1, batch_first=True)
            self.attention = Attention(self.embed_dim, self.embed_dim, self.embed_dim, batch_first=True)

        self.out = nn.ModuleList()
        for i in range(self.linear_layers - 1):
            self.out.append(nn.Linear(self.embed_dim * self.num_assets, self.embed_dim * self.num_assets))
        self.out.append(nn.Linear(self.embed_dim * self.num_assets, self.output_dim))

    def forward(self, state):
        device = state.device
        batch_size = state.shape[0]
        state = state.reshape((state.shape[0], self.window_size, -1)).cpu().numpy()

        state_positions = state[:, :, :self.num_assets]
        state_positions = np.swapaxes(state_positions, 1, 2)
        state_positions = torch.tensor(state_positions).to(device, dtype=torch.float)

        state = state[:, :, self.num_assets:]
        state = np.reshape(state, (state.shape[0], -1), order="C")
        lag = int(state.shape[1] / self.size_lag)
        # seperate price and text
        text_ind = np.array([])
        for i in range(lag):
            text_ind = np.append(text_ind, np.r_[(self.size_lag * i) + self.num_assets: (self.size_lag * i) + self.num_assets + (self.num_assets * self.embed_size)])
        text_ind = text_ind.astype(int)
        state_text = state[:, text_ind]
        temp = []
        #seperate w2v vectors
        #for i in range(self.num_assets):
        #    temp_ind = np.array([])
        #    for j in range(lag):
        #        #temp_ind = np.append(temp_ind, np.r_[(self.size_lag - self.num_assets) * j + (self.embed_size * i): (self.size_lag - self.num_assets) * j + (self.embed_size * i) + self.embed_size])
        #        temp_ind = np.append(temp_ind, np.r_[(self.size_lag - self.num_assets) * j + i: (self.size_lag - self.num_assets) * (j + 1): self.num_assets])
        #    temp_ind = temp_ind.astype(int)
        #    temp.append(state_text[:, temp_ind])
        #state_text = np.array(temp)
        #del temp
        state_text = state_text.reshape(state_text.shape[0], self.window_size, self.num_assets, self.num_texts_per_day, self.num_words_per_text)

        self.bert_model.eval()
        with torch.no_grad():
            temp1 = []
            for i, cc in enumerate(self.assets):
                temp2 = []
                for j in range(self.window_size):
                    temp3 = []
                    for k in range(self.num_texts_per_day):
                        temp3.append(np.apply_along_axis(self._create_bert_input, 1, state_text[:, j, i, k, :], cc, self.dropout_prop))
                    temp3 = np.stack(temp3, axis=1)
                    temp3[np.isnan(temp3)] = 0
                    temp3 = torch.tensor(temp3).to(device, dtype=torch.long)
                    old_shape = temp3.shape
                    temp3 = torch.reshape(temp3, (-1, *temp3.shape[2:]))
                    temp3 = self.bert_model(
                        input_ids=temp3[:, 0, :],
                        token_type_ids=temp3[:, 1, :],
                        attention_mask=temp3[:, 2, :]
                    )
                    temp3 = temp3["last_hidden_state"][:, 0, :]
                    temp3 = torch.reshape(temp3, (*old_shape[:2], -1))
                    temp2.append(temp3)
                temp1.append(torch.stack(temp2, dim=1))
            state_text = torch.stack(temp1, dim=1)
            del temp1, temp2, temp3

        if self.use_price:
            price_ind = np.array([])
            for i in range(lag):
                price_ind = np.append(price_ind, np.r_[(self.size_lag * i): (self.size_lag * i) + self.num_assets])
            price_ind = price_ind.astype(int)
            state_price = state[:, price_ind]
            state_price = np.reshape(state_price, [*state_price.shape[:-1], int(self.num_assets), int(lag)])
            #state_price = state_price.reshape([-1, *state_price.shape[2:]], order="F")
            #state_price = np.swapaxes(state_price, 1, 2)
            state_price = torch.tensor(state_price).to(device, dtype=torch.float)

        self.embed_gru.flatten_parameters()
        self.gru.flatten_parameters()

        temp1 = []
        for i, cc in enumerate(self.assets):
            temp2 = []
            for j in range(self.window_size):
                outputs, hidden = self.embed_gru(state_text[:, i, j, :])
                if isinstance(hidden, tuple):
                    hidden = hidden[1]
                hidden = hidden[-1]
                if self.use_attention:
                    energy, hidden = self.embed_attention(hidden, outputs, outputs)
                temp2.append(hidden)
            temp2 = torch.stack(temp2, dim=1)
            if self.use_price:
                combined = torch.cat((temp2, torch.unsqueeze(state_price[:, i, :], -1), torch.unsqueeze(state_positions[:, i, :], -1)),-1)
            else:
                combined = torch.cat((temp2, torch.unsqueeze(state_positions[:, i, :], -1)), -1)
            outputs, hidden = self.gru(combined)
            if isinstance(hidden, tuple):
                hidden = hidden[1]
            hidden = hidden[-1]
            if self.use_attention:
                energy, hidden = self.attention(hidden, outputs, outputs)
            temp1.append(hidden)
        combined = torch.stack(temp1, dim=1)

        combined = combined.flatten(start_dim=1)
        for layer in self.out:
            hidden = layer(combined)

        return hidden

    def _create_bert_input(self, data, cc, dropout= 0):
        if np.sum((~np.isnan(data)).astype(int)) == 0:
            input_ids = np.concatenate(([np.nan], data))
            token_type_ids = np.array([0] * (len(data) + 1))
            attention_mask = np.concatenate(([0], (~np.isnan(data)).astype(int)))
        else:
            if random.random() > dropout:
                input_ids = np.concatenate(([data[0], self.tokenizer("token")["input_ids"][1]], data[1:]))
            else:
                input_ids = np.concatenate(([data[0], self.tokenizer(cc)["input_ids"][1]], data[1:]))
            token_type_ids = np.array([0, 1] + [0] * (len(data) - 1))
            attention_mask = np.concatenate(([1], (~np.isnan(data)).astype(int)))
        return np.array([input_ids, token_type_ids, attention_mask])