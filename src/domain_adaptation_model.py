import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pdb

from . import utils


class ClassifierAdaModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, embedding_layer, nl):
        emb_nl = embedding_layer(nl)
        out = self.layers(emb_nl)
        return F.log_softmax(out)

    def loss(self, embedding_layer, nl, label):
        score = self.forward(embedding_layer, nl)
        return F.nll_loss(score, label, reduce=True)


class WassersteinAdaModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.weight_init()

    def weight_init(self):
        for layer in self.layers:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(layer.bias, 0.0)


    def clip_weights(self, val_range=0.01):
        for p in self.parameters():
            p.data.clamp_(min=-val_range, max=val_range)

    def forward(self, embedding_layer, nl):
        # nl:(batch_size, seq_length); embedding_layer: encoder model
        self.clip_weights()
        emb_nl = embedding_layer(nl) # (batch_size, n_d)
        return self.layers(emb_nl)

    def loss(self, embedding_layer, nl, label):
        score = self.forward(embedding_layer, nl).squeeze() # (batch_size,)
        label = label.float() * 2 - 1
        # goal: get things close to zero
        # i.e have to "move" zero dirt to get distributions to be equal
        loss = (score * label).mean()
        return loss


class AdversarialWrapper(object):
    def __init__(self, config, encoder_output_size, data_loader, embedding_fun):
        self.initialize_model(config, encoder_output_size)
        self.data_loader = data_loader
        self.embedding_fun = embedding_fun

    def initialize_model(self, config, encoder_output_size):
        ada_weight = config["ada_weight"]
        ada_lr = config["ada_lr"]
        hidden_size = config["ada_hidden_size"]
        ada_model_type = config["ada_model_type"]

        if ada_model_type == "classifier":
            self.model = ClassifierAdaModel(
                encoder_output_size,
                hidden_size,
            )
        elif ada_model_type == "wasserstein":
            self.model = WassersteinAdaModel(
                encoder_output_size,
                hidden_size,
            )
            self.weight = -ada_weight
        else:
            raise ValueError("Unknown ADA type: {}".format(ada_model_type))
        if utils.have_gpu():
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(utils.get_trainable_params(self.model), ada_lr)

    def train_iteration(self):
        # Note: we sample *new* docstrings, not the ones we just used
        orig_samples, target_samples = next(iter(self.data_loader))
        target_labels = torch.ones(target_samples.shape[0], dtype=torch.long)
        orig_labels = torch.zeros(
            orig_samples.shape[0], dtype=torch.long
        )
        ada_batch = torch.cat((orig_samples, target_samples), dim=0)
        ada_labels = torch.cat((orig_labels, target_labels))
        ada_labels = ada_labels.to(device=ada_batch.device)
        ada_loss = self.model.loss(
            self.embedding_fun,
            ada_batch,
            ada_labels,
        )
        return ada_loss, self.weight

    def step(self):
        utils.negate_gradient(self.optimizer)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
