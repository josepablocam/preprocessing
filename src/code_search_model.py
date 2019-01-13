import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import configuration
from . import data
from . import precomputed_embeddings
from . import utils


def _load_precomputed_embeddings(emb_layer, vocab_encoder, embeddings_file):
    # we may not have an embedding for every entry
    # so we load them one by one
    precomputed = precomputed_embeddings.read_embeddings(embeddings_file)
    with torch.no_grad():
        for i in range(0, emb_layer.weight.shape[0]):
            decoded_token = vocab_encoder.decode([i])[0]
            if decoded_token in precomputed:
                emb = precomputed[decoded_token]
                emb_layer.weight[i] = torch.tensor(emb)


def setup_precomputed_embeddings(
        model,
        vocab_encoder_path,
        embeddings_path,
):

    with open(vocab_encoder_path, "rb") as fin:
        vocab_encoder = pickle.load(fin)

    # shared embeddings to start
    _load_precomputed_embeddings(
        model.code_embeddings,
        vocab_encoder,
        embeddings_path,
    )

    _load_precomputed_embeddings(
        model.nl_embeddings,
        vocab_encoder,
        embeddings_path,
    )


def avg_ignore_padding(emb, _input):
    is_not_pad = (_input != data.PAD_TOKEN_ID).to(torch.float)
    denom = is_not_pad.sum(dim=1).unsqueeze(dim=1)
    num = torch.sum(emb * is_not_pad.unsqueeze(dim=2), dim=1)
    avg_emb = num / denom
    return avg_emb


class CodeDocstringModel(nn.Module):
    def __init__(
            self,
            margin,
            code_vocab_size,
            nl_vocab_size,
            emb_size,
            fixed_embeddings=False,
            num_dan_layers=0,
            same_embedding_fun=False,
    ):
        super().__init__()
        self.margin = margin
        self.code_embeddings = nn.Embedding(
            code_vocab_size + data.NUM_PREDEFINED_VOCAB_TERMS,
            emb_size,
            padding_idx=data.PAD_TOKEN_ID,
        )
        self.code_attention = nn.Parameter(torch.randn(1, emb_size))

        if same_embedding_fun:
            self.nl_embeddings = self.code_embeddings
            self.nl_attention = self.code_attention
        else:
            self.nl_embeddings = nn.Embedding(
                nl_vocab_size + data.NUM_PREDEFINED_VOCAB_TERMS,
                emb_size,
                padding_idx=data.PAD_TOKEN_ID,
            )
            self.nl_attention = nn.Parameter(torch.randn(1, emb_size))

        if fixed_embeddings:
            self.code_embeddings.weight.requires_grad = False
            self.nl_embeddings.weight.requires_grad = False

        self.dan_layers = None

        if num_dan_layers > 0:
            self.dan_layers = nn.Sequential()
            while num_dan_layers > 0:
                lin_layer = nn.Linear(emb_size, emb_size, bias=True)
                lin_name = 'lin-{}'.format(num_dan_layers)
                self.dan_layers.add_module(lin_name, lin_layer)
                if num_dan_layers > 1:
                    activation = nn.ReLU()
                    activation_name = 'activation-{}'.format(num_dan_layers)
                    self.dan_layers.add_module(activation_name, activation)
                num_dan_layers -= 1

        self.output_size = emb_size

    def _embed(self, _input, emb_layer, attn):
        # batch_size x num_tokens x embedding_size
        emb = emb_layer(_input)
        avg_emb = avg_ignore_padding(emb, _input)

        if self.dan_layers is None:
            return avg_emb
        return self.dan_layers(avg_emb)

    def embed_code(self, code):
        embs = self._embed(code, self.code_embeddings, self.code_attention)
        return embs

    def embed_nl(self, nl):
        embs = self._embed(nl, self.nl_embeddings, self.nl_attention)
        return embs

    def forward(self, code, nl, fake_nl):
        return self.embed_code(code), self.embed_nl(nl), self.embed_nl(fake_nl)

    def losses(self, code, nl, fake_nl):
        code_emb, nl_emb, fake_nl_emb = self.forward(code, nl, fake_nl)
        good_sim = F.cosine_similarity(code_emb, nl_emb, dim=1)
        bad_sim = F.cosine_similarity(code_emb, fake_nl_emb, dim=1)
        losses = bad_sim + self.margin - good_sim
        return losses.clamp(min=0.0)


class LSTMModel(nn.Module):
    """LSTM model"""
    def __init__(
        self,
        margin,
        code_vocab_size,
        nl_vocab_size,
        emb_size,
        hidden_size,
        bidirectional=False,
        num_layers=1,
        fixed_embeddings=False,
        same_embedding_fun=False,
    ):
        super().__init__()
        self.margin = margin
        self.code_embeddings = nn.Embedding(
            code_vocab_size + data.NUM_PREDEFINED_VOCAB_TERMS,
            emb_size,
            padding_idx=data.PAD_TOKEN_ID,
        )
        self.code_lstm = nn.LSTM(
            input_size = emb_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )
        if same_embedding_fun:
            self.nl_embeddings = self.code_embeddings
            self.nl_lstm = self.code_lstm
        else:
            self.nl_embeddings = nn.Embedding(
                nl_vocab_size + data.NUM_PREDEFINED_VOCAB_TERMS,
                emb_size,
                padding_idx=data.PAD_TOKEN_ID,
            )
            self.nl_lstm = nn.LSTM(
                input_size = emb_size,
                hidden_size = hidden_size,
                num_layers = num_layers,
                bidirectional = bidirectional,
                batch_first = True
            )

        if fixed_embeddings:
            self.code_embeddings.weight.requires_grad = False
            self.nl_embeddings.weight.requires_grad = False

        if bidirectional:
            self.output_size = 2*hidden_size
        else:
            self.output_size = hidden_size

    def _embed(self, _input, emb_layer, lstm_layer):
        embedded = emb_layer(_input) # (batch_size, num_tokens, emb_dim)
        mask = (_input != data.PAD_TOKEN_ID).to(torch.float) # (batch_size, num_tokens)
        # Pack input
        lengths_tensor, indices = torch.sort(mask.int().sum(1), descending=True)
        if lengths_tensor.is_cuda:
            lengths = lengths_tensor.data.cpu().numpy().tolist() # (batch_size)
        else:
            lengths = lengths_tensor.data.numpy().tolist()
        packed_inputs = nn.utils.rnn.pack_padded_sequence(embedded[indices], lengths, batch_first=True)
        # Compute
        packed_outputs, final_hidden_state = lstm_layer(packed_inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True) # (batch_size, num_tokens, num_dir*hidden_dim)
        # Get output in original order
        _, re_order_indices = torch.sort(indices)
        return torch.mean(outputs[re_order_indices],1) # (batch_size, num_dir*hidden_dim)

    def embed_code(self, code):
        # code: (batch_size, num_tokens)
        # return: (batch_size, out_dim)
        embs = self._embed(code, self.code_embeddings, self.code_lstm)
        return embs

    def embed_nl(self, nl):
        # nl: (batch_size, num_tokens)
        # return: (batch_size, out_dim)
        embs = self._embed(nl, self.nl_embeddings, self.nl_lstm)
        return embs

    def forward(self, code, nl, fake_nl):
        return self.embed_code(code), self.embed_nl(nl), self.embed_nl(fake_nl)

    def losses(self, code, nl, fake_nl):
        code_emb, nl_emb, fake_nl_emb = self.forward(code, nl, fake_nl)
        good_sim = F.cosine_similarity(code_emb, nl_emb, dim=1)
        bad_sim = F.cosine_similarity(code_emb, fake_nl_emb, dim=1)
        losses = bad_sim + self.margin - good_sim
        return losses.clamp(min=0.0)


class UnsupervisedModel(nn.Module):
    def __init__(
            self,
            code_vocab_size,
            nl_vocab_size,
            emb_size,
            same_embedding_fun=False,
    ):
        super().__init__()
        self.code_embeddings = nn.Embedding(
            code_vocab_size + data.NUM_PREDEFINED_VOCAB_TERMS,
            emb_size,
            padding_idx=data.PAD_TOKEN_ID,
        )
        if same_embedding_fun:
            self.nl_embeddings = self.code_embeddings
        else:
            self.nl_embeddings = nn.Embedding(
                nl_vocab_size + data.NUM_PREDEFINED_VOCAB_TERMS,
                emb_size,
                padding_idx=data.PAD_TOKEN_ID,
            )

    def embed_code(self, code):
        embs = self.code_embeddings(code)
        return avg_ignore_padding(embs, code)
        return embs.mean(dim=1)

    def embed_nl(self, nl):
        embs = self.nl_embeddings(nl)
        return avg_ignore_padding(embs, nl)


def store_unsupervised_model(
        config, code_vocab_encoder_path, nl_vocab_encoder_path,
        embeddings_path, output_path
):
    model = UnsupervisedModel(
        config["code_vocab_top_k"],
        config["nl_vocab_top_k"],
        config["embedding_size"],
    )
    setup_precomputed_embeddings(
        model, code_vocab_encoder_path, nl_vocab_encoder_path, embeddings_path
    )
    torch.save(model, output_path)


def get_args():
    parser = argparse.ArgumentParser(description="Store unsupervised model")
    parser.add_argument("config", type=str, help="Model config")
    parser.add_argument(
        "-e",
        "--embeddings_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-cv",
        "--code_vocab_encoder_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-nv",
        "--nl_vocab_encoder_path",
        type=str,
        help=None,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to store pickled"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    config = configuration.get_configuration(args.config)
    store_unsupervised_model(
        config,
        args.code_vocab_encoder_path,
        args.nl_vocab_encoder_path,
        args.embeddings_path,
        args.output,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
