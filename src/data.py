from collections import Counter
import re

import numpy as np
from nltk.corpus import stopwords
import torch.utils.data
import tqdm

UNKNOWN_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

UNKNOWN_TOKEN_ID = 0
PAD_TOKEN_ID = 1
NUM_PREDEFINED_VOCAB_TERMS = 2


def read_doc_as_tokenized_lines(doc_path):
    with open(doc_path, "r") as fin:
        lines = fin.readlines()
    tokenized_lines = [line.split() for line in tqdm.tqdm(lines)]
    return tokenized_lines


def remove_empty_entries(datasets):
    is_single_dataset = isinstance(datasets, np.ndarray)
    if is_single_dataset:
        # wrap
        datasets = [datasets]

    # remove entries that are all padding
    all_padding = np.repeat(False, datasets[0].shape[0])
    for ds in datasets:
        all_padding |= np.all(ds == PAD_TOKEN_ID, axis=1)

    if all_padding.any():
        ct = all_padding.sum()
        print("Dropping {} entries, which are all padding".format(ct))

    valid_entries = ~all_padding
    clean_datasets = []
    for ds in datasets:
        clean_ds = ds[valid_entries]
        clean_datasets.append(clean_ds)

    if is_single_dataset:
        # unwrap
        clean_datasets = clean_datasets[0]
    return clean_datasets


class VocabEncoder(object):
    """Wrapper to store vocab and encode/decode data into ids"""

    def __init__(self, vocab):
        # default values for unknown and padding
        self.vocab_map = {
            UNKNOWN_TOKEN: UNKNOWN_TOKEN_ID,
            PAD_TOKEN: PAD_TOKEN_ID
        }
        # add in new vocabulary to mapping w/ appropriate offset
        new_vocab_map = {
            token: (ix + NUM_PREDEFINED_VOCAB_TERMS)
            for ix, token in enumerate(vocab)
        }
        self.vocab_map.update(new_vocab_map)
        self.inverse_vocab_map = {
            _id: token
            for token, _id in self.vocab_map.items()
        }

    def _encode(self, tokens):
        return [self.vocab_map.get(t, UNKNOWN_TOKEN_ID) for t in tokens]

    def encode(self, _input):
        if isinstance(_input[0], list):
            return [self._encode(row) for row in _input]
        return self._encode(_input)

    def _decode(self, ids):
        return [self.inverse_vocab_map[_id] for _id in ids]

    def decode(self, _input):
        if isinstance(_input[0], list):
            return [self._decode(row) for row in _input]
        return self._decode(_input)


def build_vocab(doc, top_k=None, min_freq=None):
    if top_k is None and min_freq is None:
        raise ValueError("Provide one of top_k or min_freq")
    cter = Counter()
    for tokenized_line in tqdm.tqdm(doc):
        cter.update(tokenized_line)
    if top_k:
        vocab = cter.most_common(top_k)
        return [token for token, _ in vocab]
    if min_freq:
        vocab = [token for token, ct in cter.items() if ct >= min_freq]
        return vocab


def pad(seq, target_len, pad_token):
    n = len(seq)
    if n >= target_len:
        return seq[:target_len]
    need = target_len - n
    padding = np.repeat(pad_token, need)
    return np.append(seq, padding)


def create_padded_dataset(dataset, target_len):
    m = [pad(seq, target_len, PAD_TOKEN_ID) for seq in dataset]
    return np.array(m)


def get_random_ix(n, ix):
    randix = np.random.randint(0, n)
    if randix != ix:
        return randix
    else:
        return get_random_ix(n, ix)


class CodeSearchDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            code,
            docstrings,
    ):
        self.code = code
        self.docstrings = docstrings
        assert self.code.shape[0] == self.docstrings.shape[0], "Row mismatch"
        self.num_obs = self.code.shape[0]

    def __len__(self):
        return self.num_obs

    def __getitem__(self, ix):
        code = self.code[ix]
        docstring = self.docstrings[ix]
        randix = get_random_ix(self.num_obs, ix)
        fake_docstring = self.docstrings[randix]
        return code, docstring, fake_docstring
