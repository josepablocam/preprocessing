#!/usr/bin/env python3
import argparse
from collections import Counter
import pandas as pd
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import scipy


def vocab_counts(input_path):
    counter = Counter()
    with open(input_path, "r") as fin:
        for line in tqdm.tqdm(fin):
            words = line.split()
            counter.update(words)
    freqs = list(counter.values())
    return np.array(freqs)

def distribution(cts, cutoffs):
    fracs = []
    for cutoff in cutoffs:
        frac = np.mean(cts >= cutoff)
        fracs.append(frac)
    return fracs





cts = vocab_counts("embedding-input.txt")
cutoffs = list(range(1, 100, 1))
fractions = distribution(cts, cutoffs)
fig, ax = plt.subplots(1)
ax.plot(cutoffs, fractions)
ax.set_xlabel("Minimum frequency of token")
ax.set_ylabel("(size of new vocabulary) / (size of original vocabulary)")
fig.savefig("vocab_cutoff_distribution.pdf")
