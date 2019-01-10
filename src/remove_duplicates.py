#!/bin/usr/env python3

import argparse


def write_out(path, ixs, suffix):
    new_path = path + suffix
    with open(new_path, 'w') as fout, open(path, 'r') as fin:
        for ix, line in enumerate(fin):
            if ix in ixs:
                fout.write(line)


def remove_duplicates(reference_path, other_paths, suffix=".unique"):
    unique_ixs = []
    seen = set()
    with open(reference_path, "r") as fin:
        for ix, line in enumerate(fin):
            line = line.strip()
            if line not in seen:
                seen.add(line)
                unique_ixs.append(ix)

    paths = [reference_path] + other_paths
    for p in paths:
        write_out(p, unique_ixs, suffix)


def get_args():
    parser = argparse.ArgumentParser(
        description="Remove duplicates based on a reference file"
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        help="Reference path (lines are unique w.r.t to this file)",
    )
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        nargs="+",
        help="Other files",
    )
    parser.add_argument(
        "-s",
        "--suffix",
        type=str,
        help="Suffix to append to new files",
        default=".unique",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    remove_duplicates(args.reference, args.paths, args.suffix)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
