#!/usr/bin/env python3
import argparse
import tqdm

from . import data


def remove_empty(input_files, output_files):
    read_handles = [open(f, "r") for f in input_files]
    write_handles = [open(f, "w") for f in output_files]

    docs = [h.readlines() for h in read_handles]

    n = len(docs[0])
    if any(len(d) != n for d in docs):
        raise Exception("Files must have the same number of lines")

    for i in tqdm.tqdm(range(0, n)):
        lines = [d[i].strip() for d in docs]
        # note that the length we care about is after we have
        # split into subtokens and removed stopwords etc
        lens = [len(data.split_into_subtokens(l)) for l in lines]
        if min(lens) != 0:
            # all lines are complete
            for fout, line in zip(write_handles, lines):
                fout.write(line + "\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="""
        Remove lines that are empty in any of the inputs. Keep files in synch
        """
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        help="Input files",
    )
    parser.add_argument(
        "-o",
        "--output",
        nargs="+",
        type=str,
        help="Output files",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    remove_empty(args.input, args.output)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
