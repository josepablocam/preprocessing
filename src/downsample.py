import argparse
import os
import numpy as np


def sample_ixs(total, target, replace):
    ixs = np.arange(0, total)
    return np.random.choice(ixs, size=target, replace=replace)


def sample_from_files(files, target, replace, output_dir):
    lines_by_file = {}
    for f in files:
        with open(f, "r") as fin:
            lines = fin.readlines()
            lines_by_file[f] = lines

    _lens = [len(ls) for ls in lines_by_file.values()]
    assert len(set(_lens)) == 1, "All files must be of the same size"
    total = _lens[0]
    ixs = sample_ixs(total, target, replace)
    for f, lines in lines_by_file.items():
        new_f_path = os.path.join(output_dir, os.path.basename(f))
        sampled_lines = [lines[ix].strip() for ix in ixs]
        with open(new_f_path, "w") as fout:
            fout.write("\n".join(sampled_lines))


def get_args():
    parser = argparse.ArgumentParser(description="Sample from aligned files")
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        help="List of aligned files",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=int,
        help="Number of lines to sample",
        default=int(10e3),
    )
    parser.add_argument(
        "-r",
        "--replace",
        action="store_true",
        help="Sample with replacement",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="RNG seed",
        default=42,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Output directory to store sampled files",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if not os.path.exists(args.output_dir):
        print("Creating {}".format(args.output_dir))
        os.mkdir(args.output_dir)
    np.random.seed(args.seed)
    sample_from_files(args.files, args.target, args.replace, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
