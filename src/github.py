# Convert the Github files into canonical format
import argparse
import json
import tqdm
import pickle

from .preprocess import CanonicalInput


def parse_json_lines(json_path):
    """Parse file where each line is json"""
    parsed = []
    with open(json_path, "r") as fin:
        for line in tqdm.tqdm(fin):
            parsed_line = json.loads(line)
            parsed.append(parsed_line)
    return parsed


def parse_json(json_path):
    try:
        with open(json_path, "r") as fin:
            return json.load(fin)
    except json.JSONDecodeError:
        return parse_json_lines(json_path)


def load(code_json_path, nl_path):
    code = parse_json(code_json_path)
    with open(nl_path) as fin:
        nl = fin.readlines()
    obs = []
    for code, nl in tqdm.tqdm(zip(code, nl)):
        obs.append({"code": code.strip(), "nl": nl.strip()})
    return obs


def get_args():
    parser = argparse.ArgumentParser(description="Github to canonical")
    parser.add_argument("-c", "--code", type=str, help="Path to code json")
    parser.add_argument("-n", "--nl", type=str, help="Path to NL text")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for pickled",
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = load(args.code, args.nl)
    corpus = CanonicalInput(data)
    with open(args.output, "wb") as fout:
        pickle.dump(corpus, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
