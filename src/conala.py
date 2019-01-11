# Convert the JSON format file from CONALA to canonical format
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


def load(json_path):
    parsed = parse_json(json_path)
    obs = []
    for entry in tqdm.tqdm(parsed):
        obs.append({
            "code": entry["snippet"].strip(),
            "nl": entry["intent"].strip()
        })
    return obs


def get_args():
    parser = argparse.ArgumentParser(description="CONALA to canonical")
    parser.add_argument("-i", "--input", type=str, help="Path to input json")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for pickled",
    )
    return parser.parse_args()


def main():
    args = get_args()
    data = load(args.input)
    corpus = CanonicalInput(data)
    with open(args.output, "wb") as fout:
        pickle.dump(corpus, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
