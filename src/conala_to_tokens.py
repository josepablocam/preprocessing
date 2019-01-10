# Convert the JSON format file from CONALA to just text
import argparse
import json
import tqdm

from . import data


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


def convert(json_path, code_path, queries_path, min_prob=None):
    parsed = parse_json(json_path)
    code = []
    queries = []
    for entry in tqdm.tqdm(parsed):
        if min_prob is not None and entry["prob"] < min_prob:
            continue
        c = data.split_into_subtokens(entry["snippet"])
        q = data.split_into_subtokens(entry["intent"])
        code.append(" ".join(c))
        queries.append(" ".join(q))

    with open(code_path, "w") as fout:
        for line in code:
            fout.write(line + "\n")

    with open(queries_path, "w") as fout:
        for line in queries:
            fout.write(line + "\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert mined CONALA to text files"
    )
    parser.add_argument(
        "-j", "--json_path", type=str, help="Path to json (or jsonl) file"
    )
    parser.add_argument(
        "-c", "--code_path", type=str, help="Path to store code text file"
    )
    parser.add_argument(
        "-q", "--queries_path", type=str, help="Path to queries text file"
    )
    parser.add_argument(
        "-m",
        "--min_prob",
        type=float,
        help="Minimum probability for mined snippet",
        default=None,
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    convert(
        args.json_path,
        args.code_path,
        args.queries_path,
        args.min_prob,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb
        pdb.post_mortem()
