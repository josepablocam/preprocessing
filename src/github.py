# Convert the Github files into canonical format
import argparse
import ast
import astunparse
import json
import tqdm
import pickle

from .preprocess import CanonicalInput


def get_code_and_nl(src):
    try:
        tree = ast.parse(src)
        func = tree.body[0]
        doc = ast.get_docstring(func)
        # strip the docstring from the code body
        first_elem = func.body[0].value
        if (isinstance(first_elem, ast.Str)
                or (isinstance(first_elem, ast.Constant)
                    and isinstance(first_elem.value, str))):
            func.body = func.body[1:]
        return astunparse.unparse(func).strip(), doc.strip()
    except SyntaxError:
        return None, None


def load(json_path):
    with open(json_path, "r") as fin:
        data = json.load(fin)
    obs = []
    for entry in tqdm.tqdm(data):
        code, nl = get_code_and_nl(entry)
        if code is None or nl is None:
            continue
        obs.append({"code": code, "nl": nl})
    return obs


def get_args():
    parser = argparse.ArgumentParser(description="Github to canonical")
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
