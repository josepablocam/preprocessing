# Convert the Github files into canonical format
import argparse
import ast
import astunparse
import json
import tqdm
import pickle

from .preprocess import CanonicalInput

PASS = ast.parse("pass").body[0]


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
            if len(func.body) > 1:
                # if there are other statements, take the rest
                func.body = func.body[1:]
            else:
                # otherwise add a dummy pass statement
                func.body = [PASS]
        # remove any zero bytes (otherwise parsing can fail)
        code_str = astunparse.unparse(func).strip().replace(chr(0), '')
        doc_str = doc.strip().replace(chr(0), '')
        try:
            # make sure the modified version parses as expected
            # unparse can sometimes mess things up (rarely)
            ast.parse(code_str)
            return code_str, doc_str
        except:
            return None, None
    except SyntaxError:
        return None, None


def load(json_path):
    failed_count = 0
    total_count = 0
    with open(json_path, "r") as fin:
        data = json.load(fin)
    obs = []
    for entry in tqdm.tqdm(data):
        total_count += 1
        code, nl = get_code_and_nl(entry)
        if code is None or nl is None:
            failed_count += 1
            continue
        obs.append({"code": code, "nl": nl})
    print("{}/{} failed to parse".format(failed_count, total_count))
    return obs


def get_args():
    parser = argparse.ArgumentParser(description="Github to canonical")
    parser.add_argument("-i", "--input", type=str, help="Path to input json")
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for pickled data",
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
