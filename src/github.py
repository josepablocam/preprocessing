# Convert the Github files into canonical format
import argparse
import ast
import astunparse
import json
import tqdm
import pickle

from .preprocess import CanonicalInput, take_first_sentence

END_OF_MANUAL_MARKER = "END_OF_MANUAL"
MANUAL_TEST_N = 500

PASS = ast.parse("pass").body[0]


def get_code_and_nl(src, no_doc_ok=False):
    try:
        tree = ast.parse(src)
        func = tree.body[0]
        doc = ast.get_docstring(func)
        # strip the docstring from the code body if there
        if isinstance(func.body[0], ast.Expr):
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
        if doc is None and no_doc_ok:
            return code_str, doc
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


def load_manual_test(json_path):
    with open(json_path, "r") as fin:
        data = json.load(fin)

    manually_curated = []
    found_marker = False
    for entry in data:
        if END_OF_MANUAL_MARKER in entry:
            found_marker = True
            break
        code = entry["code"]
        # make sure we strip out docstring from the code
        code, _ = get_code_and_nl(code, no_doc_ok=True)
        nl = entry["nl"]
        # take first sentence based on period
        nl = take_first_sentence(nl)
        manually_curated.append({"code": code, "nl": nl})

    if not found_marker:
        raise Exception(
            "This file may not be manually curated, missing marker: {}".
            format(END_OF_MANUAL_MARKER)
        )

    manually_curated = manually_curated[:MANUAL_TEST_N]
    return manually_curated


def get_args():
    parser = argparse.ArgumentParser(description="Github to canonical")
    parser.add_argument("-i", "--input", type=str, help="Path to input json")
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for pickled data",
    )
    parser.add_argument(
        "--manual_test",
        action="store_true",
        help="Treat file as manually curated sample of Github examples",
    )
    return parser.parse_args()


def main():
    args = get_args()
    if args.manual_test:
        data = load_manual_test(args.input)
    else:
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
