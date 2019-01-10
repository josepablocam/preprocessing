# Process Github function code to extract only
# API calls and the function name itself, not every token

import argparse
import ast
import json

import astunparse
import tqdm


class CollectRelevant(ast.NodeVisitor):
    def __init__(self):
        self.relevant = []

    def visit_FunctionDef(self, node):
        name = node.name.strip()
        name = name.replace("\n", " ")
        self.relevant.append(name)
        self.generic_visit(node)

    def visit_Call(self, node):
        func = astunparse.unparse(node.func).strip()
        func = func.replace("\n", " ")
        self.relevant.append(func)

    def run(self, src):
        self.relevant = []
        try:
            self.visit(ast.parse(src))
            return list(self.relevant)
        except:
            print("Failed")
            return []


def extract(input_path, output_path):
    with open(input_path, "r") as fin:
        data = json.load(fin)

    collector = CollectRelevant()
    with open(output_path, "w") as fout:
        for ix, src in enumerate(tqdm.tqdm(data)):
            calls = collector.run(src)
            line = " "
            if calls:
                line += " ".join(calls)
            fout.write(line)
            fout.write("\n")


def get_args():
    parser = argparse.ArgumentParser(description="Get calls")
    parser.add_argument(
        "-i", "--input", type=str, help="JSON for original functions"
    )
    parser.add_argument("-o", "--output", type=str, help="Txt output ")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    extract(args.input, args.output)


if __name__ == "__main__":
    main()
