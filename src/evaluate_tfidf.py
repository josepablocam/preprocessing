import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .evaluation import location_of_correct, get_mrr, get_fraction_correct_at
import json


def load_evaluation_data(code_path, queries_path, train_code_path):
    code = np.load(code_path)
    queries = np.load(queries_path)
    train_code = np.load(train_code_path)
    return code, queries, train_code


def eval_traditional(queries, code, train_code, args):
    '''
    queries and code are both np matrix (n,max_len_query), (n,max_len_code)
    '''
    # Get vocab size, padding idx
    vocab_size = max(np.max(queries), np.max(code), np.max(train_code)) + 1
    padding_idx = 1
    n = code.shape[0]
    m = train_code.shape[0]

    count_matrix_code = np.zeros((n, vocab_size))
    count_matrix_queries = np.zeros((n, vocab_size))
    count_matrix_train_code = np.zeros((m, vocab_size))
    for i in range(n):
        for idx in code[i]:
            if idx != padding_idx:
                count_matrix_code[i][idx] += 1
        for idx in queries[i]:
            if idx != padding_idx:
                count_matrix_queries[i][idx] += 1
    for i in range(m):
        for idx in train_code[i]:
            if idx != padding_idx:
                count_matrix_train_code[i][idx] += 1

    # Compute idf using train code
    idf = np.count_nonzero(count_matrix_train_code, 0)
    idf = np.log(np.true_divide(m - idf + 0.5, idf + 0.5))
    #idf = np.clip(idf,0,None)
    #idf[np.nonzero(idf)] = np.log(np.true_divide(n, idf[np.nonzero(idf)])) + 1

    # Compute tf
    tf_code = np.true_divide(
        count_matrix_code, count_matrix_code.sum(1, keepdims=True)
    )
    tf_queries = np.true_divide(
        count_matrix_queries, count_matrix_queries.sum(1, keepdims=True)
    )

    if args.method == 'tfidf':
        # Compute tfidf for each entry
        tfidf_code = tf_code * np.expand_dims(idf, axis=0)
        tfidf_queries = tf_queries * np.expand_dims(idf, axis=0)
        # compute similarity matrix and from that the position of correct answer
        sim_mat = cosine_similarity(tfidf_queries, tfidf_code)
    elif args.method == 'bm25':
        query_vecs = count_matrix_queries * np.expand_dims(idf, axis=0)
        code_lens = count_matrix_code.sum(1, keepdims=True)
        code_vecs = np.true_divide(
            count_matrix_code * (args.k + 1), count_matrix_code +
            args.k * (1 - args.b + args.b * code_lens / np.mean(code_lens))
        )
        # Compute score matrix
        sim_mat = np.matmul(query_vecs, np.transpose(code_vecs))
    else:
        raise KeyError("Method invalid")

    import pdb
    # Evaluate performance
    ans_locs = location_of_correct(sim_mat)

    summary = {}
    mr = np.mean(ans_locs)
    mrr = get_mrr(ans_locs)
    summary["mrr"] = mrr

    cutoffs = [1,5,10]
    fracs = []

    for c in cutoffs:
        frac = get_fraction_correct_at(ans_locs, c)
        fracs.append(frac)
    print("Num obs: {}".format(code.shape[0]))
    print("Mean Rank: {}".format(mr))
    print("MRR: {}".format(mrr))

    for c, f in zip(cutoffs, fracs):
        print("Fraction Correct@{}: {}".format(c, f))
        summary["success@{}".format(c)] = f
    return summary


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate using keyword-based retrieve methods"
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        help="Retrieve method, e.g. tfidf, bm25",
    )
    parser.add_argument(
        "-c",
        "--code_path",
        type=str,
        help="Path to saved code dataset (must be aligned with queries)",
    )
    parser.add_argument(
        "-q",
        "--queries_path",
        type=str,
        help="Path to saved queries dataset (must be aligned with code)",
    )
    parser.add_argument(
        "-tc",
        "--train_code_path",
        type=str,
        help="Path to training code corpus, use to compute idf",
    )
    parser.add_argument(
        "-k",
        "--k",
        type=float,
        default=2.0,
        help="Parameter for bm25",
    )
    parser.add_argument(
        "-b",
        "--b",
        type=float,
        default=0.75,
        help="Parameter for bm25",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        type=str,
        help="Output file name",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    code, queries, train_code = load_evaluation_data(
        args.code_path,
        args.queries_path,
        args.train_code_path,
    )

    summary = eval_traditional(queries, code, train_code, args)
    with open(args.out_file, 'w') as fout:
        json.dump(summary, fout)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
