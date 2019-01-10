#!/bin/bash
BASE_DIR=new-data/

# Some basic parameters
# downsample large corpora to this many examples
DOWNSAMPLE_N=10000
# vocabulary size: top K words based on frequency
TOP_K=67659
# target length of a padded code bag
CODE_LEN=58
# target lenght of a padded natural language bag
NL_LEN=30
# embeddings dimension (needed to compute initialization using fasttext)
EMBEDDING_DIM=300

# Base dataset directories
GITHUB_DIR="${BASE_DIR}/github"
CONALA_DIR="${BASE_DIR}/conala/"
TRAIN_DIR="${BASE_DIR}/train/"
VALIDATION_DIR="${BASE_DIR}/validation/"
TEST_DIR="${BASE_DIR}/test/"
ALT_DIR="${BASE_DIR}/alternative/"
QRA_DIR="${BASE_DIR}/qra"


######################################
########### DOWNLOAD DATASETS ########
######################################

######################################
########### GITHUB DATASET ###########
######################################

# starting from core datasets
GITHUB_URLS="https://storage.googleapis.com/kubeflow-examples/code_search/data/test.docstring
https://storage.googleapis.com/kubeflow-examples/code_search/data/test.function
https://storage.googleapis.com/kubeflow-examples/code_search/data/test.lineage
https://storage.googleapis.com/kubeflow-examples/code_search/data/test_original_function.json.gz
https://storage.googleapis.com/kubeflow-examples/code_search/data/train.docstring
https://storage.googleapis.com/kubeflow-examples/code_search/data/train.function
https://storage.googleapis.com/kubeflow-examples/code_search/data/train.lineage
https://storage.googleapis.com/kubeflow-examples/code_search/data/train_original_function.json.gz
https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.docstring
https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.function
https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.lineage
https://storage.googleapis.com/kubeflow-examples/code_search/data/valid_original_function.json.gz
https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings.function
https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings.lineage
https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings_original_function.json.gz
"

mkdir -p ${GITHUB_DIR}

if [[ $1 != "--no-download" ]]
then
    echo "Download GITHUB data"
    for url in ${GITHUB_URLS}
    do
      wget ${url} --directory-prefix=${GITHUB_DIR}
    done

    echo "Unzipping GITHUB data"
    pushd ${GITHUB_DIR}
    gunzip *.json.gz
    popd
fi


######################################
########### CONALA DATASET ###########
######################################
CONALA_URL="http://www.phontron.com/download/conala-corpus-v1.1.zip"

mkdir -p ${CONALA_DIR}

if [[ $1 != "--no-download" ]]
then
    echo "Downloading CONALA data"
    wget ${CONALA_URL} --directory-prefix=${CONALA_DIR}
    pushd ${CONALA_DIR}
    echo "Unzipping CONALA data"
    # -j to avoid conala-corpus directory
    unzip -j *.zip
    popd
fi

######################################
####### Preparing datasets ###########
######################################

# extract only API calls from functions
echo "Extracting relevant tokens from github functions"
python -m src.function_json_to_tokens \
  --input "${GITHUB_DIR}/train_original_function.json" \
  --output "${GITHUB_DIR}/train.function_relevant" || exit 1

# Remove entries that are empty
echo "Removing potentially empty lines from training data"
python -m src.remove_empty \
  --input "${GITHUB_DIR}/train.function_relevant" "${GITHUB_DIR}/train.docstring" \
  --output "${GITHUB_DIR}/train.function_relevant_complete" "${GITHUB_DIR}/train.docstring_complete" || exit 1

# overwrite the original which may have emtpy lines
mv -f "${GITHUB_DIR}/train.function_relevant_complete" "${GITHUB_DIR}/train.function_relevant"
mv -f "${GITHUB_DIR}/train.docstring_complete" "${GITHUB_DIR}/train.docstring"


# Extract tokens from the .jsonl and .json files
echo "Extracting tokens from CONALA json files"
python -m src.conala_to_tokens \
-j ${CONALA_DIR}/conala-mined.jsonl \
-c ${CONALA_DIR}/conala-mined-code.txt \
-q ${CONALA_DIR}/conala-mined-queries.txt || exit 1

python -m src.conala_to_tokens \
-j ${CONALA_DIR}/conala-train.json \
-c ${CONALA_DIR}/conala-train-code.txt \
-q ${CONALA_DIR}/conala-train-queries.txt || exit 1

python -m src.conala_to_tokens \
-j ${CONALA_DIR}/conala-test.json \
-c ${CONALA_DIR}/conala-test-code.txt \
-q ${CONALA_DIR}/conala-test-queries.txt || exit 1

echo "Removing potentially empty lines from conala data"
python -m src.remove_empty \
  --input "${CONALA_DIR}/conala-mined-code.txt" "${CONALA_DIR}/conala-mined-queries.txt" \
  --output "${CONALA_DIR}/conala-mined-code-complete.txt" "${CONALA_DIR}/conala-mined-queries-complete.txt" || exit 1

python -m src.remove_empty \
  --input "${CONALA_DIR}/conala-train-code.txt" "${CONALA_DIR}/conala-train-queries.txt" \
  --output "${CONALA_DIR}/conala-train-code-complete.txt" "${CONALA_DIR}/conala-train-queries-complete.txt" || exit 1

python -m src.remove_empty \
  --input "${CONALA_DIR}/conala-test-code.txt" "${CONALA_DIR}/conala-test-queries.txt" \
  --output "${CONALA_DIR}/conala-test-code-complete.txt" "${CONALA_DIR}/conala-test-queries-complete.txt" || exit 1


######################################
########### Main Corpus ##############
######################################
# Compute initial embeddings based on large github code data
# compute it by concatenating code and docstring
# and tokenized to split up things like camelcase before computing embeddings
echo "Computing embeddings from GITHUB data"
cat "${GITHUB_DIR}/train.function_relevant" > /tmp/embeddings_input
cat "${GITHUB_DIR}/train.docstring" >> /tmp/embeddings_input

# Note different embedding dimension relative to QRA code
python -m src.precomputed_embeddings \
  -i /tmp/embeddings_input \
  -o "${GITHUB_DIR}/github_embeddings" \
  -d ${EMBEDDING_DIM} || exit 1


mkdir -p ${TRAIN_DIR}
cp "${GITHUB_DIR}/github_embeddings.vec" "${TRAIN_DIR}/embeddings.vec"


echo "Producing vocabularies directly from embeddings"
python -m src.vocabulary \
  load-from-embeddings \
  -i "${GITHUB_DIR}/github_embeddings.vec" \
  -o "${TRAIN_DIR}/code_vocab_encoder.pkl" || exit 1

# use the same vocab for both
cp -r "${TRAIN_DIR}/code_vocab_encoder.pkl" "${TRAIN_DIR}/nl_vocab_encoder.pkl"

# echo "Computing code/NL vocabularies from GITHUB data"
# python -m src.vocabulary \
#  compute \
#  -i "${GITHUB_DIR}/train.function_relevant" \
#  -k ${TOP_K} \
#  -o "${TRAIN_DIR}/code_vocab_encoder.pkl" || exit 1
#
#
# cat "${GITHUB_DIR}/train.docstring" > /tmp/nl.txt
# cat "${CONALA_DIR}/conala-mined-queries.txt" >> /tmp/nl.txt
# python -m src.vocabulary \
#   compute \
#   -i /tmp/nl.txt \
#   -k ${TOP_K} \
#   -o "${TRAIN_DIR}/nl_vocab_encoder.pkl" || exit 1


# Downsample github training data to 10k to produce dataset
# that we can train on in reasonable amounts of time for experiments
# sampled with replacement
echo "Assembling training data from github downsampled corpus"
mkdir -p "${GITHUB_DIR}/downsampled/"
python -m src.downsample \
  --files "${GITHUB_DIR}/train.function_relevant" "${GITHUB_DIR}/train.docstring" \
  --target ${DOWNSAMPLE_N} \
  --replace \
  --seed 42 \
  --output_dir "${GITHUB_DIR}/downsampled" || exit 1

# Training data
python -m src.vocabulary \
  apply \
  -i "${GITHUB_DIR}/downsampled/train.function_relevant" \
  -v "${TRAIN_DIR}/code_vocab_encoder.pkl" \
  -l ${CODE_LEN} \
  -o "${TRAIN_DIR}/code.npy" \
  -f numpy || exit 1

python -m src.vocabulary \
  apply \
  -i "${GITHUB_DIR}/downsampled/train.docstring" \
  -v "${TRAIN_DIR}/nl_vocab_encoder.pkl" \
  -l ${NL_LEN} \
  -o "${TRAIN_DIR}/nl.npy" \
  -f numpy || exit 1

python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-mined-queries-complete.txt" \
  -v "${TRAIN_DIR}/nl_vocab_encoder.pkl" \
  -l ${NL_LEN} \
  -o "${TRAIN_DIR}/target_domain_nl.npy" \
  -f numpy || exit 1

python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-mined-code-complete.txt" \
  -v "${TRAIN_DIR}/code_vocab_encoder.pkl" \
  -l ${CODE_LEN} \
  -o "${TRAIN_DIR}/target_domain_code.npy" \
  -f numpy || exit 1

# Validation data
echo "Assembling validation data from conala train corpus"
mkdir -p "${VALIDATION_DIR}"
python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-train-code-complete.txt" \
  -v "${TRAIN_DIR}/code_vocab_encoder.pkl" \
  -l ${CODE_LEN} \
  -o "${VALIDATION_DIR}/code.npy" \
  -f numpy || exit 1

python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-train-queries-complete.txt" \
  -v "${TRAIN_DIR}/nl_vocab_encoder.pkl" \
  -l ${NL_LEN} \
  -o "${VALIDATION_DIR}/nl.npy" \
  -f numpy || exit 1

# Test data
echo "Assembling test data"
mkdir -p ${TEST_DIR}
python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-test-code-complete.txt" \
  -v "${TRAIN_DIR}/code_vocab_encoder.pkl" \
  -l ${CODE_LEN} \
  -o "${TEST_DIR}/code.npy" \
  -f numpy || exit 1

python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-test-queries-complete.txt" \
  -v "${TRAIN_DIR}/nl_vocab_encoder.pkl" \
  -l ${NL_LEN} \
  -o "${TEST_DIR}/nl.npy" \
  -f numpy || exit 1


######################################
######### Alternative Corpus #########
######################################
# Alternative corpus based on target-domain data
echo "Assembling alternative training corpus using CONALA mined corpus"
mkdir -p ${ALT_DIR}
mkdir -p "${ALT_DIR}/train"
# compute vocabularies based on CONALA mined corpus
python -m src.vocabulary \
 compute \
 -i "${CONALA_DIR}/conala-mined-code-complete.txt" \
 -k ${TOP_K} \
 -o "${ALT_DIR}/train/code_vocab_encoder.pkl" || exit 1

python -m src.vocabulary \
  compute \
  -i "${CONALA_DIR}/conala-mined-queries-complete.txt" \
  -k ${TOP_K} \
  -o "${ALT_DIR}/train/nl_vocab_encoder.pkl" || exit 1

# compute embeddings based on conala-mined-queries and code
cat "${CONALA_DIR}/conala-mined-code-complete.txt" > /tmp/embeddings_input
cat "${CONALA_DIR}/conala-mined-queries-complete.txt" >> /tmp/embeddings_input

# Note different embedding dimension relative to QRA code
python -m src.precomputed_embeddings \
  -i /tmp/embeddings_input \
  -o "${CONALA_DIR}/conala_embeddings" \
  -d ${EMBEDDING_DIM} || exit 1

cp "${CONALA_DIR}/conala_embeddings.vec" "${ALT_DIR}/train/embeddings.vec"

# Training data
python -m src.downsample \
  --files "${CONALA_DIR}/conala-mined-code-complete.txt" "${CONALA_DIR}/conala-mined-queries-complete.txt" \
  --target ${DOWNSAMPLE_N} \
  --replace \
  --seed 42 \
  --output_dir "${CONALA_DIR}/conala-mined-downsampled/" || exit 1

python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-mined-downsampled/conala-mined-code-complete.txt" \
  -v "${ALT_DIR}/train/code_vocab_encoder.pkl" \
  -l ${CODE_LEN} \
  -o "${ALT_DIR}/train/code.npy" \
  -f numpy || exit 1

python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-mined-downsampled/conala-mined-queries-complete.txt" \
  -v "${ALT_DIR}/train/nl_vocab_encoder.pkl" \
  -l ${NL_LEN} \
  -o "${ALT_DIR}/train/nl.npy" \
  -f numpy || exit 1


# Validation data
mkdir -p "${ALT_DIR}/validation/"
python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-train-code-complete.txt" \
  -v "${ALT_DIR}/train/code_vocab_encoder.pkl" \
  -l ${CODE_LEN} \
  -o "${ALT_DIR}/validation/code.npy" \
  -f numpy || exit 1

python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-train-queries-complete.txt" \
  -v "${ALT_DIR}/train/nl_vocab_encoder.pkl" \
  -l ${NL_LEN} \
  -o "${ALT_DIR}/validation/nl.npy" \
  -f numpy || exit 1

# Test data
mkdir -p "${ALT_DIR}/test/"
python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-test-code-complete.txt" \
  -v "${ALT_DIR}/train/code_vocab_encoder.pkl" \
  -l ${CODE_LEN} \
  -o "${ALT_DIR}/test/code.npy" \
  -f numpy || exit 1

python -m src.vocabulary \
  apply \
  -i "${CONALA_DIR}/conala-test-queries-complete.txt" \
  -v "${ALT_DIR}/train/nl_vocab_encoder.pkl" \
  -l ${NL_LEN} \
  -o "${ALT_DIR}/test/nl.npy" \
  -f numpy || exit 1


######################################
######### QRA Corpus #################
######################################

# Create data to run on QRA (Darsh) code
# Note that this doesn't create the same training set as above
# it resamples from the entire github (seed defined in script below)
mkdir -p ${QRA_DIR}
# export so script below has this defined
export QRA_DIR=${QRA_DIR}
export GITHUB_DIR=${GITHUB_DIR}
export CONALA_DIR=${CONALA_DIR}
bash create_qra_data/prepare_data.sh
