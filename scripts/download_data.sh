#!/usr/bin/env bash

BASE_DIR=data/

# Base dataset directories
GITHUB_DIR="${BASE_DIR}/github"
CONALA_DIR="${BASE_DIR}/conala/"


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



#################################################
########### Create canonical data sets ###########
#################################################

python -m src.conala \
  --input "${CONALA_DIR}/conala-test.json" \
  --output "${CONALA_DIR}/dataset.pkl"

python -m src.github \
  --code "${GITHUB_DIR}/train_original_function.json" \
  --nl "${GITHUB_DIR}/train.docstring" \
  --output "${GITHUB_DIR}/dataset.pkl"
