cd ..
# # Code experiments
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test_names conala github \
--test_paths data/conala/dataset.pkl data/github/test-dataset.pkl \
--output experiment_pipelines \
--downsample 10000 --seed 42 \
--subset code-1 code-2 code-3 code-4 code-5 code-6

# # NL experiments
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test_names conala github \
--test_paths data/conala/dataset.pkl data/github/test-dataset.pkl \
--output experiment_pipelines \
--downsample 10000 --seed 42 \
--subset nl-1 nl-2 nl-3 nl-4 nl-5 nl-6 nl-7


# Size experiments
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test_names conala github \
--test_paths data/conala/dataset.pkl data/github/test-dataset.pkl \
--output experiment_pipelines \
--downsample 10000 --seed 42 \
--subset size-1 size-2 size-3 size-4 size-5 size-6 size-7

# Full data set training
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test_names conala github \
--test_paths data/conala/dataset.pkl data/github/test-dataset.pkl \
--output experiment_pipelines \
--downsample 10000 --seed 42 \
--subset full

# Experiments for increasing amounts of training data
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test_names conala github \
--test_paths data/conala/dataset.pkl data/github/test-dataset.pkl \
--output experiment_pipelines \
--downsample 10000 --seed 42 \
--subset partial-50 partial-100 partial-250 partial-500 partial-750
