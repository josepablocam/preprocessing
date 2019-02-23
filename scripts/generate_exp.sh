cd ..
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test_names conala github \
--test_paths data/conala/dataset.pkl data/github/test-dataset.pkl \
--output experiment_pipelines \
--subset code-1 code-2 code-5 code-6
