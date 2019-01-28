cd ..
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test_names conala github \
--test_paths data/conala/dataset.pkl data/github/test-dataset.pkl \
--output experiment_pipelines \
--subset nl-1 nl-2 nl-3 nl-4 nl-5 \
--downsample 10000 --seed 42
