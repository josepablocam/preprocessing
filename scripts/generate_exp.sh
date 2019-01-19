cd ..
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test data/conala/dataset.pkl \
--output experiment_pipelines \
--downsample 10000 --seed 42
