cd ..
#python -m src.performance_over_time \
#-c new-data/test/code.npy \
#-q new-data/test/nl.npy \
#-n 1000 \
#-r 42 \
#-m models/lstm_fix_new_wass/timestamps.csv \
#-l models/lstm_fix_new_wass

python -m src.evaluate_tfidf \
-c experiment_pipelines/nl-3/test-code-github.npy \
-q experiment_pipelines/nl-3/test-nl-github.npy \
-tc experiment_pipelines/nl-3/seed-10/train-code.npy \
-m tfidf \
-o tfidf-github3.json \
#-k 1 \
#-b 0.2

python -m src.evaluate_tfidf \
-c experiment_pipelines/nl-3/test-code-github.npy \
-q experiment_pipelines/nl-3/test-nl-github.npy \
-tc experiment_pipelines/nl-3/seed-10/train-code.npy \
-m bm25 \
-o bm25-github3.json \
-k 1 \
-b 0.2

python -m src.evaluate_tfidf \
-c experiment_pipelines/nl-3/test-code-conala.npy \
-q experiment_pipelines/nl-3/test-nl-conala.npy \
-tc experiment_pipelines/nl-3/seed-10/train-code.npy \
-m tfidf \
-o tfidf-conala3.json \
#-k 1 \
#-b 0.2

python -m src.evaluate_tfidf \
-c experiment_pipelines/nl-3/test-code-conala.npy \
-q experiment_pipelines/nl-3/test-nl-conala.npy \
-tc experiment_pipelines/nl-3/seed-10/train-code.npy \
-m bm25 \
-o bm25-conala3.json \
-k 1 \
-b 0.2
