cd ..
#python -m src.performance_over_time \
#-c new-data/test/code.npy \
#-q new-data/test/nl.npy \
#-n 1000 \
#-r 42 \
#-m models/lstm_fix_new_wass/timestamps.csv \
#-l models/lstm_fix_new_wass

python -m src.evaluate_tfidf \
-c new-data/test/code.npy \
-q new-data/test/nl.npy \
-m tfidf \
#-k 1 \
#-b 0.2
