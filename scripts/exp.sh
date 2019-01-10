cd ..
#python3 -m src.train basic --code_path data/train/github_code.npy --docstrings_path data/train/github_docstrings.npy --queries_path data/train/so_queries.npy --print_every 10 --save_every 100 --use_precomputed_embeddings --use_adversarial
: '
python3 -m src.train downsampled-wass \
--code_path data/new-data/train/code.npy \
--docstrings_path data/new-data/train/nl.npy \
--embeddings_path data/new-data/train/embeddings.vec \
--code_vocab_encoder_path data/new-data/train/code_vocab_encoder.pkl \
--nl_vocab_encoder_path  data/new-data/train/nl_vocab_encoder.pkl \
--print_every 1000 \
--save_every 1 \
--output_folder average_new

python3 -m src.train downsampled-wass \
--code_path data/new-data/train/code.npy \
--docstrings_path data/new-data/train/nl.npy \
--embeddings_path data/new-data/train/embeddings.vec \
--code_vocab_encoder_path data/new-data/train/code_vocab_encoder.pkl \
--nl_vocab_encoder_path  data/new-data/train/nl_vocab_encoder.pkl \
--print_every 1000 \
--save_every 1 \
--queries_path data/new-data/train/target_domain_nl.pkl.npy \
--output_folder average_new_wass

python3 -m src.train downsampled-wass \
--code_path data/new-data/alternative/train/code.npy \
--docstrings_path data/new-data/alternative/train/nl.npy \
--embeddings_path data/new-data/alternative/train/embeddings.vec \
--code_vocab_encoder_path data/new-data/alternative/train/code_vocab_encoder.pkl \
--nl_vocab_encoder_path  data/new-data/alternative/train/nl_vocab_encoder.pkl \
--print_every 1000 \
--save_every 1 \
--output_folder average_new_in_domain
'


python3 -m src.train lstm-fix \
--code_path new-data/train/code.npy \
--docstrings_path new-data/train/nl.npy \
--embeddings_path new-data/train/embeddings.vec \
--code_vocab_encoder_path new-data/train/code_vocab_encoder.pkl \
--nl_vocab_encoder_path  new-data/train/nl_vocab_encoder.pkl \
--print_every 1000 \
--save_every 1 \
--queries_path new-data/train/target_domain_nl.pkl.npy \
--output_folder lstm_fix_new_wass
