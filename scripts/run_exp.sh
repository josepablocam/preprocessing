cd ..
## Fixed embeddings
# Code experiments
python -m src.experiments run \
 --data experiment_pipelines/ \
 --model all \
 --test all \
 --subset code-1 code-2 code-3 code-4 code-5 code-6

# NL experiments
 python -m src.experiments run \
  --data experiment_pipelines/ \
  --model all \
  --test all \
  --subset nl-1 nl-2 nl-3 nl-4 nl-5 nl-6 nl-7


# Size experiments
python -m src.experiments run \
 --data experiment_pipelines/ \
 --model all \
 --test all \
 --subset size-1 size-2 size-3 size-4 size-5 size-6 size-7

# Full dataset
python -m src.experiments run \
 --data experiment_pipelines/ \
 --model all \
 --test all \
 --subset full

 # Experiments for increasing amounts of training data
 python -m src.experiments run \
 --data experiment_pipelines/ \
 --model all \
 --test all \
 --subset partial-50 partial-100 partial-250 partial-500 partial-750

 ## Tuned embeddings
 # Add --tune to each of the calls above (note that it overwrites model folders)
