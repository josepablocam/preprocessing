cd ..
python -m src.experiments run \
 --data experiment_pipelines/ \
 --model dannew \
 --test all \
 --subset code-1 code-2 code-5 code-6 nl-1 nl-2 nl-5 nl-6 nl-7
