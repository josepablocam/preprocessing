cd ..
python -m src.experiments run \
 --data experiment_pipelines/ \
 --model all \
 --test all \
 --tune \
 --subset nl-1 nl-2 nl-3 nl-4 nl-5 nl-6 nl-7
