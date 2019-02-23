cd ..
python -m src.get_result_stats compute --data experiment_pipelines/ --subset code-1 code-2 code-5 code-6 nl-1 nl-2 nl-5 nl-6 nl-7 --model dan2 dan3
python -m src.get_result_stats latex --data experiment_pipelines/  --output results-dannew.tex --subset code-1 code-2 code-5 code-6 nl-1 nl-2 nl-5 nl-6 nl-7
