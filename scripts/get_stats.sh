cd ..
python -m src.get_result_stats compute --data experiment_pipelines/
python -m src.get_result_stats latex --data experiment_pipelines/  --output results.tex
