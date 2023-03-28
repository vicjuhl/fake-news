python src/main.py -p reduce shorten get_dups -v 2 -n 10000000
python src/main.py -p reduce shorten split get_dups -v 2 -n 10000000
python src/main.py -p json stem_json summarize -v 2 -q 0.5 0.05 -n 10000000
python src/model_pipeline.py -v 2 -t1 3 4 5 6 7 -t2 8 9 10 -md linear pa simple multi_nb compl_nb svm random_forest meta -mt train dump_model infer4_mm_training infer evaluate -nt 10000000 -nv 10000000