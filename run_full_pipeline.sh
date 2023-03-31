# # Preprocessing
python src/main.py -p reduce shorten get_dups -v 2 -n 100000000
python src/main.py -p reduce shorten split get_dups -v 2 -n 100000000
python src/main.py -p json stem_json summarize -v 2 -q 0.5 0.05 -n 100000000

# Validation and preprocessing
python src/model_pipeline.py -v 1 -t1 3 4 5 6 7 -t2 8 9 10 -p prep_val -nt 100000000 -nv 100000000 -t 1
python src/model_pipeline.py -v 2 -t1 3 4 5 6 7 -t2 8 9 10 -p prep_val -nt 100000000 -nv 100000000

# Training
python src/model_pipeline.py -v 2 -t1 3 4 5 6 7 -t2 8 9 10 -md linear pa simple multi_nb compl_nb svm random_forest meta -mt train dump_model infer4_mm_training -nt 100000000 -nv 100000000

# Test on test and liar sets
python src/model_pipeline.py -v 2 -t1 3 4 5 6 7 -t2 8 9 10 -md linear pa simple multi_nb compl_nb svm random_forest meta -mt infer evaluate -nt 100000000 -nv 100000000 # val set
python src/model_pipeline.py -v 2 -t1 3 4 5 6 7 -t2 8 9 10 -md linear pa simple multi_nb compl_nb svm random_forest meta -mt infer evaluate -nt 100000000 -nv 100000000 -t 1 # test set
python src/model_pipeline.py -v 2 -t1 3 4 5 6 7 -t2 8 9 10 -md linear pa simple multi_nb compl_nb svm random_forest meta -mt infer evaluate -nt 100000000 -nv 100000000 -l 1 # liar set
