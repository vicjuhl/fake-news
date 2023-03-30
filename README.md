### Fake news detection project for the course Data Science on DIKU, University of Copenhagen
# Contributors: Hjalti Petursson Poulsen, Frederik van Wylich-Muxoll, Hans Peter Lyngsøe Raaschou-Jensen, Victor Kaplan Kjellerup

This repository contains the code base used to reproduce the reported results from the Fake News Detection project by the authors.

## The results as they appear in the report were created, using the following specifications:
# Hardware:
Apple 2023 MacBook Pro M2 Pro, 16GB RAM, 10-core CPU, 16-core GPU, 512 GB SSD, Serial Number NLWWKY70G3
# Software:
You need a functioning installation of miniconda to install the virtual environment. The Miniconda3 macOS Apple M1 64-bit bash version should be download here: https://docs.conda.io/en/latest/miniconda.html. Use version number 23.1.0.

With conda installed, please do the following to intall environment:
    To generate new conda env from env file (default name fake-news):
    conda env create -f environment.yml

    To update existing conda env from env file:
    conda env update -f environment.yml --prune

# Preparing the corpus data
you need to first download the fake news corpus from https://github.com/several27/FakeNewsCorpus/releases/tag/v1.0. From the requirements to the fake news project: "You will need to use a multi-part decompression tool, e.g. 7z. Given all the files, execute the following command: 7z x news.csv.zip. This should create a 27GB file on disk (29.322.513.705 bytes)."

After downloading and decompressing the corpus, create the folders fake-news/data_files/corpus/ and place the corpus file her.Do not rename the file.

To execute the code, go to root of repository and run:
bash run_full_pipeline.sh

As you will see in the bash file, this runs a sequence of python scripts, each responsible for one part of the pipeline and you will see various files and folders being created in the folders fake-news/data_files/ and fake-news/model_files/ TODO.

Firstly, the preprocessing is done in three runs, including duplicate removal, tokenization, stemming, vocabulary reduction splitting and reduction of corpus into more managably sized csv-files for later fitting and inference. Some of these files contain the {val_set_num} tag in their name. This is a number given to at runtime, 2 is used for the project. The files created in this process are:

fake-news/corpus/duplicates.csv :
    duplicate id's
fake-news/corpus/reduced_corpus.csv :
    corpus without duplicates and excluded type labels
fake-news/corpus/splits.csv :
    split numbers for id's, used for test, validation and test splitting
fake-news/processed_csv/shortened_corpus.csv :
    corpus with only 600 character articles; used for finding duplicates
fake-news/processed_csv/summarized_corpus_valset{val_set_num}.csv :
    summarized corpus, containing a bag of words dictionary for each article alongside a few other features.
fake-news/words/included_words_valset{val_set_num}.json :
    vocabulary of training data, after stemming and before vocabulary reduction
fake-news/words/stop_words_removed{val_set_num}.json :
    vocabulary of training set, after removed stopwords

To custom run the preprocessing (without bash script), see help messages for the arguments in fake-news/src/main.py

# Modelling
The remainder of the run_full_pipeline.sh concerns model training, validation and testing.

Firstly both the validation and test sets are prepared for later inference. Then the models are trained (this itself takes around 3 hours on specified hardware). The trained models are dumped to files for later loading for inference.

Custom split numbers can be chosen for validation set and the two training set: some for training the hidden layer models and some for training the meta estimator model. Please specify all splits as such: -v 2 -t1 3 4 5 6 7 -t2 8 9 10, or whatever you prefer. The validation set number should be the same as was used for preprocessing, but the training set numbers are optional among what remains from 2 to 10 without (1 being reserved for the test split).

Inference and evaluation can be run on either the validation set or the test or LIAR sets without training the models again (a liar set, preprocessed by us is available in the root of the data_files folder). Note that the validation set argument still needs to be passed as e.g. -v 2, even when running on the test sets. To run with liar, use -l 1. To run with test split 1, use -t 1.

The results of the inference and evaluation is dumped to fake-news/model_files/ TODO.

Please consult help messages of fake-news/model_pipeline.py for instructions on custom runs.

# Notebooks
All jupyter notebooks in the repository can be run for high-level analysis. Preprocessing and sometimes modelling is required before running the notebooks.

# Run logs
Manual run logs from our project have been saved to fake-news/logs/fake-news-data-log-full.txt and fake-news/logs/fake-news-data-log-short.txt, which includes the prints from our own pipeline production (verbose and compact, respectively)
