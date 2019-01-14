
## General
Everything is run from the preprocess/ root folder.
You should use [conda](https://conda.io/docs/) to setup your environment.
 Assuming you have conda installed, you can then setup your environment with:

```
conda env create --name <env> --file environment.yml
```

You can then activate this environment by running

```
source activate <env>
```

# Download data

```
bash scripts/download_data.sh
```

downloads necessary data and creates a canonical representation for a corpus
(see `CanonicalInput` in src/preprocess.py for details).

# Build fastttext

```
cd fastText-0.1.0 && make clean && make
```

builds the necessary fasttext library for creating the initial embeddings.

# Generate experiment folders and run

```
python -m src.experiments generate \
--train data/github/dataset.pkl  \
--test data/conala/dataset.pkl \
--output experiment_pipelines \
--downsample 10000 --seed 42
```

Will produce a folder for each experiment configuration. You can add
more experiments by adding them to the source in
`generate_experiments` in src/experiments.py.
Each folder is self-contained.

You can then run the experiments by pointing to the root folder of the
experiments.

```
python -m src.experiments run --data experiment_pipelines/
```

will train a biLSTM on the data in each of the folders and will evaluate
these on the test data and write results to the corresponding folder
in a file called results.json.


## Visualize training

```
tensorboard --logdir="./runs"
```

and then you can navigate to a browser and see at `localhost:6006`
(or whatever port you specify, if you change)

## Running on GPU
I would recommend using AWS P2 machines for GPU access (unless you have something else easily available).

If you have an AWS account and have already setup all the AWS command line tools on your machine, the easiest way to setup a machine that runs this code is simply to copy the setup in  https://github.com/fastai/courses/blob/master/setup/setup_p2.sh

You can execute this and get a configured machine. Then I would setup the conda environment as done here (you may need to install conda), and copy the necessary data/code on to the machine.

On the machine, you would also need to run

```
sudo apt-get install cuda-drivers
```

and reboot the machine. This would set up the cuda drivers correctly.

## Data and Result Storage
We store the data and trained models on Dropbox. A convenient tool for uploading/downloading files to/from Dropbox to the remote machine (e.g. AWS) can be found in https://github.com/andreafabrizi/Dropbox-Uploader

If you want to use this, you will need an access token for Dropbox API. Please contact David for this.
