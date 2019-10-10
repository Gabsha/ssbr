# SSBR
Implementation of self-supervised body part regression.

Reference: 
Unsupervised body part regression using convolutional neural network

(Ke Yan, Le Lu, Ronald M. Summers)

[https://arxiv.org/pdf/1707.03891.pdf]


## Setup

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

The ircad dataset (https://www.ircad.fr/research/3dircadb/) is used to showcase this model.

```bash
python ssbr/main.py train \
    -c ./ssbr/configs/train.json \
    -d ircad \
    -o experiments/exp3
```

## Notebooks

See the experiment [notebook](notebooks/experiment.ipynb) for data and results exploration

## Run tests

Althought coverage is low, some critical functions are covered by unit tests.

```
pytest
```
