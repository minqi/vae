# Variational Auto-Encoders

A basic VAE implementation to reproduce the results in [Kingma and Welling, 2014](https://arxiv.org/abs/1312.6114).

### Train on MNIST
```
conda env create -f environment.yml
conda activate torch
python vae.py
```

### Command-line args
|Arg|Value|
|-|-|
|seed|random seed|
|batch_size|batch size|
|learning_rate|learning rate|
|n_epochs|# training epochs|
|no_cuda|true means don't use cuda|
|hidden_size|dim of encoder/decoder hidden state|
|latent_size|dim of latent encoding|
|test_output|how test samples after each epoch are generated|
|test_output_size|square dimension of test sample plots|

### Examples
Save decodings of 20x20 uniformly spaced latent codes in the latent space after each epoch as a .png.  
`python vae.py --n_epochs 5 --latent_size 2 --test_output uniform --tn 20`

Save reconstructions of 20x20 random test samples after each epoch as a .png.  
`python vae.py --test_output random --tn 20`