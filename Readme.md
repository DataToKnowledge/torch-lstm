Torch LSTM for Char, Number and Sequence of Numbers
===================================================

This code implements a **multi-layer Recurrent Neural Network (RNN, LSTM and GRU)** for training, generating and predicting character/number-level models. The algorithms learn to predict the probability of the next character/number in a sequence. In other words, it can be used to:

1.	train generative language models starting from characters
2.	train a generative time-series models starting from single observations

This code was originally based on Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6) and from [char-rnn](https://github.com/karpathy/char-rnn).

Requirements
------------

This code is written in Lua and requires [Torch 7](http://torch.ch/). To install Torch please follow the [Getting started with Torch](http://torch.ch/docs/getting-started.html#_).

After tath, to train a model you need to install the `nngraph` and `optim` packges using[LuaRocks](https://luarocks.org/).

```bash
$ luarocks install nngraph
$ luarocks install optim
```

If you like to use CUDA GPU computing platform, you'll need to install the [CUDA Toolkit]((https://developer.nvidia.com/cuda-toolkit), and then `cutorch` and `cunn` packages:

```bash
$ luarocks install cutorch
$ luarocks install cunn
```

If you'd like to use OpenCL GPU computing, you'll first need to install the `cltorch` and `clnn` packages, and then use the option `-opencl 1` during training ([cltorch issues](https://github.com/hughperkins/cltorch/issues)\):

```bash
$ luarocks install cltorch
$ luarocks install clnn
```

If you want to debug the code you can use the [zbs-torch IDE ](zbs-torch).

Usage
-----

### Data

All input data is stored inside the `data/` directory. There are included two datasets:

1.	`data/tinyshakespeare`, and
2.	`data/timeseries`

Inside the directories is assumed that there is an `input.txt` file that contains all the input data. You can add your own data by creating a new folder inside data with a `input.txt` file. Text files are split character by character, while numerical files are split by line.

### Training

To train a model uses the `train.lua`.

```bash
$ th train.lua -data_dir data/some_folder -gpuid -1
```

The `-data_dir` flag is most important since it specifies the dataset to use. Notice that in this example we're also setting `gpuid` to -1 which tells the code to train using CPU, otherwise it defaults to GPU 0. There are many other flags for various options. Consult `$ th train.lua -help` for comprehensive settings. Here's another example:

```
$ th train.lua -data_dir data/some_folder -rnn_size 512 -num_layers 2 -dropout 0.5
```

When the model is training the checkpoint are written to the `data/<dataset>/cp` folder. The frequency with which these checkpoints are written is controlled with number of iterations, as specified with the `eval_val_every` option (e.g. if this is 1 then a checkpoint is written every iteration). The filename of these checkpoints contains a very important number: the **loss**.

> For example, a checkpoint with filename `lm_lstm_epoch0.95_2.0681.t7` indicates that at this point the model was on epoch 0.95 (i.e. it has almost done one full pass over the training data), and the loss on validation data was 2.0681. This number is very important because the lower it is, the better the checkpoint works. Once you start to generate data (discussed below), you will want to use the model checkpoint that has the lowest validation loss. Notice that this might not necessarily be the last checkpoint at the end of training (due to possible overfitting).

Another important quantities to be aware of are `batch_size` (call it B), `seq_length` (call it S), and the `train_frac` and `val_frac` settings.

1.	**The batch size** specifies how many streams of data are processed in parallel at one time.
2.	**The sequence length** specifies the length of each chunk, which is also the limit at which the gradients get clipped. For example, if `seq_length` is 20, then the gradient signal will never backpropagate more than 20 time steps, and the model might not *find* dependencies longer than this length in number of characters.
3.	At runtime your input text file has N characters, these first all get split into chunks of size BxS. These chunks then get allocated to three splits: train/val/test according to the `frac` settings. If your data is small, it's possible that with the default settings you'll only have very few chunks in total (for example 100). This is bad: In these cases you may want to decrease batch size or sequence length.

#### Parameters Description

The main parameters are:

1.	-loader: specifies the type of loader used for the dataset: Pos, Text, Series
2.	-dataDir: 'data/postagging', 'data directory. Should contain the file input.txt with input data'
3.	-model: the model to be used LTSM (for categorical values), LTSMN (for numerical values) , GRU or RNN'
4.	-layerSize: the number of neurons/cell per layers
5.	-layersNumber: the number of layers in the architecture

In the following a complete list of the parameters

```lua
cmd:option('-loader', 'Pos', 'Pos, Text, Series')
cmd:option('-dataDir', 'data/postagging', 'data directory. Should contain the file input.txt with input data')
cmd:option('-model', 'LTSM', 'LTSM, LTSMN , GRU or RNN')
-- model params
cmd:option('-layerSize', 128, 'size of LSTM internal state')
cmd:option('-layersNumber', 2, 'number of layers in the LSTM')
-- optimization
cmd:option('-learningRate', 2e-3, 'learning rate')
cmd:option('-learningRateDecay', 0.97, 'learning rate decay')
cmd:option('-learningRateDecayAfter', 10, 'in number of epochs, when to start decaying the learning rate')
cmd:option('-decayRate', 0.95, 'decay rate for rmsprop')
cmd:option('-dropout', 0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seqLength', 50, 'number of timesteps to unroll for')
cmd:option('-batchSize', 50, 'number of sequences to train on in parallel')
cmd:option('-maxEpochs', 50, 'number of full passes through the training data')
cmd:option('-gradClip', 5, 'clip gradients at this value')
cmd:option('-trainFrac', 0.95, 'fraction of data that goes into train set')
cmd:option('-valFrac', 0.5, 'fraction of data that goes into validation set')
-- bookkeeping
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-printEvery', 10, 'how many steps/minibatches between printing out the loss')
cmd:option('-evalValEvery', 500, 'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpointDir', 'cp', 'output directory where checkpoints get written, relative to dataDir')
cmd:option('-saveFileName', 'lstm', 'filename to autosave the checkpoint to. Will be inside checkpointDir/')
cmd:option('-initFrom', '', 'checkpoint file from which initialize network parameters, relative to checkpointDir')
cmd:option('-noResume', false, 'whether resume or not from last checkpoint')
-- GPU/CPU
cmd:option('-gpuId', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-openCL', false, 'use OpenCL (instead of CUDA)')
```

### Generating Text

Given a checkpoint file (such as those written to `cp`) we can generate new text. For example:

```
$ th sample.lua data/<dataset>/cp/some_checkpoint.t7 -gpuid -1
```

Make sure that if your checkpoint was trained with GPU it is also sampled from with GPU, or vice versa. Otherwise the code will (currently) complain. As with the train script, see `$ th sample.lua -help` for full options. One important one is (for example) `-length 10000` which would generate 10,000 characters (default = 2000).

1.	**Temperature**. An important parameter you may want to play with a lot is `-temperature`, which takes a number in range \[0, 1\] (notice 0 not included), default = 1. The temperature is dividing the predicted log probabilities before the Softmax, so lower temperature will cause the model to make more likely, but also more boring and conservative predictions. Higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes.

2.	**Priming**. It's also possible to prime the model with some starting text using `-primetext`. This starts out the RNN with some hardcoded characters to *warm* it up with some context before it starts generating text.

3.	**Training with GPU but sampling on CPU**. Right now the solution is to use the `convert_gpu_cpu_checkpoint.lua` script to convert your GPU checkpoint to a CPU checkpoint. In near future you will not have to do this explicitly. E.g.:

```
$ th convert_gpu_cpu_checkpoint.lua cv/lm_lstm_epoch30.00_1.3950.t7
```

will create a new file `cv/lm_lstm_epoch30.00_1.3950.t7_cpu.t7` that you can use with the sample script and with `-gpuid -1` for CPU mode.

Happy sampling!

### Predicting Text/Numbers
