# Linearity-based compression
![Figure1](assets/figure1.svg)

Linearity-based compression is a novel technique to compress neural networks by exploiding very active activation functions. This repo implements the basic compression algorithm and presents some example compressions.

## Overview 
Neural network compression aims to reduce model size of already trained models. Most current approaches focus on identifying unimportance or redundancy in the network or use general data compression methods.

The best results are achieved by combination of multiple compression approaches. The introduction of linearity-based compression offers a new angle to compress yet not considered components and thus further improve the state-of-the-art in neural network compression.

### basic idea

Neural networks require non-linear activation functions to exploid their compositional architecture. With linear activation functions the entire model could be collapsed into a single layer between input and output.

However, similar to dead ReLUs can ReLU-like activation functions end up linear dependent on their inputs and trained parameters. Linearity-based compression identifies linear or often linear ReLUs and collapses them into the next layer. 

## Dependencies

Python 3.11.x

all dependencies can be loaded with the requirements file

```sh
pip install -r requirements.txt
```

## Repo structure

The main class of interest is the model_compression class in the compression/compression.py file. This class is used for the compression and can be inspected to better understand the procedure of the compression idea.

The other file compression/importance_based_pruning.py in the folder implements another simple compression approach which is used to compare and combine with the linearity-based compression.

The Prepare folder contains all files needed to prepare the data in the example_compression notebook

The example_compression.ipynb demonstrates how to apply the compression on different models.

## Usage

The model to be compressed needs to be extended to include a method called "extended_relu_forward" that outputs a list of all ReLU outputs in the network in each forward pass. The simple keras2torch_converter in the utils.py file shows how this method generally can be implemented.

The model_compression class has an layer_threshold parameter that specifies how many linear neurons are required before a layer is skipped. There are three options for this parameter
- a integer results in a absolute layer threshold in size of the integer
- a float results in a relative layer threshold and therefore should be between 0 and 1
- a string results in the calculated optimal layer-threshold 

## Computational time

The entire example_compression.ipynb notebook took approx. 10 hours on a basic consumers CPU.

One single compression loop consisting of 20 compressions takes approx. 30 minutes.
