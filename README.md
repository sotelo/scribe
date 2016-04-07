# Your own personal scribe.

This repo has an implementation of handwriting synthesis using recurrent neural networks. The details of the algorithm are described in this [paper](http://arxiv.org/abs/1308.0850) by Alex Graves.

It uses [blocks](https://github.com/mila-udem/blocks) for the model and [fuel](https://github.com/mila-udem/fuel) for the data processing.

This is work in progress...

## Getting started

### Write something
Since there's a trained model included, the only thing you need to make your scribe write something is to run:
```
	python sample.py --phrase "I want you to write me."
```

### Train from scratch
The first thing you need to do is to download the data. You have to register [here](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database) and download these two files:
 * lineStrokes-all.tar.gz
 * ascii-all.tar.gz

Save these two files in $FUEL_DATA_PATH/handwriting and run:
```
	python preprocess_data.py
	python train.py
```

## To Do
 * Improve documentation.
 * Fix code to decide when to finish sampling.
 * Write code for weight noise and variational objective.
 * Make model more flexible. Right now the number of layers is hardcoded.
 * Implement Samy Bengio's scheduled sampling. 
 * Test multigpu results. Speed. Generalization.
 * Benchmark standardized data against scaled-only data.

## References:
 * https://github.com/udibr/sketch
