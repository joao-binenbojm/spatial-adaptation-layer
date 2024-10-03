# SPATIAL ADAPTATION LAYER
**Article:** https://arxiv.org/pdf/2409.08058

This repository will be used to investigate the effect of the newly implemented Spatial Adaptation Torch layer. This layer assumes that from one session to another, performance drops can be accounted for by accounting for an affine transformation applied to grid coordinates, and accounting for baseline normalization (subtracting by the baseline amount of activity besides rest). This study focuses on EMG surface images only, with explicit temporal component (although sliding RMS surface images are primarily considered). This repository contains various different elements and it's important to be familiar with them when working with them.

## Novelties
The primary novelties introduced in this study are the Spatial Adaptation Torch layer and the learnable baseline. Both operations are inspired by the work of the creators of the CSL dataset [1], and based on the theorical framework developed in the "Spatial Tranformers Network" study [3]. They consider both accounting for the electrode shift of the surface RMS image with two separate image processing algorithms, as well as subtracting the average EMG RMS activity during rest from every given surface RMS image, hence accounting for cross-session differences in EMG baseline. The novelty in this approach comes where affine transformation parameters are treated as learnable parameters in a domain adaptation setting. After adaptation, shifted coordinates to return a translated image via interpolation (linear or cubic). Additionally, since baseline normalization is used to account for activity baseline differences across sessions, we simply subtract learnable biases (baseline) from the test session, learning how to account for such baseline fluctuations. This way, both baseline, spatial transformation parameters can be jointly and simply be optimized in an interpretable, few-shot adaptation setting.

## Data Pipeline
Before the model initialization, the data pipeline has two components:
__EMG tensorizer__: this carries out most of the preprocessing, and essentially extracts from different time-windows/time-instants HD-sEMG frames. It extracts a consistent number of frames per repetition of each gesture, resulting in a regular data structure that can be stored in a Tensor. This enables us to go from unstructed EMG time-series to a tensor with different dimensions for different sessions, gestures, repetitions, given frame, ypos and xpos, which enables a very flexible structure for a variety of experiments with simple unfolding, slicing and flattening operations.  

__EMG Frame Loader__: this primarily acts as a PyTorch Dataset wrapper to load the appropriate tensors and add compatibility with Torch DataLoaders.

## Running Scripts
The two primary scripts to be used in this investigation are the intrasession/intersession Python scripts, which are capable of running each of the respective experiments for a given dataset. While you may need to tweak some things within the respective scripts, most of the conditions that will vary along experiments are encoded in two .json files loaded before any data loading:
___{dataset}_.json__: which contains information regarding the dataset that will be employed (e.g. input shape or number of gestures).
__exp.json__: this contains information regarding the experimental conditions, and includes things such as optimizer hyperparameters, preprocessing choices, or even fine-tuning choices.

## Changing Capgmyo data format
There is a file in utils that is used to convert Capgmyo file naming/folder conventions into the CSL conventions (excluding the MVC recordings) to make the data loading procedure between the two datasets more uniform. Check out the EMG tensorizers to understand how to structure your directories!

## References
1. https://dl.acm.org/doi/10.1145/2702123.2702501
2. https://www.mdpi.com/1424-8220/17/3/458
3. https://arxiv.org/abs/1506.02025

