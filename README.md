# SHIFT LAYER
This repository will be used to investigate the effect of the newly implemented Shift Torch layer. This layer assumes that from one session to another, performance drops can be accounted for by accounting for electrode shift (vertical and horizontal), and accounting for baseline normalization (subtracting by the baseline amount of activity besides rest). This study focuses on EMG surface images only, with explicit temporal component (although sliding RMS surface images are primarily considered). This repository contains various different elements and it's important to be familiar with them when working with them.

## Novelties
The primary novelties introduced in this study are the Shift Torch layer and the learnable baseline. Both operations are inspired by the work of the creators of the CSL dataset [1]. They consider both accounting for the electrode shift of the surface RMS image with two separate image processing algorithms, as well as subtracting the average EMG RMS activity during rest from every given surface RMS image, hence accounting for cross-session differences in EMG baseline. The novelty in this approach comes from the fact that the horizontal and vertical electrode shifts are considered as learnable parameters (xshift, yshift) via the Shift layer, which uses these learnable parameters in a layer that takes shifted coordinates to return a translated image via interpolation (linear or cubic). Additionally, since baseline normalization is used to account for activity baseline differences across sessions, we simply subtract learnable biases (baseline) from the test session, learning how to account for such baseline fluctuations. This way, both baseline, xshift and yshift can be jointly and simply be optimized in an interpretable, few-shot adaptation setting.

Given the more complex/general Spatial Transformer module introduced by DeepMind in 2015 [3], the Shift layer is implemented via the torch.grid_sample function that supports backpropagation, making it for a very straightforward and well-understood implementation.

## Data Pipeline
Before the model initialization, the data pipeline has two components:
__EMG tensorizer__: this carries out most of the preprocessing, and essentially extracts from different time-windows/time-instants HD-sEMG frames. It extracts a consistent number of frames per repetition of each gesture, resulting in a regular data structure that can be stored in a Tensor. This enables us to go from unstructed EMG time-series to a tensor with different dimensions for different sessions, gestures, repetitions, given frame, ypos and xpos, which enables a very flexible structure for a variety of experiments with simple unfolding, slicing and flattening operations.  

__EMG Frame Loader__: this primarily acts as a PyTorch Dataset wrapper to load the appropriate tensors and add compatibility with Torch DataLoaders.


## Capgmyo Extra Considerations
Since, unlike the CSL dataset, Capgmyo is not a uniform grid (the distance between every adjacent electrode is not the same) [2], we need to give it a special preprocessing in order to convert it to a uniform grid. Given the capgmyo structure, we know the vertical displacement between electrodes. Within a single grid, we also know the horizontal displacement between electrodes. The only unknown becomes the horizontal distance between adjacent electrodes from different grids, referred to in the code as dg.

To deal with this inconsistency, we treat dg as a hyperparameter explored within a human-realistic range of values (20-40mm), and given a dg, we can use the original Capgmyo electrode coordinates to regrid them into a uniform grid within the same area. We do this ensuring that all adjacent electrodes are separated by an interelectrode distance of 1mm (same as in CSL). While dg could probably be learned like the learnable electrode shift parameters, since it's only specific to the Capgmyo dataset, we chose to optimize with brute-force without loss of generalization.

With this approach, we are not exploiting the circular nature of the Capgmyo data (grid wrapping around the forearm), and depending on the value of dg, we may have a different number of interpolated 'electrodes' in the horizontal direction (as we can fit more electrodes separated by 1mm in the same region).

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

