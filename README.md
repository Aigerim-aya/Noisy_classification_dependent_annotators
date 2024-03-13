# Noisy_classification_dependent_annotators
Noisy Classification for MNIST dependent annotators

# Data Simulator (Morphological Transformations) and Noisy Labels
We have assembled the dataset that is based on MNIST, where noise level depends on input image style for various annotators. Three type of image styles were simulated by performing morphological transformations on the original images, using [Morpho-MNIST software](https://github.com/dccastro/Morpho-MNIST). Generated image styles (annotations) are: good-segmentation (which is similar to original images), thin (under-segmentation), and thick (over-segmentation). Noise types are symmetric, pairflip, asymmetric and pairflip with permutation where applied. The type and level of noises applied to original labels are provided in this table: 

<img width="550" alt="image" src="https://github.com/Aigerim-aya/Noisy_classification_dependent_annotators/assets/95924311/cbbf08bf-d407-4933-8447-76a69e5b4ade">.

DataPreparation_MNIST.ipynb notebook was used to create dataset with noisy labels according to this table.







For a dataset consisting of three different types of images (original, thin, and thick) and three different annotators (Table 4), we compare (i) base classifier model without annotators and regularization, (ii) our approach without reg-
ularization, and (iii) our approach with information-based regularization. Each annotator NN has similar architecture as in classifier model and takes images as an input.
