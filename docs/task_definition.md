# Predictive task definition

The objective from these experiments is to the identify the machine's patterns through Machine Learing models. A way to interpret this objective, is to visualize its sensor data, generate interesting features, and use those features at a machine learning model to predict its classes.

Two possible problems approaches:

- attribute a single class for each sample (multi-class classfication problem);
- consider that multiple faults can happen at the same time (multi-label classification, can be considered binary classification for each fault).

The latter problem is more realistic considering a few factors:

- a multi-class approach fails when new classes appear (there must be an "others" class);
- multiple faults DO happen in practice;

But there are a few problems:

- For this dataset, the classes are labeled in a way that each sample only has a single fault happening. Therefore, it is not fair to the multi-label model;
- We turn a perfectly stratified dataset into a imbalanced dataset. Different metrics such as balanced accuracy, or ROC AUC must be used;
- More parameters to tune, such as the threshold for the ROC curve. It is possible to fix a certain metric such as TPR to diminish FNs, trading-off for FPR.

A few considerations:

- it would be better to have different machines/sensors and have them separated in the training/testing sets to avoid a possibe data leakage, where the model learns specific configurations of the machine. It is not possible to do this with this dataset.
- the classes are perfectly stratified for a multi-class approach, but will turn imbalanced when turning the problem into a multi-label.
