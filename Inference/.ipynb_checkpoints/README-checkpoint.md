# Inference
This folder contains the code to use the model previously adapted and finetuned for inference.

## Inference
This notebook contains the inference procedure. It might take sometimes running, given the high number of examples in the dataset.

## Join corpus and prediction
This notebook perform the join between the predicted tokens and the original corpus (notice: the model made 1 prediction per word. Now we have to go back to the original sessions, i.e., sequences of words).