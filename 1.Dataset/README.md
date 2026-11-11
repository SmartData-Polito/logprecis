# Dataset 💾

This folder contains the datasets we use for the LogPrécis project. More info in the subfolders.

## Training

Here we report the datasets for:

- the **self-supervised** (i.e., Masked Language Modelling) training to perform the domain-adaptation of the original LM toward Unix;
- the **supervised task** of Named-Entity Recognition to obtain the attackers' intents from the coarse sessions

## Inference

Here we report the dataset to use at inference time once that the LM has been trained for the supervised task.

## License

Our annotations, curation and data splits are released under [CC BY 4.0](../LICENSE-DATA). This does **not** relicense the third-party sources these datasets are built on — each keeps its upstream license (HaaS: Terms of Use; NL2Bash `data/bash`: MIT; CyberLab: CC BY 4.0). See [`../NOTICE`](../NOTICE) and the per-folder notes below.
