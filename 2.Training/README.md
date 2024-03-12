# Training

This folder train an existing Language Model (i.e., CodeBERT) to solve a Named Entity Recognition (NER) task.
Specifically, we want to assign to each entity (i.e., BASH token) a [MITRE Tactic](https://attack.mitre.org/tactics/enterprise/).

## Domain Adaptation

The chosen LM is pre-trained (i.e., domain-adapted) with bash scripts via Self-Supervision: in particular, we will use the bening and malign UNIX examples provided in `1.Dataset/Training/Self_Supervision`.

## Supervised Training

Once pre-trained, one can attach a classification head on top of the LM and solve downstream tasks such as NER. To solve this task, hopefully a smaller number of labeled examples is required (i.e., few-shot learning). In this second phase, use the examples from `1.Dataset/Training/Supervision`.
