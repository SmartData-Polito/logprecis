# Training

This folder train an existing NLP LM (i.e., CodeBERT) to solve a Named Entity Recognition (NER) task.
Specifically, we want to assign to each entity (i.e., BASH token) a [MITRE Tactic](https://attack.mitre.org/tactics/enterprise/).

## Domain Adaptation

Each LM is pre-trained via Self-Supervision: in particular, is trained with Masked Language Modelling on Natural Language (NL). To adapt the model to UNIX language, we can complete this pre-training with bening and malign UNIX examples.

## Finetuning

Once pre-trained, one can attach a classification head on top of the LM and solve downstream tasks such as NER. To solve this task, hopefully a smaller number of labeled examples is required (i.e., few-shot learning).