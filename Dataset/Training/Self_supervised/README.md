# HaaS and NLP2Bash

This folder contains the data from the [Honeypot as a Service](https://haas.nic.cz/) and [NLP2Bash](https://github.com/TellinaTool/nl2bash/blob/master/data/bash/all.cm) projects.

## Honeypot as a service (HaaS)

Dataset chosen as representative of **malign sessions**. 

If you find this project useful, please consider citing the following URL:

- Title: HaaS
- URL: (https://haas.nic.cz/)

## NLP2Bash

Dataset chosen as representative of **benign sessions**. 

If you use this code or find it helpful, please consider citing the following dataset:

- Dataset Name: NL2Bash: A Corpus and Semantic Parser for Natural Language Interface to the Linux Operating System
- Author(s): Xi Victoria Lin and Chenglong Wang and Luke Zettlemoyer and Michael D. Ernst
- Year: 2018
- Publisher: Proceedings of the Eleventh International Conference on Language Resources
               and Evaluation
- URL: (https://github.com/TellinaTool/nl2bash/tree/master)

## HOWTO

To group the files from the `.\Training_Chunks` and `.\Validation_Chunks` folders, run:

```shell
cat ./Training_Chunks/chunk* > training_set.csv
```
and 
```shell
cat ./Validation_Chunks/chunk* > validation_set.csv
```
