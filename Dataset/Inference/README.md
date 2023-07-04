# Cyberlab

This folder contains the data from the [CyberLab honeynet dataset](https://zenodo.org/record/3687527#.YmEr9pJBxQL).

To group the files from the `.\Chunks` folder, run:

```shell
cat ./Chunks/chunk* > all_files.csv
```

Please consider citing the following dataset:

- Dataset Name: CyberLab honeynet dataset
- Author(s): Sedlar Urban, Kren Matej, Štefanič Južnič Leon, Volk Mojca
- Year: 2020
- Publisher: Zenodo
- DOI or URL: 10.5281/zenodo.3687527
- URL: {https://doi.org/10.5281/zenodo.3687527}

## Preprocessing
The dataset is first characterized on a Vanilla non-neural way (e.g., simple info on collection period, etc). Then, is processed to make it coherent with training stats (chunked accordingly).