# Cyberlab

This folder contains the data from the [CyberLab honeynet dataset](https://zenodo.org/record/3687527#.YmEr9pJBxQL).

Notice that the other collection we used in the paper (Polito data) are available in the [SmartData webpage](https://mplanestore.polito.it:5001/sharing/S66xSfAiF).

To group the files from the `.\Chunks` folder, run:

```shell
cat ./Chunks/cyberlab_chunk* > cyberlab_data.csv
```

The `csv` file contains 233,037 sessions, containing metadata such as `sensor` (sensor which collected the attack), `first_timestamp` (beginning of attack) and `date` (parsed date from the timestamp).

**Please consider citing the following**:

- Dataset Name: CyberLab honeynet dataset
- Author(s): Sedlar Urban, Kren Matej, Štefanič Južnič Leon, Volk Mojca
- Year: 2020
- Publisher: Zenodo
- DOI or URL: 10.5281/zenodo.3687527
- URL: {https://doi.org/10.5281/zenodo.3687527}
