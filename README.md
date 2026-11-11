# LogPrécis

[![Code License: Apache 2.0](https://img.shields.io/badge/Code%20License-Apache%202.0-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cose.2024.103805-blue.svg)](https://doi.org/10.1016/j.cose.2024.103805)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-SmartDataPolito-yellow.svg)](https://huggingface.co/SmartDataPolito)

LogPrécis is a powerful tool designed to automate and enhance the workflow of security experts by leveraging the capabilities of Language Models (LMs). With LogPrécis, you can efficiently analyze and extract valuable information from large-scale log data, significantly improving your incident response and threat-hunting processes.

# Availability 🤗

**Models** are available on [Huggingface](https://huggingface.co/SmartDataPolito):

- [LogPrecis](https://huggingface.co/SmartDataPolito/logprecis): model to solve intents classification
- [SecureShellBert](https://huggingface.co/SmartDataPolito/SecureShellBert): model adapted for Unix-Shell

**Datasets** are available both in this [repo](https://github.com/SmartData-Polito/logprecis/tree/main/1.Dataset) and on the [SmartData site](https://smartdata.polito.it/logprecis/).

## Features

- **Automated Log Analysis** LogPrécis utilizes advanced LMs to automatically analyze and process vast amounts of log data. It can understand and interpret log entries, allowing for the effective identification of security incidents, anomalies, and patterns.

- **Precise Log Summarization** The tool provides concise and accurate log fingerprints, extracting essential details from lengthy log entries. By condensing log information, LogPrécis enables security experts to quickly grasp the key insights and focus on critical security events.

- **Visualization and Reporting**: LogPrécis offers interactive visualizations and customizable reports to present log analysis results in a meaningful way. Visual representations help in identifying trends, outliers, and patterns, while reports aid in communicating findings and recommendations to stakeholders.

## Getting Started

1. Clone the LogPrécis repository to your local machine:

```shell
git clone https://github.com/SmartData-Polito/logprecis.git
```

2. Install the required dependencies:

```shell
conda create --name <env> --file requirements.txt
```

## Howto - Reproducibility

![LogPrécis flow](map.png)
It also explains how you should access the code.

**Notice**: while, if you don't have you're own inference data, you cannot skip the first point of the pipeline (follow the `README` at `1.Dataset/Inference/`), you can already leverage both the domain adapted and finetuned models. Those are available on Huggingface:

- Domain adapted (step 2): [SmartDataPolito/SecureShellBert](https://huggingface.co/SmartDataPolito/SecureShellBert)
- Finetuned for intent detection (step 3): [SmartDataPolito/logprecis](https://huggingface.co/SmartDataPolito/logprecis)

To load those models, simply use their names when running the scripts at `2.Training/supervised_training/` setting the parameter `model_name`.

## License

Our own contributions in this repository are released as follows:

- **Code and models** — [Apache License 2.0](LICENSE). This covers the source code and the released models [LogPrecis](https://huggingface.co/SmartDataPolito/logprecis) and [SecureShellBert](https://huggingface.co/SmartDataPolito/SecureShellBert).
- **Datasets** — [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA). This covers the annotations, curation and data splits we release.

### Third-party data and models

These licenses apply to **our own work only**. They do **not** relicense third-party material we build on: each upstream source keeps its original license, and you must comply with it (and attribute it) in addition to the LogPrécis license. See [`NOTICE`](NOTICE) for the full list. In short:

| Component | Where used | Upstream license |
|---|---|---|
| [CodeBERT](https://github.com/microsoft/CodeBERT) (`microsoft/codebert-base`) | Base model for both released models | MIT — compatible with our Apache 2.0 release; MIT notice retained |
| [CyberLab honeynet](https://doi.org/10.5281/zenodo.3687527) | Inference data | CC BY 4.0 — attribute Sedlar et al. |
| [NL2Bash](https://github.com/TellinaTool/nl2bash) (`data/bash`) | Self-supervised (benign sessions) | MIT (dataset) — retain MIT notice |
| [HaaS](https://haas.nic.cz/) | Supervised corpus + self-supervised (malign sessions) | HaaS **Terms of Use** — *not* an open-data license |

> **Note on HaaS.** HaaS does not publish an open-data license such as CC BY. Our CC BY 4.0 license therefore covers **only our own annotations, selection and curation** of the 360 labeled sessions — **not** the underlying HaaS session content. To obtain and reuse the raw HaaS sessions, follow the [HaaS Terms of Use](https://haas.nic.cz/).

## Contact

If you have any questions, feedback, or suggestions regarding LogPrécis, feel free to reach out to our team at [matteo.boffa@polito.it](mailto:matteo.boffa@polito.it) and [idilio.drago@unito.it](mailto:idilio.drago@unito.it).

LogPrécis is an invaluable tool that empowers security experts to harness the full potential of Language Models for log analysis. By automating log processing and providing precise summaries, LogPrécis helps security teams stay one step ahead in the ever-evolving threat landscape. Start leveraging LogPrécis today and streamline your security operations like never before!
