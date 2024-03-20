# SecureShellBert 🤖

This folder contains the code to adapt an [Huggingface model](https://huggingface.co/models) for Unix-Bash language. The adaptation is performed solving a **Masked Language Modelling** (MLM) task and follows

Current model used for LogPrecis was adapted with a corpus of 21k sessions, both malign and benign, available at `1.Dataset/Training/Self_supervised`. In general, the input must be a Pandas Dataframe (accepted format: json, parquet, csv) used for training with the following format:

|                       session                       |
| :-------------------------------------------------: |
|  scp -t /tmp/Muw3fuvA ; cd /tmp && chmod +x Muw...  |
| cat /proc/cpuinfo \| grep name \| wc -l ; echo r... |
|     echo -en '\x31\x33\x33\x37' ; cat /bin/ls ;     |

The script also accepts both an `eval_size` and a `validation_path`. In the first case, the script will first load the dataset and then split it into train and validation; in the second, the script will try loading a validation dataframe (with the same format as above) at the specified path.

### Reproduce SecureShellBert

Refer to `./experiments/reproduce_secure_shell_bert.sh` to reproduce [SecureShellBert](https://huggingface.co/SmartDataPolito/SecureShellBert). For the domain adaptation we used:

- 10 epochs
- mlm probability of 0.15
- batch size = 16
- learning rate of 1e-5
- chunk size = 256
