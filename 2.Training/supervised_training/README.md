# LogPrécis 🤖

This folder contains the code to perform training and inference leveraging [LogPrécis](https://arxiv.org/abs/2307.08309), a tool that solves an entity classification problem over attacking logs.

## Input data 🗂️

The code takes as input:

- Pandas Dataframes (accepted format: json, parquet, csv) as **training dataframe** with the following format:

|                       session                       |                      labels                       |
| :-------------------------------------------------: | :-----------------------------------------------: |
|  scp -t /tmp/Muw3fuvA ; cd /tmp && chmod +x Muw...  |                   Execution - 4                   |
| cat /proc/cpuinfo \| grep name \| wc -l ; echo r... | Discovery - 2 -- Persistence - 5 -- Discovery ... |
|     echo -en '\x31\x33\x33\x37' ; cat /bin/ls ;     |                   Discovery - 1                   |

- Pandas Dataframes (accepted format: json, parquet, csv) as **inference dataframe** with the following format:

|                       session                       |
| :-------------------------------------------------: |
|  scp -t /tmp/Muw3fuvA ; cd /tmp && chmod +x Muw...  |
| cat /proc/cpuinfo \| grep name \| wc -l ; echo r... |
|     echo -en '\x31\x33\x33\x37' ; cat /bin/ls ;     |

On both cases, the code automatically pre-process the dataset according to pre-processing (e.g., truncation of long words) described in the paper. Also, according to the truncation strategy chosen as parameter by the user, the text will be raw-chunked or context-chunked. When possible, the operations will be performed only once and **cached** for the following times.

## Train a model 🏋️

To train a model, refer to the train.sh script.

Here, the user can specify training parameters, such as an `identifier` for the run, the chosen model, whether to use a finetuned model or not, etc.

To obtain a guideline of possible parameters, run:

`python ./train.py --help`

User can also specify an `output_path` where the model saves its scores/metrics. Once finished the training, you can access the training results through a Tensorboard interface (`tensorboard --logdir=PATH_TO_LOGS`). Notice that logs are in the same `output_path` folder of before.

### Reproduce LogPrecis

After a grid search with 5 different seeds (1, 1997, 29, 533, 7) and a ratio 70-30 between training and validation, LogPrecis (our best model) was trained for:

- 47 epochs
- token classification
- context chunking
- all 360 labelled sessions (no validation set)
- eval size = 0
- batch size of 8
- seed = 1
- lr = 1e-6

## Use a trained model for test/inference 💪

To use a model for inference/test, refer to the inference.sh script.

Remember that, to obtain reasonable performance, you must perform inference/test only after finetuning a model for the desired task/using a model from the Huggingface Hub that was finetuned for that task.

In the first case when specifying the `finetuned_path`, make sure that the path has access to the subfolder `best_model`, where the weights of the finetuned model are stored. For instance, it could be:

`finetuned_path="./experiments/bert-uncased-base/attempt/seed_1"`,
from which you can access:
`finetuned_path="./experiments/bert-uncased-base/attempt/seed_1/best_model"`

On the second case, simply specify a `model_name` that exists [here](https://huggingface.co/models). In that case, you can leave the `finetuned_path` empty. Remember that to obtain the best results, it is better that the loaded model was trained with the same truncation strategy (`truncation`) and entities (`entity`) as your inference data.

In general, both testing and inference saves the model predictions, logits and embeddings (CLS token of the last layer by default). The model will also compute the classification metrics (test) if the input data contains a `labels` columns.

To obtain a guideline of possible parameters, run:

`python ./inference.py --help`
