import os
from transformers import AutoTokenizer
from core.functions.tokenizer_functions import (
    align_labels_with_statements,
    align_labels_with_tokens,
    mask_statement_context_predictions,
    mask_token_context_predictions,
)
from core.functions.utils import create_dir


class LogPrecisTokenizer:
    def __init__(self, opts):
        self.tokenizer_name = opts["tokenizer_name"]
        self.classified_entity = opts["entity"]
        self.special_token = opts["special_token"]
        self.max_chunk_length = opts["max_chunk_length"]
        self.task = opts["task"]
        self.adaptation = opts["adaptation"]
        self.load_tokenizer()

    def load_tokenizer(self):
        """Load the tokenizer.

        This method loads the tokenizer from the specified tokenizer name.
        If the tokenizer is not found, it downloads the default tokenizer and caches it.
        If the classified entity is a statement, it adds a special token to the tokenizer.
        """
        if not os.path.exists(self.tokenizer_name):
            # Default tokenizer, will be downloaded the first time and then cached in default cache folder
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, add_prefix_space=True
            )
            if self.classified_entity == "statement":
                # Adding special token...
                self.tokenizer.add_tokens([self.special_token], special_tokens=True)
                self.id_special_token = self.tokenizer(
                    [self.special_token], is_split_into_words=True
                )["input_ids"][1]
        else:
            # If finetuned, no need to add token
            config_name = os.path.join(self.tokenizer_name, "tokenizer_config.json")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, config=config_name, add_prefix_space=True
            )

    def tokenize_data(self, dataset_obj, logger):
        """Tokenize the dataset.

        This method tokenizes the dataset using the provided tokenizer and tokenizing function,
        and saves the tokenized datasets to cache files.

        Args:
            dataset_obj (DatasetObject): The dataset object containing the dataset to be tokenized.
            logger (Logger): A logger object for logging and debugging purposes.
        """
        dataset_name = dataset_obj.dataset_name
        cache_file_name = os.path.join(
            logger.experiment_dir, "cache", f"{dataset_name}_tokenized_corpus"
        )
        create_dir(cache_file_name)
        ds = dataset_obj.ds
        cache_file_names = {
            partition: os.path.join(cache_file_name, f"{partition}.arrow")
            for partition in ds.keys()
        }
        if self.adaptation == "single_session":
            tokenizing_function = (
                self.mlm_tokenizing_function_single_session
                if self.task == "self_supervision"
                else self.entity_classification_tokenizing_function
            )
        else:
            tokenizing_function = (
                self.mlm_tokenizing_function
                if self.task == "self_supervision"
                else self.entity_classification_tokenizing_function
            )
        tokenized_datasets = ds.map(
            tokenizing_function,
            batched=True,
            remove_columns=(
                ds["train"].column_names if "train" in ds else ds["test"].column_names
            ),
            fn_kwargs={"max_length": self.max_chunk_length},
            num_proc=1,
            cache_file_names=cache_file_names,
        )
        dataset_obj.set_tokenized_corpus(tokenized_datasets)

    def mlm_tokenizing_function(self, examples, max_length):
        """Tokenizing function for MLM from https://huggingface.co/learn/nlp-course/chapter7/3.
        All tokenized sessions will be concatenated and chunked. Last chunks will be discarded if < max_length.
        Args:
            examples (dict): The input examples to be tokenized.
            max_length (int): The maximum length allowed for the tokenized inputs.
        Returns:
            dict: The tokenized inputs and aligned labels.
        """
        #### Tokenize data ####
        result = self.tokenizer(examples["final_input"], truncation=False)
        #### Now, create chunks of max_lenght size ####
        # Concatenate all texts
        concatenated_examples = {k: sum(result[k], []) for k in result.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(result.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column (since self_supervised, labels come from data themselves)
        result["labels"] = result["input_ids"].copy()
        return result
    
    def mlm_tokenizing_function_single_session(self, examples, max_length):
        """Tokenizing function that puts at maximum one session for each batch
        Args:
            examples (dict): The input examples to be tokenized.
            max_length (int): The maximum length allowed for the tokenized inputs.
        Returns:
            dict: The tokenized inputs and aligned labels.
        """
        #### Tokenize data ####
        result = self.tokenizer(examples["final_input"], truncation=False)
        tmp = []
        for r in result["input_ids"]:
            if len(r) < self.max_chunk_length:
                tmp.append(r)
            for i in range(0, int(len(r)/self.max_chunk_length)):
                if len(r[i*self.max_chunk_length:i*self.max_chunk_length+self.max_chunk_length]) > 10:
                    tmp.append(r[i*self.max_chunk_length:i*self.max_chunk_length+self.max_chunk_length].copy())
        fill = [0] * (self.max_chunk_length)
        tmp_attention = [[1]*len(sublist[:self.max_chunk_length]) + fill[len(sublist):] for sublist in tmp]

        fill = [self.tokenizer.pad_token_id]* (self.max_chunk_length)
        tmp_input = [
            sublist[: self.max_chunk_length]
            + fill[len(sublist[: self.max_chunk_length]):]
            for sublist in tmp
        ]
        #### Now, create chunks of max_lenght size ####
        # Create a new labels column (since self_supervised, labels come from data themselves)
        result["input_ids"] = tmp_input
        result["attention_mask"] = tmp_attention
        result["labels"] = result["input_ids"].copy()

        return result

    def entity_classification_tokenizing_function(self, examples, max_length):
        """Tokenize examples.
        This method tokenizes the input examples using the provided tokenizer, truncating them if necessary,
        and aligns the labels with the tokenized inputs.
        Args:
            examples (dict): The input examples to be tokenized.
            max_length (int): The maximum length allowed for the tokenized inputs.
        Returns:
            dict: The tokenized inputs and aligned labels.
        """
        # First, handle the inputs (same function for training and inference)
        tokenized_inputs = self.tokenizer(
            examples["final_input"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
        )
        # Get context statements and words
        list_context_statements = examples["indexes_statements_context"]
        list_context_words = examples["indexes_words_context"]
        # We have to prevent backpropagation from those words/statements that are part of the context.
        if "final_labels" in examples.keys():
            all_labels = examples["final_labels"]
            new_labels = []
            for i, labels in enumerate(all_labels):
                input_ids = tokenized_inputs["input_ids"][i]
                if self.classified_entity == "statement":
                    context_statements = list_context_statements[i]
                    new_labels.append(
                        align_labels_with_statements(
                            labels=labels,
                            tokens_ids=input_ids,
                            special_token_id=self.id_special_token,
                            context_statements=context_statements,
                        )
                    )
                elif self.classified_entity == "word":
                    word_ids = tokenized_inputs.word_ids(i)
                    context_words = list_context_words[i]
                    new_labels.append(
                        align_labels_with_tokens(
                            labels=labels,
                            word_ids=word_ids,
                            indexes_context_words=context_words,
                            loss_per_token=False,
                        )
                    )
                elif self.classified_entity == "token":
                    word_ids = tokenized_inputs.word_ids(i)
                    context_words = list_context_words[i]
                    new_labels.append(
                        align_labels_with_tokens(
                            labels=labels,
                            word_ids=word_ids,
                            indexes_context_words=context_words,
                            loss_per_token=True,
                        )
                    )
            tokenized_inputs["labels"] = new_labels
        # If there are no labels, there will be no loss and we simply have to ignore the predictions for context words
        else:
            context_predictions = []
            for it in range(len(examples["final_input"])):
                context_words = list_context_words[it]
                word_ids = tokenized_inputs.word_ids(it)
                input_ids = tokenized_inputs.input_ids[it]
                if self.classified_entity == "statement":
                    context_predictions.append(
                        mask_statement_context_predictions(
                            context_words=context_words,
                            word_ids=word_ids,
                            input_ids=input_ids,
                            special_token_id=self.id_special_token,
                        )
                    )
                else:
                    context_predictions.append(
                        mask_token_context_predictions(
                            context_words=context_words,
                            word_ids=word_ids,
                            input_ids=input_ids,
                        )
                    )
            # THE KEY MUST BE 'labels' (EVEN IF THERE ARE NO) > OTHERWISE DATA COLLATOR DOES NOT WORK!
            tokenized_inputs["labels"] = context_predictions
        tokenized_inputs["session_id"] = examples["session_id"]
        tokenized_inputs["order_id"] = examples["order_id"]
        return tokenized_inputs
