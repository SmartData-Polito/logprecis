from core.functions.preprocessing_functions import (
    divide_statements,
    assign_labels2tokens,
    strip_strings,
    statement2word,
)
from core.classes.dataset import DataHandler

# To ignore verbose warnings
import datasets  # nopep8
import warnings  # nopep8

warnings.simplefilter(action="ignore", category=FutureWarning)  # nopep8
datasets.utils.logging.set_verbosity(datasets.utils.logging.ERROR)  # nopep8
datasets.utils.logging.enable_progress_bar()  # nopep8


class HoneyDataForClassification(DataHandler):
    # Class to load and prepare data for token classification.
    def __init__(self, opts):
        super().__init__(opts)
        self.truncation = opts["truncation"]
        self.training_stage = opts["training_stage"]
        self.classified_entity = opts["entity"]
        self.available_percentage = opts["available_percentage"]
        self.eval_size = opts["eval_size"]
        self.id2labels = None
        self.labels2id = None

    def load_dataset(self, logger, model_obj=None):
        """Load and preprocess the dataset.
        This method preprocesses the dataset, splits it into training, validation, and test sets,
        loads information on labels, and converts Pandas DataFrames into Huggingface Datasets format.
        Args:
            logger (Logger): A logger object for logging and debugging purposes.
            model_obj (AutoModelForTokenClassification, optional): model containing the mapping id2label and label2id. Present only during inference.
        """
        logger.experiment_logger.info(
            f"Pre-processing dataset (might take sometimes)..."
        )
        self.preprocess_data(logger)
        logger.experiment_logger.info(f"Split dataset into train and validation...")
        if self.training_stage == "training":
            train_dataset, valid_dataset, test_dataset = self.split_data(
                logger.experiment_logger, self.eval_size
            )
            logger.experiment_logger.debug(f"Load/save info on labels...")
            self.load_label_mapping(logger)
        else:
            # Train and valid here are simple placeholders
            train_dataset, valid_dataset, test_dataset = self.extract_test()
            self.load_label_mapping(logger, model_obj)
        logger.experiment_logger.debug(
            f"Convert Pandas Dataframe into Huggingface Dataset..."
        )
        self.pd_2_hf(train_dataset, valid_dataset, test_dataset)

    def preprocess_data(self, logger):
        """Preprocess the dataset.
        This method loads preprocessed data if available, otherwise preprocesses the dataset
        by selecting a subsample, truncating long words, dividing sessions into statements,
        creating session IDs, splitting sessions into chunks, performing sanity checks,
        and saving the preprocessed data.
        Args:
            logger (Logger): A logger object for logging and debugging purposes.
        """
        self.final_dataset = logger.load_cached_data(
            f"preprocessed_{self.dataset_name}.parquet"
        )
        if self.final_dataset.shape[0] == 0:  # No cache availble, do preprocessing
            logger.experiment_logger.debug("\tPreprocessing the data...")
            logger.experiment_logger.debug(
                "\t\tSelect subsample of data if required..."
            )
            self.select_subsample()
            logger.experiment_logger.debug("\t\tTruncate long words...")
            self.truncate_long_words()
            logger.experiment_logger.debug("\t\tDividing sessions into statements...")
            self.obtain_statements_info(logger=logger)
            logger.experiment_logger.debug("\t\tCreating session id...")
            self.create_session_id()
            logger.experiment_logger.debug("\t\tSplit sessions into chunks...")
            self.create_chunks()
            logger.experiment_logger.debug("\t\tComplete preprocessing...")
            self.complete_preprocessing()
            logger.experiment_logger.debug(
                "\t\tSanity check: remove sessions in which the split failed..."
            )
            if self.classified_entity == "statement":
                self.chunking_sanity_check(
                    logger.experiment_logger, add_special_token=True
                )
            else:
                self.chunking_sanity_check(
                    logger.experiment_logger, add_special_token=False
                )
            logger.experiment_logger.debug(
                f"\tPreprocescomplete_preprocessingsing endend: created {self.final_dataset.shape[0]} chunks from {self.dataset.shape[0]} original sessions"
            )
            logger.experiment_logger.debug("\t\tSaving cached file now...")
            logger.save_cache(
                self.final_dataset, f"preprocessed_{self.dataset_name}.parquet"
            )
        else:
            # We still need an id for the original sessions
            logger.experiment_logger.debug("\t\tCreating session id...")
            self.create_session_id()

    def load_label_mapping(self, logger, model_obj=None):
        """Load or create label mappings.
        Args:
            logger (Logger): A logger object for logging and debugging purposes.
            model_obj (AutoModelForTokenClassification, optional): model containing the mapping id2label and label2id. Present only during inference.
        """
        if not model_obj:  # training phase
            labels = list(self.final_dataset["final_labels"].explode().unique())
            self.id2labels = {it: label for it, label in enumerate(labels)}
            self.labels2id = {label: it for it, label in enumerate(labels)}
            logger.save_labels_mapping(self.id2labels, self.labels2id)
        else:
            self.id2labels = model_obj.model.config.id2label
            self.labels2id = model_obj.model.config.label2id

    def complete_preprocessing(self):
        """This functions complete the preprocessing. It strips the strings of some columns + get the indexes of context words."""
        columns_to_strip = (
            ["sessions", "labels"]
            if "labels" in self.final_dataset.columns
            else ["sessions"]
        )
        for column in columns_to_strip:
            self.final_dataset = strip_strings(self.final_dataset, column)
        if self.classified_entity == "statement":
            self.final_dataset["indexes_words_context"] = self.final_dataset.apply(
                lambda row: statement2word(row, add_special_token=True), axis=1
            )
        else:
            self.final_dataset["indexes_words_context"] = self.final_dataset.apply(
                lambda row: statement2word(row, add_special_token=False), axis=1
            )

    def selection_sanity_check(self, count_special_tokens=False):
        """Perform a sanity check on the selection process.
        This method compares the lengths of final inputs and final labels in the dataset to ensure consistency.
        Args:
            count_special_tokens (bool, optional): Whether to count special tokens. Defaults to False.
        """
        if not count_special_tokens:
            len_sessions = self.final_dataset["final_input"].apply(
                lambda final_input: len(final_input)
            )
        else:
            len_sessions = self.final_dataset["final_input"].apply(
                lambda final_input: len([el for el in final_input if el == "[STAT]"])
            )
        len_tokens_labels = self.final_dataset["final_labels"].apply(
            lambda labels: len(labels)
        )
        assert len_sessions.equals(
            len_tokens_labels
        ), "Error: not all entities are labeled!"

    def select_subset_columns(self):
        """Select a subset of columns from the final dataset.
        This method prepares the final input and final labels columns based on the classified entity.
        It also performs a sanity check on the selection process.
        Returns:
            DataFrame: A DataFrame containing selected columns from the final dataset.
        """
        self.final_dataset["statements_special_token"] = self.final_dataset[
            "sessions"
        ].apply(divide_statements, add_special_token=True)
        if self.classified_entity in ["token", "word"]:
            self.final_dataset["final_input"] = self.final_dataset["sessions"].apply(
                lambda session: session.split(" ")
            )
            if "labels" in self.final_dataset.columns:
                self.final_dataset["final_labels"] = self.final_dataset.apply(
                    lambda row: assign_labels2tokens(
                        row.labels, row.statements_special_token
                    ),
                    axis=1,
                )
                self.selection_sanity_check()
        else:
            self.final_dataset["final_input"] = self.final_dataset[
                "statements_special_token"
            ].apply(lambda statements: " ".join(statements).split(" "))
            if "labels" in self.final_dataset.columns:
                self.final_dataset["final_labels"] = self.final_dataset["labels"].apply(
                    lambda labels: labels.split(" -- ")
                )
                self.selection_sanity_check(count_special_tokens=True)
        if "labels" in self.final_dataset.columns:
            return self.final_dataset[
                [
                    "final_input",
                    "final_labels",
                    "session_id",
                    "order_id",
                    "indexes_statements_context",
                    "indexes_words_context",
                ]
            ]
        else:
            return self.final_dataset[
                [
                    "final_input",
                    "session_id",
                    "order_id",
                    "indexes_statements_context",
                    "indexes_words_context",
                ]
            ]
