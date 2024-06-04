from core.functions.preprocessing_functions import (
    word_truncation,
    divide_statements,
    expand_labels,
    check_consistency_statement_labels,
    split_session,
    recreate_original_sessions,
    convert2id,
)
from core.functions.utils import get_filename_from_path
import pandas as pd
import numpy as np
from datasets import DatasetDict, Dataset

# To ignore verbose warnings
import datasets  # nopep8
import warnings  # nopep8

warnings.simplefilter(action="ignore", category=FutureWarning)  # nopep8
datasets.utils.logging.set_verbosity(datasets.utils.logging.ERROR)  # nopep8
datasets.utils.logging.enable_progress_bar()  # nopep8


class DataHandler:
    # Generic class to handle data. More specific handlers will inherit from this class
    def __init__(self, opts):
        """All DataHandlers will load a dataset > part of initialization that can be shared
        Args:
            opts (dict): Parameters containing useful information (e.g., location of input data)
        """
        self.experiment_seed = opts["seed"]
        self.dataset_name = get_filename_from_path(opts["input_data"])
        # Useful to save the original dataset in case we want to export predictions
        self.dataset = self.load_data(opts["input_data"])
        self.final_dataset = None
        self.ds = None
        self.tokenized_ds = None

    def load_data(self, input_path):
        """Load data from various file formats.
        This function loads data from CSV, Parquet, or JSON files based on the file extension of the input path.
        Args:
            input_path (str): The path to the input file.
        Returns:
            pandas.DataFrame or None: The loaded data. Returns a pandas DataFrame if the file format is CSV, Parquet, or JSON; otherwise, returns None.
        """
        if input_path.endswith("csv"):
            df = pd.read_csv(input_path, lineterminator="\n")
        elif input_path.endswith("parquet"):
            df = pd.read_parquet(input_path)
        elif input_path.endswith("json"):
            df = pd.read_json(input_path, orient="index").reset_index(drop=True)
        else:
            return None
        return df.dropna()

    def preprocess_data(self):
        # Preprocessing is task specific, each subclass implement its own
        pass

    def truncate_long_words(self, threshold_chars=35):
        """Truncate long words in the dataset.
        This method truncates long words in the 'session' column of the dataset DataFrame to a specified maximum length.
        Args:
            threshold_chars (int, optional): The maximum number of characters allowed for a word. Defaults to 35.
        """
        self.dataset["truncated_session"] = self.dataset["session"].apply(
            word_truncation, max_length=threshold_chars
        )

    def create_session_id(self):
        """Function to create an identifier > Useful when chunking the sessions.
        Notice: since session_id must be an integer, remove the column if it already exists and it's a string.
        """
        if "session_id" in self.dataset.columns:
            # Replace it with numerical index
            self.dataset.drop("session_id", axis=1, inplace=True)
        self.dataset = self.dataset.reset_index().rename(
            {"index": "session_id"}, axis=1
        )

    def create_chunks(self):
        """Create chunks of sessions in the dataset.
        This method creates chunks of sessions based on the specified truncation method and parameters.

        If truncation is set to 'default', sessions are not truncated.
        If truncation is set to 'simple_chunking', sessions are split into chunks of 17 statements.
        Otherwise, sessions are split into chunks of 14 statements with a context of 4 statements.

        NOTICE: Numbers here are hardcoded (grid search over training data) and specific for BASH
        """
        if self.truncation == "default":
            threshold_n_stats, context = np.inf, 0
        elif self.truncation == "simple_chunking":
            threshold_n_stats, context = 17, 0
        else:
            threshold_n_stats, context = 14, 4
        split_data = self.dataset.apply(
            split_session, threshold=threshold_n_stats, context=context, axis=1
        )
        self.final_dataset = pd.concat(split_data.tolist(), ignore_index=True)

    def select_subsample(self):
        """Function that selects a subset of the data. Useful for debugging purposes or to make specific experiments (performance vs number of training samples)."""
        original_shape = self.dataset.shape[0]
        reduced_shape = int(original_shape * self.available_percentage)
        self.dataset = self.dataset.sample(
            frac=1, random_state=self.experiment_seed
        ).iloc[:reduced_shape]

    def remove_faulty_splits(self, faulty_splits):
        """Remove sessions identified as faulty splits (sessions for which truncation failed).
        This method removes sessions identified as faulty splits from the dataset.
        Args:
            faulty_splits (DataFrame): A DataFrame containing information about faulty splits.
        Returns:
            int: The number of session IDs removed from the dataset.
        """
        sessions_to_remove = faulty_splits["self"].values
        session_ids_to_remove = self.dataset[
            self.dataset["truncated_session"].isin(sessions_to_remove)
        ]["session_id"].values
        self.dataset = self.dataset[
            ~self.dataset["session_id"].isin(session_ids_to_remove)
        ]
        self.final_dataset = self.final_dataset[
            ~self.final_dataset["session_id"].isin(session_ids_to_remove)
        ]
        return len(session_ids_to_remove)

    def chunking_sanity_check(self, logger, add_special_token):
        """Perform a sanity check on chunking operations.
        This method compares the truncated sessions in the dataset with the sessions recreated from chunks.
        It identifies faulty splits and removes them from the dataset.
        Args:
            logger (Logger): A logger object for debugging and logging purposes.
        """

        dataset_2_check = self.final_dataset.copy()
        dataset_2_check["statements"] = dataset_2_check.sessions.apply(lambda el: divide_statements(el, add_special_token=add_special_token))
        dataset_2_check["sessions"] = dataset_2_check["statements"].apply(lambda el: " ".join(el))
        grouped_df = (
            dataset_2_check.groupby("session_id")
            .apply(recreate_original_sessions)
            .reset_index()
        )
        grouped_df["recreated_session"] = grouped_df["recreated_session"].apply(lambda el: el.replace("[STAT] ", ""))
        sanity_check = self.dataset[["session_id", "truncated_session"]].merge(
            grouped_df, on="session_id"
        )
        faulty_splits = sanity_check["truncated_session"].compare(
            sanity_check["recreated_session"]
        )
        n_faulty_sessions_ids = self.remove_faulty_splits(faulty_splits)
        logger.debug(f"\t\t\t{n_faulty_sessions_ids} sessions got discarded!")

    def obtain_statements_info(self, logger):
        """Obtain information about statements in the dataset.
        This method divides truncated sessions into statements and adds special tokens if required.
        Additionally, if labels are present in the dataset, it expands the statement labels and checks their consistency.
        Args:
            logger (Logger): A logger object for debugging and logging purposes.
        """
        self.dataset["statements"] = self.dataset["truncated_session"].apply(
            divide_statements, add_special_token=False
        )
        self.dataset["statements_special_token"] = self.dataset[
            "truncated_session"
        ].apply(divide_statements, add_special_token=True)
        logger.experiment_logger.debug("\t\tExpand the labels...")
        if "labels" in self.dataset.columns:
            self.dataset["statement_labels"] = self.dataset["labels"].apply(
                expand_labels
            )
            check_consistency_statement_labels(self.dataset)

    def select_subset_columns(self):
        # Selection is task specific, each subclass implement its own
        pass

    def split_data(self, logger, eval_size):
        """Split the dataset into training, validation, and test sets.
        This method selects meaningful columns from the dataset, removes duplicate session IDs,
        shuffles indexes, and partitions the dataset into training and validation sets based on the specified evaluation size.
        Args:
            logger (Logger): A logger object for debugging and logging purposes.
            eval_size (float): The proportion of the dataset to be allocated for validation.
        Returns:
            tuple: A tuple containing three DataFrames representing the training, validation, and test datasets.
        """
        logger.debug("\tSelect meaningful columns...")
        df_tmp = self.select_subset_columns()
        logger.debug(f"\tFocus on indexes (remove duplicates)...")
        session_ids = df_tmp.drop_duplicates("session_id")
        logger.debug(f"\tShuffle indexes according to seed...")
        session_ids = session_ids.sample(frac=1, random_state=self.experiment_seed)
        train_indexes, valid_indexes = self.get_partitioned_indexes(
            session_ids, eval_size
        )
        train_dataset = df_tmp[df_tmp["session_id"].isin(train_indexes)]
        valid_dataset = df_tmp[df_tmp["session_id"].isin(valid_indexes)]
        # Eventually, we must have 3 partitions > create an empty one for test!
        test_dataset = pd.DataFrame(columns=valid_dataset.columns)
        return train_dataset, valid_dataset, test_dataset

    def get_partitioned_indexes(self, original_ids, eval_size):
        """Partition the session IDs into training and validation sets.
        This method partitions the session IDs into training and validation sets based on the specified evaluation size.
        Args:
            original_ids (DataFrame): The original session IDs.
            eval_size (float): The proportion of the dataset to be allocated for validation.
        Returns:
            tuple: A tuple containing two arrays representing the training and validation session IDs.
        """
        train_size = 1 - eval_size
        train_indexes = original_ids[
            : int(original_ids.shape[0] * train_size)
        ].session_id
        valid_indexes = original_ids[
            int(original_ids.shape[0] * train_size) :
        ].session_id
        return train_indexes.values, valid_indexes.values

    def extract_test(self):
        """Extract the test dataset.
        This method selects meaningful columns from the dataset to create the test dataset.
        The training and validation partitions are empty and only kept for compatibility.
        Returns:
            tuple: A tuple containing three DataFrames representing the training, validation, and test datasets.
        """
        test_dataset = self.select_subset_columns().copy()
        # Since we will gather the predictions for the test datasets, we need the chunks to be sorted according to:
        #   - session_id
        #   - order_id
        test_dataset = test_dataset.sort_values(by=["session_id", "order_id"])
        # The other partitions are empty and only kept for compatibility
        train_dataset = pd.DataFrame(columns=test_dataset.columns)
        valid_dataset = pd.DataFrame(columns=test_dataset.columns)
        return train_dataset, valid_dataset, test_dataset

    def pd_2_hf(self, train_dataset, valid_dataset, test_dataset):
        """Convert pandas DataFrames to Hugging Face Datasets.

        This method converts pandas DataFrames representing training, validation, and test datasets
        into Hugging Face Datasets format.

        Args:
            train_dataset (DataFrame): The training dataset in pandas DataFrame format.
            valid_dataset (DataFrame): The validation dataset in pandas DataFrame format.
            test_dataset (DataFrame): The test dataset in pandas DataFrame format.
        """
        self.ds = DatasetDict(
            {
                "train": Dataset.from_pandas(train_dataset),
                "valid": Dataset.from_pandas(valid_dataset),
                "test": Dataset.from_pandas(test_dataset),
            }
        )
        for key in self.ds.keys():
            if "__index_level_0__" in self.ds[key].column_names:
                self.ds[key] = self.ds[key].remove_columns("__index_level_0__")
        if (
            "final_labels" in train_dataset.columns
            or "final_labels" in test_dataset.columns
        ):
            self.ds = self.ds.map(convert2id, fn_kwargs={"label2id": self.labels2id})

    def set_tokenized_corpus(self, tokenized_corpus):
        """Simple setter for the tokenized corpus
        Args:
            tokenized_corpus (DatasetDict): Tokenized dataset. Contains multiple partitions.
        """
        self.tokenized_ds = tokenized_corpus
