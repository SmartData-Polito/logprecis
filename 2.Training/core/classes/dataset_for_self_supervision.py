from core.classes.dataset import DataHandler
from core.functions.preprocessing_functions import divide_statements
import pandas as pd
import re


class UnixDataMaskedLanguage(DataHandler):
    # Class to load and prepare data for Masked Language Modelling.
    # Main thing to decide here is whether to add the [STA] tokens or not
    # Adapting a model with [STA] tokens might bring value when solving a statement classification.
    def __init__(self, opts):
        super().__init__(opts)
        self.validation_path = opts["validation_path"]
        self.eval_size = opts["eval_size"]
        self.classified_entity = opts["entity"]
        assert self.classified_entity in [
            "token",
            "statement",
        ], "Error: when adapting a model with self-supervision, this flag is necessary only to choose whether to add the '[STA]' token or not!"

    def load_dataset(self, logger):
        """Load and preprocess the dataset.
        This method loads the dataset and splits it into training and validation (if not done yet)
        Eventually, it converts the Pandas DataFrames into Huggingface Datasets format.
        Args:
            logger (Logger): A logger object for logging and debugging purposes.
        """
        logger.experiment_logger.info(f"Split dataset into train and validation...")
        if self.validation_path != "":  # If specified, use this as default.
            train_dataset = self.dataset.copy()  # keep as is
            valid_dataset = self.load_data(self.validation_path)
        else:
            train_dataset, valid_dataset, _ = self.split_data(
                logger.experiment_logger, self.eval_size
            )
        train_dataset["statements_special_token"] = train_dataset["sessions"].apply(
            lambda el: " ".join(divide_statements(el, add_special_token=True))
        )
        valid_dataset["statements_special_token"] = valid_dataset["sessions"].apply(
            lambda el: " ".join(divide_statements(el, add_special_token=True))
        )
        if self.classified_entity == "token":
            train_dataset.rename({"sessions": "final_input"}, axis=1, inplace=True)
            valid_dataset.rename({"sessions": "final_input"}, axis=1, inplace=True)
        else:  # On this case, also the "[STA]" token will be added > might be useful when later training a model for statement classification
            train_dataset.rename(
                {"statements_special_token": "final_input"}, axis=1, inplace=True
            )
            valid_dataset.rename(
                {"statements_special_token": "final_input"}, axis=1, inplace=True
            )
        test_dataset = pd.DataFrame(
            columns=valid_dataset.columns
        )  # fake initialization
        self.pd_2_hf(train_dataset, valid_dataset, test_dataset)

    def select_subset_columns(self, logger):
        # Required for the dataset function "split_data"
        # No preprocessing required here > return orinal dataset as it is
        df_tmp = self.dataset.copy()
        return df_tmp
