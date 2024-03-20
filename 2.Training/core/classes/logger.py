import shutil
import os
import json
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from core.functions.utils import get_logger, create_dir, set_seeds
from warnings import warn


class ExperimentLogger:
    """Basic class to keep track of the experiment logs. All following Loggers will inherit these classes."""

    def __init__(self, opts):
        """Initializer of loggers.
        Loggers will both display log messages and save and the metrics.
        Args:
            opts (Dictionary): Experiment parameters. Contains important information such as the run name.
        """
        # Initializations
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        log_file = os.path.join(
            self.log_dir, f"experiment_logs_{opts['training_stage']}.log"
        )
        log_level = opts["log_level"]
        # Create log dir
        create_dir(self.log_dir)
        # Create new logging objects (one for console outputs, other for metrics, etc.)
        self.experiment_logger = get_logger(filename=log_file, log_level=log_level)
        self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)

    def add_other_params(self, dict_attributes):
        """Adds additional parameters to the object.
        Args:
            dict_attributes (dict): A dictionary containing the parameter names as keys and their corresponding values.
        """
        for key, value in dict_attributes.items():
            setattr(self, key, value)

    def get_params(self):
        """Gets the parameters of the object.
        Returns:
            dict_items: A view object providing a key-value pair of the object's parameters.
        """
        return self.__dict__.items()

    def clean_old_results(self, clean_start):
        """Cleans up old experiment results if specified.
        Args:
            clean_start (bool): Indicates whether to perform the cleanup or not.
        """
        if clean_start:
            try:
                shutil.rmtree(self.experiment_dir)
            except OSError as e:
                # Folder does not exist (first time that experiment id was used)
                pass

    def load_cached_data(self, filename):
        """Loads cached data from the cache directory.
        Args:
            filename (str): The name of the file to load.
        Returns:
            DataFrame: The cached data as a Pandas DataFrame, or an empty DataFrame if the file does not exist.
        """
        cache_dir = os.path.join(self.experiment_dir, "cache")
        if os.path.isfile(os.path.join(cache_dir, filename)):
            df = pd.read_parquet(os.path.join(cache_dir, filename))
        else:
            df = pd.DataFrame()
        return df

    def save_cache(self, df, filename):
        """Saves the DataFrame to the cache directory as a parquet file.
        Args:
            df (DataFrame): The DataFrame to save.
            filename (str): The name of the file to save.
        """
        cache_dir = os.path.join(self.experiment_dir, "cache")
        create_dir(cache_dir)
        df.to_parquet(os.path.join(cache_dir, filename), index=False)

    def log_text(self, tag, txt, global_step):
        """Logs text data to TensorBoard.
        Args:
            tag (str): The tag associated with the text data.
            txt (str): The text data to be logged.
            global_step (int): The global step at which the text data is logged.
        """
        self.tensorboard_writer.add_text(
            tag=tag, text_string=txt, global_step=global_step
        )

    def log_image(self, tag, img, global_step):
        """Logs an image to TensorBoard.
        Args:
            tag (str): The tag associated with the image.
            img (matplotlib.figure.Figure): The image to be logged.
            global_step (int): The global step at which the image is logged.
        """
        self.tensorboard_writer.add_figure(tag=tag, figure=img, global_step=global_step)

    def log_parquet(self, df, filename):
        """Logs a DataFrame to a Parquet file.
        Args:
            df (pd.DataFrame): The DataFrame to be logged.
            filename (str): The filename of the Parquet file.
        """
        output_file = os.path.join(self.log_dir, filename)
        df.to_parquet(output_file, index=False)

    def end_experiment(self):
        """Closes the TensorBoard writer, ending the experiment."""
        self.tensorboard_writer.close()

    def report_micro_scores(self, report, epoch, partition):
        """Reports micro scores (precision, recall, F1-score) for each class in the classification report.
        Args:
            report (dict): Classification report containing precision, recall, and F1-score for each class.
            epoch (int): Current epoch number.
            partition (str): Name of the data partition (e.g., train, validation, test).
        """
        for class_name in report.keys():
            # Metrics here are hardcoded
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                self.tensorboard_writer.add_scalar(
                    f"micro_precision/{partition}/{class_name}",
                    report[class_name]["precision"],
                    epoch,
                )
                self.tensorboard_writer.add_scalar(
                    f"micro_recall/{partition}/{class_name}",
                    report[class_name]["recall"],
                    epoch,
                )
                self.tensorboard_writer.add_scalar(
                    f"micro_fscore/{partition}/{class_name}",
                    report[class_name]["f1-score"],
                    epoch,
                )


class TrainingExperimentLogger(ExperimentLogger):
    """Object for generic training (can be self-supervised or supervised). With respect to the base object, it saves a configuration file (with the parameters used for training) and saves the best model once trained."""

    def __init__(self, opts):
        model_name = opts["model_name"].replace("/", "_")
        self.experiment_dir = os.path.join(
            opts["output_path"],
            opts["task"],
            opts["entity"],
            model_name,
            opts["run_name"],
            f"seed_{opts['seed']}",
        )
        # Clean old results if required
        self.clean_old_results(opts["clean_start"])
        super().__init__(opts)
        self.add_other_params(opts)

    def experiment_initialization(self, opts):
        self.save_training_config()
        set_seeds(opts["seed"])

    def save_best_model(self, model):
        path_best_model = os.path.join(self.experiment_dir, "best_model")
        model.save_pretrained(path_best_model)

    def save_training_config(self):
        """Saves the training parameters to a JSON file.
        This method retrieves the training parameters using the `get_params()` method and removes the `experiment_logger` and `tensorboard_writer` entries from the parameters before saving them to a JSON file named "training_parameters.json" in the experiment directory.
        Note:
            This method assumes that the `get_params()` method returns a dictionary containing all the training parameters.
        """
        params = dict(self.get_params())
        del params["experiment_logger"]
        del params["tensorboard_writer"]
        with open(
            os.path.join(self.experiment_dir, "training_parameters.json"), "w+"
        ) as f:
            json.dump(params, f, indent=4)


class InferenceExperimentLogger(ExperimentLogger):
    """Object for generic inference (can be self-supervised or supervised). With respect to the base object, it loads  the configuration file used for training."""

    def __init__(self, opts):
        self.experiment_dir = opts["finetuned_path"]
        # Before instantiating the father object, check if experiment dir is a valid path
        self.training_parameters = self.load_training_config(opts)
        super().__init__(opts)
        self.check_consistency(opts)

    def experiment_initialization(self, opts):
        set_seeds(opts["seed"])

    def load_training_config(self, opts):
        """Loads the training parameters from a JSON file.
        This method attempts to load the training parameters from a JSON file named "training_parameters.json" in the experiment directory. It checks if the file exists, and if it does, it reads the parameters from the file and returns them.
        Returns:
            dict: A dictionary containing the loaded training parameters.
        Raises:
            AssertionError: If the "training_parameters.json" file does not exist in the experiment directory.
        """
        config_path = os.path.join(self.experiment_dir, "training_parameters.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                training_parameters = json.load(f)
        else:
            # no trained model is available at that finetuned path
            model_name = opts["model_name"].replace("/", "_")
            self.experiment_dir = os.path.join(
                opts["output_path"],
                opts["task"],
                opts["entity"],
                model_name,
                opts["run_name"],
                f"seed_{opts['seed']}",
            )
            training_parameters = {}
        return training_parameters


class ClassificationTrainingLogger(TrainingExperimentLogger):
    """Object for classification training. With respect to the base object, it export a json containing the mapping from labels to ids and viceversa (useful for testing)."""

    def __init__(self, opts):
        super().__init__(opts)
        self.experiment_initialization(opts)

    def save_labels_mapping(self, id2labels, labels2id):
        """Saves the mapping between label indices and label names to JSON files.
        This method saves the provided mappings between label indices and label names to separate JSON files in the "labels_mapping" directory within the experiment directory. It creates the directory if it doesn't exist. The mappings are stored in two JSON files: "id2labels.json" and "labels2id.json".
        Args:
            id2labels (dict): A dictionary mapping label indices to label names.
            labels2id (dict): A dictionary mapping label names to label indices.
        """
        labels_mapping_dir = os.path.join(self.experiment_dir, "labels_mapping")
        create_dir(labels_mapping_dir)
        with open(os.path.join(labels_mapping_dir, "id2labels.json"), "w+") as f:
            json.dump(id2labels, f, indent=4)
        with open(os.path.join(labels_mapping_dir, "labels2id.json"), "w+") as f:
            json.dump(labels2id, f, indent=4)


class ClassificationInferenceLogger(InferenceExperimentLogger):
    """Object for classification inference. With respect to the base object, it loads a json containing the mapping from labels to ids and viceversa (useful for testing). Also, it checks whether the inference requests are consistent with the trained model."""

    def __init__(self, opts):
        super().__init__(opts)
        self.experiment_initialization(opts)

    def load_labels_mapping(self):
        """Loads the mapping between label indices and label names from JSON files.
        This method loads the mappings between label indices and label names from separate JSON files in the "labels_mapping" directory within the experiment directory. It assumes that the mappings are stored in two JSON files: "id2labels.json" and "labels2id.json". It converts the loaded keys to integers in the case of id2labels and to integers in the case of labels2id.
        Returns:
            dict, dict: A tuple containing the loaded id2labels mapping (mapping from label indices to label names) and the loaded labels2id mapping (mapping from label names to label indices).
        """
        labels_mapping_dir = os.path.join(self.experiment_dir, "labels_mapping")
        if os.path.isfile(os.path.join(labels_mapping_dir, "id2labels.json")):
            # Files exists because we're loading a locally finetuned model
            with open(os.path.join(labels_mapping_dir, "id2labels.json")) as f:
                id2labels = json.load(f)
            with open(os.path.join(labels_mapping_dir, "labels2id.json")) as f:
                labels2id = json.load(f)
            # Remember: id must be int
            id2labels = {int(key): value for (key, value) in id2labels.items()}
            labels2id = {key: int(value) for (key, value) in labels2id.items()}
            return id2labels, labels2id
        else:
            # We're using a finetuned model from Huggingface Hub and hence those files do not exist
            return {}, {}

    def check_consistency(self, opts):
        """Checks the consistency between training parameters and inference options.
        This method compares the training parameters used to train the model with the inference options provided. It specifically checks whether the truncation strategy and the entity type used for training are consistent with the inference options. If there are inconsistencies, it raises AssertionError with corresponding error messages.
        Args:
            opts (dict): Inference options to be checked for consistency.
        Raises:
            AssertionError: If the truncation strategy or entity type used for training is inconsistent with the inference options.
        """
        if len(self.training_parameters.keys()) != 0:
            training_truncation = self.training_parameters["truncation"]
            inference_truncation = opts["truncation"]
            assert (
                training_truncation == inference_truncation
            ), "Error: trained model was using a corpus truncated differently"
            training_entity = self.training_parameters["entity"]
            inference_entity = opts["entity"]
            assert (
                training_entity == inference_entity
            ), "Error: model was trained using different entities"
        else:
            self.experiment_logger.warn(
                f"You are loading a model from Huggingface Hub: to obtain the best performance, make sure that the truncation policy `truncation` and classification entity `entity` on your inference data fit the ones used to finetuned the loaded model. Ignore this message if you are trying to classify `entity='word'` with a model tuned for token classification."
            )
