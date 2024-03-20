import sys  # nopep8

sys.path.append("../")  # nopep8
from core.functions.option_parser import get_inference_options
from core.classes.logger import ClassificationInferenceLogger
from core.classes.dataset_for_classification import HoneyDataForClassification
from core.classes.classification_model import LogPrecisModel
from core.classes.tokenizer import LogPrecisTokenizer


def run(opts):
    ### Experiment Initialization ###
    logger = ClassificationInferenceLogger(opts)
    logger.experiment_logger.info(
        f"Inference/Testing with model '{opts['model_name']}' started!"
    )
    ### Tokenizer ###
    logger.experiment_logger.debug(f"Loading tokenizer...")
    tokenizer_obj = LogPrecisTokenizer(opts)
    ### Model ###
    logger.experiment_logger.info(f"Load the model...")
    model_obj = LogPrecisModel(opts, tokenizer_obj)
    ### Dataset ###
    logger.experiment_logger.info(f"Loading the dataset...")
    dataset_obj = HoneyDataForClassification(opts)
    dataset_obj.load_dataset(logger, model_obj)
    logger.experiment_logger.info(f"Tokenizing data...")
    tokenizer_obj.tokenize_data(dataset_obj, logger)
    ### Inference or Test ###
    if "final_labels" in dataset_obj.ds["test"].column_names:
        logger.experiment_logger.info(f"Start testing...")
        model_obj.test(opts, logger, dataset_obj)
    else:
        logger.experiment_logger.info(f"Start inference...")
        model_obj.inference(opts, logger, dataset_obj)
    ### End ###
    logger.experiment_logger.info(f"End inference/testing...")
    logger.end_experiment()


if __name__ == "__main__":
    run(get_inference_options())
