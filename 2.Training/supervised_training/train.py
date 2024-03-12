import sys  # nopep8
sys.path.append("../")  # nopep8
from core.functions.option_parser import get_training_options
from core.classes.logger import ClassificationTrainingLogger
from core.classes.dataset_for_classification import HoneyDataForClassification
from core.classes.classification_model import LogPrecisModel
from core.classes.tokenizer import LogPrecisTokenizer


def run(opts):
    ### Experiment Initialization ###
    logger = ClassificationTrainingLogger(opts)
    logger.experiment_logger.info(f"Experiment '{opts['identifier']}' started!")
    ### Dataset ###
    logger.experiment_logger.info(f"Loading and processing the dataset...")
    dataset_obj = HoneyDataForClassification(opts)
    dataset_obj.load_dataset(logger)
    ### Tokenizer ###
    logger.experiment_logger.debug(f"Loading tokenizer and tokenizing data...")
    tokenizer_obj = LogPrecisTokenizer(opts)
    tokenizer_obj.tokenize_data(dataset_obj, logger)
    ### Model ###
    logger.experiment_logger.info(f"Load the model...")
    model_obj = LogPrecisModel(opts, tokenizer_obj, dataset_obj)
    ### Train ###
    logger.experiment_logger.info(f"Start training...")
    model_obj.train(opts, logger, dataset_obj)
    ### Save best and run on validation ###
    logger.experiment_logger.info(f"Save best model...")
    model_obj.save_best(logger)
    logger.experiment_logger.info(f"Best model on validation...")
    model_obj.best_round_validation(logger, dataset_obj)
    ### End ###
    logger.experiment_logger.info(f"End training...")
    logger.end_experiment()


if __name__ == "__main__":
    run(get_training_options())
