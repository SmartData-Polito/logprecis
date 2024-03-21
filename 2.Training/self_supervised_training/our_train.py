import sys  # nopep8

sys.path.append("../")  # nopep8

try:
    from core.functions.option_parser import get_training_options
    from core.classes.logger import TrainingExperimentLogger
    from core.classes.dataset_for_self_supervision import UnixDataMaskedLanguage
    from core.classes.mlm_model import MaskedLogModel
    from core.classes.tokenizer import LogPrecisTokenizer
except ImportError:
    sys.path.append("2.Training")
    from core.functions.option_parser import get_training_options
    from core.classes.logger import TrainingExperimentLogger
    from core.classes.dataset_for_self_supervision import UnixDataMaskedLanguage
    from core.classes.mlm_model import MaskedLogModel
    from core.classes.tokenizer import LogPrecisTokenizer


def run(opts):
    ### Experiment Initialization ###
    logger = TrainingExperimentLogger(opts)
    logger.experiment_logger.info(f"Experiment '{opts['identifier']}' started!")
    ### Dataset ###
    logger.experiment_logger.info(f"Loading and processing the dataset...")
    dataset_obj = UnixDataMaskedLanguage(opts)
    dataset_obj.load_dataset(logger)
    ### Tokenizer ###
    logger.experiment_logger.debug(f"Loading tokenizer and tokenizing data...")
    tokenizer_obj = LogPrecisTokenizer(opts)
    tokenizer_obj.tokenize_data(dataset_obj, logger)
    ### Model ###
    logger.experiment_logger.info(f"Load the model...")
    model_obj = MaskedLogModel(opts)
    ### Train ###
    logger.experiment_logger.info(f"Start training...")
    model_obj.train(logger, tokenizer_obj.tokenizer, dataset_obj)
    ### Save best and run on validation ###
    logger.experiment_logger.info(
        "\nTraining ended, obtain best score on validation...\n"
    )
    model_obj.inference(logger)
    logger.experiment_logger.info(f"Save best model...")
    model_obj.save_best(logger)
    ### End ###
    logger.experiment_logger.info(f"End training...")
    logger.end_experiment()


if __name__ == "__main__":
    dict_input = {
        "identifier": "Your_self_supervised_eperiment_ID",
        "task": "self_supervision",
        "entity": "token",
        "seed": 1,
        "log_level": "info",
        "no_cuda": False,
        "output_path": "results/",
        "use_date": False,
        "clean_start": True,
        "input_data": "1.Dataset/Training/Self_supervised/training_set.csv",
        "validation_path": "1.Dataset/Training/Self_supervised/validation_set.csv",
        "eval_size": 0.2,
        "truncation": "default",
        "available_percentage": 1.0,
        "model_name": "microsoft/codebert-base",
        "finetuned_path": "",
        "tokenizer_name": "microsoft/codebert-base",
        "special_token": "[STAT]",
        "max_chunk_length": 256,
        "epochs": 1,
        "lr": 5e-06,
        "patience": 4,
        "observed_val_metric": "loss",
        "batch_size": 16,
        "mlm_probability": 0.15,
        "use_cuda": False,
        "run_name": "Your_self_supervised_eperiment_ID",
        "training_stage": "training",
    }
    run(dict_input)
    # run(get_training_options())