import time
import argparse
from torch.cuda import is_available


def get_training_options(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters to train a Language Model for entity classification"
    )

    # Experiment details
    parser.add_argument(
        "--identifier", type=str, required=True, help="Special identifier."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["self_supervision", "entity_classification"],
        help="Task solved by the experiment.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        choices=["token", "word", "statement"],
        default="token",
        help="Entity to classify. Best performance obtained with token classification.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Experiment seed. Influences the partitions divisions and the model's initialization.",
        default=1,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning"],
        help="Level of logs. If debug and info, script will have mode prints.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Whether to enforce the script not to use GPU even if available.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Output path where to save the run's output.",
    )
    parser.add_argument(
        "--use_date",
        action="store_true",
        help="Flag indicating whether to include the date info in the experiment name.",
    )
    parser.add_argument(
        "--clean_start",
        action="store_true",
        help="Flag indicating whether, given the same experiment ID, we want to clean previous logs, results and cache before starting.",
    )

    # Data
    parser.add_argument(
        "--input_data",
        required=True,
        type=str,
        help="Path to the input data. Supposed to be a parquet or csv file. See more in the README.",
    )
    parser.add_argument(
        "--validation_path",
        type=str,
        default="",
        help="Path to the validation data. Alternative to obtaining the validation set from the training.",
    )
    parser.add_argument(
        "--eval_size", type=float, default=0.2, help="Size of the evaluation partition."
    )
    parser.add_argument(
        "--truncation",
        type=str,
        choices=["default", "simple_chunking", "context_chunking"],
        default="context_chunking",
        help="How data are going to be pre-processed chunkin-wise.",
    )
    parser.add_argument(
        "--available_percentage",
        type=float,
        default=1,
        help="Percentage of available data. Useful for ablation studies.",
    )

    # Model
    parser.add_argument("--model_name", type=str, required=True, help="Chosen model.")
    parser.add_argument(
        "--finetuned_path",
        type=str,
        default="",
        help="Path to the domain-adapted finetuned model. Remember that the path must point to a folder that contains a subfolder 'best_model' (run a simple training for an example).",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Chosen tokenizer. \
                        Can be a simple name (e.g., bert-uncased) or the path to the finetuned tokenizer.",
    )
    parser.add_argument(
        "--special_token",
        type=str,
        default="[STAT]",
        help="Special token to classify in case of statement classification",
    )
    parser.add_argument(
        "--max_chunk_length",
        type=int,
        default=512,
        help="Maximum number of tokens before truncation.",
    )

    # Hyper-parameters
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        help="Number of epochs before stopping training.",
    )
    parser.add_argument(
        "--lr",
        default=5e-6,
        type=float,
        help="Number of epochs before stopping training.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="How many epochs to wait before reducing on plateau. Default to 4.",
    )
    parser.add_argument(
        "--adaptation",
        type=str,
        default="concatenated",
        choices=["concatenated", "single_session"],
        help="Type of adaptation used in the tokenizer. Default to concatenated",
    )
    parser.add_argument(
        "--observed_val_metric",
        type=str,
        default="loss",
        help="Metric used to define which is the best model during validation.",
        choices=[
            "loss",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "rouge1",
            "rouge2",
            "fidelity",
        ],
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="How many samples per batch."
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of masking a token.",
    )

    opts = parser.parse_args(args)
    # We use GPU if any available
    opts.use_cuda = is_available() and not opts.no_cuda
    # Run identifier
    opts.run_name = (
        f'{opts.identifier}_{time.strftime("%Y%m%dT%H%M%S")}'
        if opts.use_date
        else opts.identifier
    )
    # Set training stage
    opts.training_stage = "training"
    return vars(opts)


def get_inference_options(args=None):
    """Functions that gets the parameters passed from the .sh script.
    Args:
        args (_type_): Parameters from the .sh script.
    Returns:
        Dictionary: Parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters to perform inference with a Language Model for entity classification"
    )

    # Experiment details
    parser.add_argument(
        "--entity",
        type=str,
        choices=["token", "word", "statement"],
        default="token",
        help="Entity to classify. Best performance obtained with token classification.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["self_supervision", "entity_classification"],
        help="Task solved by the experiment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Experiment seed. Influences the partitions divisions and the model's initialization.",
        default=1,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="debug",
        choices=["debug", "info", "warning"],
        help="Level of logs. If debug and info, script will have mode prints.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Whether to enforce the script not to use GPU even if available.",
    )
    parser.add_argument(
        "--store_embeddings",
        action="store_true",
        help="Whether to store the embeddings of the processed sentences.",
    )
    parser.add_argument(
        "--identifier", type=str, default="inference", help="Special identifier."
    )

    # Data
    parser.add_argument(
        "--input_data",
        required=True,
        help="Path to the input data. Supposed to be a parquet or csv file. See more in the README.",
    )
    parser.add_argument(
        "--truncation",
        type=str,
        choices=["default", "simple_chunking", "context_chunking"],
        default="context_chunking",
        help="How data are going to be pre-processed chunkin-wise.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Output path where to save the run's output. Used only if using a finetuned model from Huggingface for inference.",
    )

    # Model
    parser.add_argument("--model_name", type=str, required=True, help="Chosen model.")
    parser.add_argument(
        "--finetuned_path",
        type=str,
        required=True,
        help="Finetuned Path. \
                        Must be the path to the finetuned model.",
    )
    parser.add_argument(
        "--adaptation",
        type=str,
        default="concatenated",
        choices=["concatenated", "single_session"],
        help="Type of adaptation used in the tokenizer. Default to concatenated",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Chosen tokenizer. \
                        Can be a simple name (e.g., bert-uncased) or the path to the finetuned tokenizer.",
    )
    parser.add_argument(
        "--special_token",
        type=str,
        default="[STAT]",
        help="Special token to classify in case of statement classification",
    )

    # Hyper-parameters
    parser.add_argument(
        "--batch_size", type=int, default=16, help="How many samples per batch."
    )
    parser.add_argument(
        "--max_chunk_length",
        type=int,
        default=512,
        help="Maximum number of tokens before truncation.",
    )

    opts = parser.parse_args(args)
    # We use GPU if any available
    opts.use_cuda = is_available() and not opts.no_cuda
    # Set training stage
    opts.training_stage = "inference"
    opts.run_name = opts.identifier
    # Option only for training > set as placeholders
    opts.available_percentage = 1
    opts.eval_size = 0
    return vars(opts)
