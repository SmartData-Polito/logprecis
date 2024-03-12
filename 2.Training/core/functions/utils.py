import logging
import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_logger(filename, log_level):
    """Get logger.
    This function initializes and configures a logger.
    Args:
        filename (str): The filename to write logs to, if applicable.
        log_level (str): The log level, either 'debug', 'info', or 'warning'.

    Returns:
        Logger: The configured logger object.
    """
    if log_level == "debug":
        log_level = logging.DEBUG
    elif log_level == "info":
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    if log_level in [logging.DEBUG, logging.INFO]:
        logging.basicConfig(format='%(asctime)s - %(message)s')
    else:  # Only writing the into a log file
        logging.basicConfig(filename=filename, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    logger.setLevel(log_level)
    return logger


def get_filename_from_path(path):
    """Get filename from path.
    This function extracts the filename from a given path.
    Args:
        path (str): The file path.
    Returns:
        str: The extracted filename.
    """
    return os.path.basename(os.path.normpath(path)).split(".")[0]


def create_dir(dir_name):
    """Create directory.
    This function creates a directory if it does not exist.
    Args:
        dir_name (str): The directory name.
    """
    os.makedirs(dir_name, exist_ok=True)  # Create if it does not exist


def set_seeds(seed):
    """Set seeds.
    This function sets seeds for random number generators.
    Args:
        seed (int): The seed value.
    """
    # Equivalent of https://huggingface.co/docs/accelerate/v0.1.0/_modules/accelerate/utils.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def postprocess_tensor(tensor):
    """Postprocess tensor.
    This function detaches tensor, moves it to CPU, and converts to numpy array.
    Args:
        tensor (Tensor): The input tensor.
    Returns:
        ndarray: The postprocessed numpy array.
    """
    return tensor.detach().cpu().clone().numpy()


def create_dataframe(values, columns):
    """Create DataFrame.
    This function creates a DataFrame from given values and columns.
    Args:
        values (iterable): The values to populate the DataFrame.
        columns (list): The column names.
    Returns:
        DataFrame: The created DataFrame.
    """
    df_tmp = pd.DataFrame(list(values), columns=columns)
    return df_tmp


def remove_repetitions(list_elements):
    """Remove repetitions from a list.
    This function removes repeated elements from a list.
    Args:
        list_elements (list): The input list.
    Returns:
        list: The list with repeated elements removed.
    """
    prev_el = list_elements[0]
    non_repeated_list = [str(prev_el)]
    for el in list_elements[1:]:
        if prev_el != el:
            non_repeated_list.append(str(el))
            prev_el = el
    return non_repeated_list


def remove_digits_from_labels(labels):
    """Remove digits from labels.
    This function removes digits from labels.
    Args:
        labels (str): The input labels.
    Returns:
        str: The labels with digits removed.
    """
    new_labels = []
    for label in labels.split(" -- "):
        new_labels.append(label.split(" - ")[0])
    return " -- ".join(new_labels)


def recreate_sessions(chunks, aggregating_function):
    """Recreate sessions.
    This function recreates sessions from chunks.
    Args:
        chunks (DataFrame): The chunks containing embeddings.
        aggregating_function (str): The function used for aggregating embeddings.
    Returns:
        Series: The series containing reconstructed sessions.
    """
    chunks = chunks.sort_values(by="Orders")
    # Recreate matrix N_chunks x Hidden_dim
    cls_embeddings = np.array(chunks["Embeddings"].tolist())
    if aggregating_function == "avg":
        session_embedding = cls_embeddings.mean(axis=0)
    elif aggregating_function == "sum":
        session_embedding = cls_embeddings.sum(axis=0)
    elif aggregating_function == "first":
        session_embedding = cls_embeddings[0]
    to_return = {
        "session_embedding": session_embedding,
        "cluster": chunks["Clusters"].iloc[0]
    }
    return pd.Series(to_return, index=['session_embedding', 'cluster'])


def obtain_unpadded_tensors(tensor_to_unpadd, masking_tensor):
    """This function flatten a tensor and obtain all token ids != 100 (padding token)
    Args:
        tensor_to_unpadd (Tensor): Tensor that contains padding. Will be flatten and unpadded.
        masking_tensor (Tensor): Mask saying where is the padding.
    Returns:
        Tensor: unpadded tensor
    """
    flatten_masking_tensor = masking_tensor.reshape(masking_tensor.shape[0] * masking_tensor.shape[1])
    mask = flatten_masking_tensor != - 100
    flatten_tensor_to_unpadd = tensor_to_unpadd.reshape(tensor_to_unpadd.shape[0] * tensor_to_unpadd.shape[1], -1)
    upadded_tensor = flatten_tensor_to_unpadd[mask]
    return upadded_tensor


def expand_and_mask_tensor(tensor_to_expand, expanded_dim, masking_tensor):
    """Function that expands and masks a tensor. 

    Args:
        tensor_to_expand (Tensor): Tensor to expand
        expanded_dim (int): dimension that we want to expand
        masking_tensor (tensor): Masking tensor.

    Returns:
        Tensor: Expanded and masked tensor.
    """
    flatten_masking_tensor = masking_tensor.reshape(masking_tensor.shape[0] * masking_tensor.shape[1])
    mask = flatten_masking_tensor != - 100
    expanded_tensor = tensor_to_expand.expand(tensor_to_expand.shape[0], expanded_dim).reshape(
        tensor_to_expand.shape[0]*expanded_dim)
    return expanded_tensor[mask]


def create_progress_bar(dataloader, epochs):
    """Function that creates a progres bar.
    Args:
        dataloader (DataLoader): required to know how many steps are required to end an epoch
        steps (int): how many training epochs
    Returns:
        _type_: progress bar
    """
    num_update_steps_per_epoch = len(dataloader)
    num_training_steps = epochs * num_update_steps_per_epoch
    return tqdm(range(num_training_steps))
