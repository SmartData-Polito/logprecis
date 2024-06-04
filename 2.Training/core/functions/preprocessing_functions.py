import re
import pandas as pd
import ast
from datasets import Features, Sequence, Value, ClassLabel


def divide_statements(session, add_special_token, special_token="[STAT]"):
    """Divide a session into statements.
    This function splits a session into statements using specified separators. Optionally,
    it adds a special token at the beginning of each statement.
    Args:
        session (str): The session to be divided into statements.
        add_special_token (bool): Whether to add a special token to each statement.
        special_token (str, optional): The special token to be added. Defaults to "[STAT]".
    Returns:
        list of str: A list of statements.
    """
    statements = re.split(r"(; |\|\|? |&& )", session + " ")
    # concatenate with separators
    if len(statements) != 1:
        statements = [
            "".join(statements[i : i + 2]).strip()
            for i in range(0, len(statements) - 1, 2)
        ]
    else:  # cases in which there is only 1 statement > must end with " ;"
        statements = [statements[0].strip() + " ;"]
    if add_special_token:
        # Add separator
        statements = [f"{special_token} " + el for el in statements]
    return statements


def statement2word(row, add_special_token=False):
    """Convert statements to word indexes.
    This function converts statements into word indexes based on the provided context indexes.
    Args:
        row (DataFrame row): The row containing session information and context indexes.
        add_special_token (bool, optional): whether to add the special tokens. Default to False.
    Returns:
        list of int: A list of word indexes.
    """
    context_indexes = row.indexes_statements_context
    sessions = row.sessions
    # Notice: if we are doing statement classification, it's important to consider some [STA] tokens as context as well!
    statements = divide_statements(sessions, add_special_token=add_special_token)
    indexes_words_context = []
    current_index = 0
    for it, statement in enumerate(statements):
        if it in context_indexes:
            for word_id in range(len(statement.split(" "))):
                indexes_words_context.append(current_index + word_id)
        # So that, while we progress with statements, words_id keep increasing
        current_index += len(statement.split(" "))
    return indexes_words_context


def recreate_original_sessions(rows, statement=False):
    """Recreate the original sessions from sub-sessions.
    This function reconstructs the original sessions from sub-sessions by removing words
    based on the provided word context indexes.
    Args:
        rows (DataFrame): The rows containing sub-sessions and word context indexes.
    Returns:
        Series: A Series containing the recreated original session.
    """
    rows.sort_values(by="order_id", inplace=True)
    if statement:
        sub_sessions = rows["session_stat"]
    else:
        sub_sessions = rows["sessions"]
    indexes_words_context = rows["indexes_words_context"].values
    original_session = []
    for sub_session, word_context in zip(sub_sessions, indexes_words_context):
        words = sub_session.split(" ")
        original_session.append(
            " ".join([words[it] for it in range(len(words)) if it not in word_context])
        )
    return pd.Series({"recreated_session": " ".join(original_session)})


def assign_labels2tokens(labels, statements):
    """Assign labels to tokens based on statements.
    This function assigns labels to tokens based on the provided labels and statements.
    Args:
        labels (str): The labels separated by '--'.
        statements (list of str): The statements to assign labels to.
    Returns:
        list of str: A list of labels assigned to tokens.
    """
    labels = labels.split(" -- ")
    tokens_labels = list()
    for label, statement in zip(labels, statements):
        for word in statement.split(" "):
            if word != "[STAT]":
                tokens_labels.append(label)
    return tokens_labels


def strip_strings(df, column):
    """Strip leading and trailing whitespaces from strings in a DataFrame column.
    This function removes leading and trailing whitespaces from strings in the specified column of the DataFrame.
    Args:
        df (DataFrame): The DataFrame containing the column to be stripped.
        column (str): The name of the column containing strings to be stripped.
    Returns:
        DataFrame: The DataFrame with stripped strings in the specified column.
    """
    df[column] = df[column].apply(lambda session: session.strip())
    return df


def word_truncation(session, max_length):
    """Truncate words in a session to a maximum length.
    This function truncates words in a session to a maximum length specified.
    Args:
        session (str): The session to truncate words in.
        max_length (int): The maximum length allowed for each word.
    Returns:
        str: The session with truncated words.
    """
    return " ".join(
        map(
            lambda word: word[:max_length] if len(word) > max_length else word,
            session.split(" "),
        )
    )


def expand_labels(labels):
    """Expand abbreviated labels to statement labels.
    This function expands abbreviated labels to 1 label per statement based on the provided input.
    Args:
        labels (str): The labels separated by '--' with index information.
    Returns:
        list of str: A list of expanded labels.
    """
    labels = labels.split(" -- ")
    statement_labels = []
    prev_index = 0
    for label in labels:
        label, index = label.split(" - ")
        index = int(index)
        for _ in range(index - prev_index + 1):
            statement_labels.append(label.strip())
        prev_index = index + 1
    return statement_labels


def check_consistency_statement_labels(df):
    """Check the consistency between the number of statements and their corresponding labels.
    This function ensures that the number of statements and their corresponding labels are consistent.
    Args:
        df (DataFrame): The DataFrame containing statements and their corresponding labels.
    """
    number_of_statements = df["statements_special_token"].apply(
        lambda statements: len(
            [el for el in " ".join(statements).split("[STAT]") if el != ""]
        )
    )
    number_of_labels = df["statement_labels"].apply(
        lambda statements_labels: len(statements_labels)
    )
    assert number_of_statements.equals(
        number_of_labels
    ), "Error: not all statements are labeled!"


def split_session(row, threshold, context):
    """Split a session into multiple sessions.

    This function splits a session into multiple sessions based on the specified threshold and context.

    Args:
        row (DataFrame row): The row containing session information.
        threshold (int): The threshold for splitting the session.
        context (int): The context size to be included in each split.

    Returns:
        DataFrame: A DataFrame containing the splitted sessions.
    """
    statements, truncated_session, session_id = (
        row.statements,
        row.truncated_session,
        row.session_id,
    )
    labels = row.statement_labels if "statement_labels" in row else []
    n_statements = len(statements)
    if n_statements > (threshold + context):
        splitted_sessions, splitted_labels, context_indeces = [], [], []
        start, prev_end = 0, 0
        end = threshold + context
        while end <= n_statements:
            if (
                end == n_statements
            ):  # means that the stride has reached exactly last partition of statements
                splitted_sessions.append(" ".join(statements[start:]))
                context_indeces.append([el for el in range(context)])
                splitted_labels.append(" -- ".join(labels[start:]))
            else:
                splitted_sessions.append(" ".join(statements[start:end]))
                if start == 0:
                    context_indeces.append(
                        [
                            el
                            for el in range(
                                len(statements[start:end]) - 1,
                                len(statements[start:end]) - 1 - context,
                                -1,
                            )
                        ]
                    )
                else:
                    context_indeces.append(
                        [el for el in range(context)]
                        + [
                            el
                            for el in range(
                                len(statements[start:end]) - 1,
                                len(statements[start:end]) - 1 - context,
                                -1,
                            )
                        ]
                    )
                splitted_labels.append(" -- ".join(labels[start:end]))
            start += threshold - context
            prev_end = end
            end += threshold - context
        if prev_end != n_statements:
            splitted_sessions.append(" ".join(statements[start:]))
            context_indeces.append([el for el in range(context)])
            splitted_labels.append(" -- ".join(labels[start:]))

        split_data = pd.DataFrame(
            {
                "sessions": splitted_sessions,
                "labels": splitted_labels,
                "session_id": [session_id] * len(splitted_labels),
                "order_id": range(1, len(splitted_labels) + 1),
                "indexes_statements_context": context_indeces,
            }
        )
    else:
        split_data = pd.DataFrame(
            {
                "sessions": [truncated_session],
                "labels": [" -- ".join(labels)],
                "session_id": [session_id],
                "order_id": [1],
                "indexes_statements_context": [[]],
            }
        )
    if len(labels) == 0:
        _ = split_data.pop("labels")
    return split_data


def cast_dataset_columns(ds, labels):
    """Cast dataset columns to appropriate types based on labels.
    This function casts dataset columns to appropriate types based on the provided labels.
    Args:
        ds (DataFrame): The dataset containing columns to be casted.
        labels (list): The list of labels.
    Returns:
        DataFrame: The dataset with casted columns.
    """
    if "final_labels" in ds.columns():
        ds = label_casting(ds, labels)
    else:
        ds = inference_casting(ds)
    return ds


def label_casting(ds, labels):
    """Cast dataset columns to appropriate types for labeled data.
    This function casts dataset columns to appropriate types for labeled data based on the provided labels.
    Args:
        ds (Dataset): The dataset containing columns to be casted.
        labels (list): The list of labels.
    Returns:
        Dataset: The dataset with casted columns for labeled data.
    """
    features = Features(
        {
            "final_input": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
            "final_labels": Sequence(
                feature=ClassLabel(num_classes=len(labels), names=labels)
            ),
            "indexes_statements_context": Sequence(
                feature=Value(dtype="int32", id=None), length=-1, id=None
            ),
            "indexes_words_context": Sequence(
                feature=Value(dtype="int32", id=None), length=-1, id=None
            ),
            "session_id": Value(dtype="int32", id=None),
            "order_id": Value(dtype="int32", id=None),
        }
    )
    return ds.map(process_with_labels, features=features)


def process_with_labels(ex):
    """Process dataset examples with labels.
    This function processes dataset examples with labels.
    Args:
        ex (dict): The dataset example.
    Returns:
        dict: The processed dataset example.
    """
    return {
        "final_input": ast.literal_eval(ex["final_input"]),
        "final_labels": ast.literal_eval(ex["final_labels"]),
        "session_id": int(ex["session_id"]),
        "order_id": int(ex["order_id"]),
        "indexes_statements_context": ast.literal_eval(
            ex["indexes_statements_context"]
        ),
        "indexes_words_context": ast.literal_eval(ex["indexes_words_context"]),
    }


def inference_casting(ds):
    """Cast dataset columns to appropriate types for inference.
    This function casts dataset columns to appropriate types for inference.
    Args:
        ds (Dataset): The dataset containing columns to be casted.
    Returns:
        Dataset: The dataset with casted columns for inference.
    """
    features = Features(
        {
            "final_input": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
            "indexes_statements_context": Sequence(
                feature=Value(dtype="int32", id=None), length=-1, id=None
            ),
            "indexes_words_context": Sequence(
                feature=Value(dtype="int32", id=None), length=-1, id=None
            ),
            "session_id": Value(dtype="int32", id=None),
            "order_id": Value(dtype="int32", id=None),
        }
    )
    return ds.map(process_without_labels, features=features)


def process_without_labels(ex):
    """Process dataset examples without labels.
    This function processes dataset examples without labels.
    Args:
        ex (dict): The dataset example.
    Returns:
        dict: The processed dataset example.
    """
    return {
        "final_input": ast.literal_eval(ex["final_input"]),
        "session_id": int(ex["session_id"]),
        "order_id": int(ex["order_id"]),
        "indexes_statements_context": ast.literal_eval(
            ex["indexes_statements_context"]
        ),
        "indexes_words_context": ast.literal_eval(ex["indexes_words_context"]),
    }


def convert2id(sample, label2id):
    """Convert labels to their corresponding ids.
    This function converts labels to their corresponding ids using the provided label-to-id mapping.
    Args:
        sample (dict): The sample containing labels to be converted.
        label2id (dict): The dictionary mapping labels to their corresponding ids.
    Returns:
        dict: The sample with labels converted to ids.
    """
    for it in range(len(sample["final_labels"])):
        sample["final_labels"][it] = label2id[sample["final_labels"][it]]
    return sample


def groupy_labels_and_predictions(group):
    """Group labels and predictions.
    This function groups labels and predictions.
    Args:
        group (DataFrame): The DataFrame containing labels and predictions.
    Returns:
        Series: The series with reconstructed predictions.
    """
    ordered_group = group.sort_values(by="Orders")
    predictions = list(ordered_group["Predictions"].values)
    return pd.Series({"reconstructed_predictions": predictions})
