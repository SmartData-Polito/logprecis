def align_labels_with_statements(
    labels, tokens_ids, context_statements, special_token_id
):
    """Function that puts to -100 (CrossEntropy will ignore) all labels not associated to NON-CONTEXT statement tokens.
    Basically, useful to only backpropagate the loss related to the STAs tokens.
    Args:
        labels (list): List of labels. Each token (also non-context ones) has 1 associated.
        tokens_ids (list): Token ids from the tokenizer.
        context_statements (list): List of ids corresponding to context statements. Incremental from 0.
        special_token_id (int): Special token id assigned by the tokenizer to the STA token.

    Returns:
        list: list of new labels (only related to NON-CONTEXT statement tokens)
    """
    new_labels = []
    label_it = 0
    for token_id in tokens_ids:
        if token_id == special_token_id:
            if label_it not in context_statements:
                # '[STA]' token not in context
                new_labels.append(labels[label_it])
            else:
                # '[STA]' token in context
                new_labels.append(-100)
            label_it += 1
        else:
            # Anything else
            new_labels.append(-100)
    return new_labels


def align_labels_with_tokens(labels, word_ids, indexes_context_words, loss_per_token):
    """Function to put to -100 all labels non associated to meaningful tokens. Those are:
        1) first non-context tokens per word (exluding special tokens), if loss_per_token = False
        2)  all non-context tokens (exluding special tokens), if loss_per_token = True

    Args:
        labels (list): List of labels. Each token (also non-context ones) has 1 associated.
        word_ids (list): Word ids from the tokenizer. Each token has one associated.
        indexes_context_words (list): List of ids corresponding to context words.
        loss_per_token (bool): if True, we solve a token classification problem.

    Returns:
        list: list of new labels (only related to meaningful tokens)
    """
    new_labels = []
    previous_word = None
    for word_id in word_ids:
        if word_id is None:
            # Special token
            new_labels.append(-100)
        elif word_id in indexes_context_words:
            # Part of context words I don't want to label
            new_labels.append(-100)
        elif word_id != previous_word:
            # Start of a new word!
            previous_word = word_id
            label = labels[word_id]
            new_labels.append(label)
        else:
            if loss_per_token:
                # Same word as previous token
                label = labels[word_id]
                new_labels.append(label)
            else:
                new_labels.append(-100)
    return new_labels


def mask_token_context_predictions(context_words, word_ids, input_ids):
    """Function used at inference (when no labels are available). It basically tells which are the tokens for which we want a prediction. It's the equivalent of `align_labels_with_tokens`, but always considers only the 1st token per word (classified entity = word).

    Args:
        context_words (list): List of ids corresponding to context words.
        word_ids (list): Word ids from the tokenizer. Each token has one associated.
        input_ids (list): Token ids from the tokenizer.

    Returns:
        list: token ids we want to a prediction for.
    """
    tokens_to_keep = []
    previous_word = None
    for word_id, input_id in zip(word_ids, input_ids):
        if word_id is None:
            # Special token
            tokens_to_keep.append(-100)
        elif word_id in context_words:
            # Part of context words I don't want to label
            tokens_to_keep.append(-100)
        elif word_id == previous_word:
            # Same word as previous token
            tokens_to_keep.append(-100)
        else:
            # Start of a new word!
            previous_word = word_id
            tokens_to_keep.append(input_id)
    return tokens_to_keep


def mask_statement_context_predictions(
    context_words, word_ids, input_ids, special_token_id
):
    """Function used at inference (when no labels are available). It basically tells which are the statements for which we want a prediction. It's the equivalent of `align_labels_with_statements`.

    Args:
        context_words (list): List of ids corresponding to context words.
        word_ids (list): Word ids from the tokenizer. Each token has one associated.
        input_ids (list): Token ids from the tokenizer.
        special_token_id (int): Special token id assigned by the tokenizer to the STA token.

    Returns:
        list: token ids we want to a prediction for.
    """
    tokens_to_keep = []
    for word_id, input_id in zip(word_ids, input_ids):
        if input_id == special_token_id:
            if word_id in context_words:
                # Part of context words I don't want to label
                tokens_to_keep.append(-100)
            else:
                tokens_to_keep.append(input_id)
        else:
            # Anything else
            tokens_to_keep.append(-100)
    return tokens_to_keep
