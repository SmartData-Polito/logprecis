def align_labels_with_statements(labels, tokens_ids, context_statements, special_token_id):
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


def mask_statement_context_predictions(context_words, word_ids, input_ids, special_token_id):
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
