from core.functions.utils import (
    postprocess_tensor,
    create_dataframe,
    remove_repetitions,
    remove_digits_from_labels,
)
from core.functions.preprocessing_functions import (
    groupy_labels_and_predictions,
    recreate_original_sessions,
    divide_statements,
)
from core.functions.utils import (
    obtain_unpadded_tensors,
    expand_and_mask_tensor,
    create_progress_bar,
)
from torch.optim import AdamW
import os
import torch
import evaluate
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    utils,
)
from rouge import Rouge  # During test
import warnings  # nopep8

utils.logging.set_verbosity(utils.logging.ERROR)  # nopep8
utils.logging.enable_progress_bar()  # nopep8


class LogPrecisModel:
    """Class to solve a NER task on the data. Specifically, we want to assign an intent to each session's entity. Notice that an entity can be a token, a word or a statement.
    We can create a LogPrecisModel object both for training and testing.
    """

    def __init__(self, opts, tokenizer_obj, dataset_obj=None):
        ### PARAMETERS FROM ARGPARSER ###
        self.model_name = opts["model_name"]
        self.finetuned_path = opts["finetuned_path"]
        self.classified_entity = opts["entity"]
        self.device = torch.device("cuda:0" if opts["use_cuda"] else "cpu")
        self.training_stage = opts["training_stage"]
        self.load_model(tokenizer_obj, dataset_obj)
        # data collator here is the same both for training and testing
        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer_obj.tokenizer
        )
        self.batch_size = opts["batch_size"]
        self.id2labels, self.label2id = (
            self.model.config.id2label,
            self.model.config.label2id,
        )
        ### Dummy initializations ###
        # Metricsß
        self.metrics, self.rouge = None, None
        # Training paraeters > will be initialized in the `train` function
        self.best_model = None
        self.optimizer = None
        self.epochs = None
        self.patience = None
        self.observed_val_metric = None
        self.train_dataloader, self.eval_dataloader = None, None
        self.progress_bar = None
        # Inference/Test > will be initialized in the `test/inference` function
        self.test_dataloader = None
        self.store_embeddings = None

    def load_model(self, tokenizer_obj, dataset_obj):
        """Function that loads the model.
        The model can be loaded from the filesystem if the user finetuned/domain adapted an Huggingface model or directly from the Huggingface hub.
        Remember that, if solving a statement classification, we must add the <STA> token to the model.
        Args:
            tokenizer_obj (LogPrecisTokenizer): Tokenizer > required in case we have to add the STA token.
            dataset_obj (HoneyDataForClassification): during training, loads the id2label and label2id
        """
        if self.finetuned_path != "":
            # If finetuned path is specified, check whether it comes from a local repo or from huggingface
            path_best_model = (
                os.path.join(self.finetuned_path, "best_model")
                if os.path.exists(os.path.join(self.finetuned_path, "best_model"))
                else self.finetuned_path
            )
            config = (
                os.path.join(path_best_model, "config.json")
                if os.path.exists(path_best_model)
                else None
            )
        else:
            # If no finetuned path is specified, simply keep the model name as reference for huggingface
            path_best_model = self.model_name
            config = None
        if (
            self.training_stage == "training"
        ):  # If it's training, we also have to specify the id2label and label2id
            self.model = AutoModelForTokenClassification.from_pretrained(
                path_best_model,
                id2label=dataset_obj.id2labels,
                label2id=dataset_obj.labels2id,
                config=config,
            ).to(self.device)
            if self.classified_entity == "statement":
                self.model.resize_token_embeddings(len(tokenizer_obj.tokenizer))
        else:  # if inference, mapping were already saved in the model's config (must be)
            self.model = AutoModelForTokenClassification.from_pretrained(
                path_best_model, config=config
            ).to(self.device)

    def create_dataloaders(self, ds_obj):
        """Function that creates the dataloaders.
        If we're training a model, then we create a training and validation set (training with shuffling)
        Otherwise, only test set.
        Args:
            ds_obj (HoneyDataForClassification): dataset object containing the data.
        """
        if self.training_stage == "training":
            train_ds, valid_ds = (
                ds_obj.tokenized_ds["train"],
                ds_obj.tokenized_ds["valid"],
            )
            self.train_dataloader = DataLoader(
                train_ds,
                shuffle=True,
                collate_fn=self.data_collator,
                batch_size=self.batch_size,
            )
            self.eval_dataloader = DataLoader(
                valid_ds, collate_fn=self.data_collator, batch_size=self.batch_size
            )
        else:
            test_ds = ds_obj.tokenized_ds["test"]
            # The prediction's gathering works only because we do not shuffle the chunks here
            # This means that chunks belonging to the same session will come in the correct chunk order
            self.test_dataloader = DataLoader(
                test_ds, collate_fn=self.data_collator, batch_size=self.batch_size
            )

    def create_lr_scheduler(self, patience):
        """Function that creates the LR scheduler.
        Refer to https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        Args:
            patience (int): Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then.
        """
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=patience, verbose=False, min_lr=1e-8
        )

    def load_metric(self):
        """Function that loads the metrics. Metrics are both NLP (rouge) and classification metrics"""
        metrics_list = [
            "accuracy",
            "recall",
            "precision",
            "f1",
        ]  # list of metrics is hardcoded here
        self.metrics = {}
        self.rouge = Rouge()
        for metric in metrics_list:
            self.metrics[metric] = evaluate.load(metric)

    def compute_classification_metrics(
        self, batch_predictions, batch_labels, epoch, logger, partition
    ):
        """This function computes and report the classification metrics on the Tensorboad interface.
        We reoirt both micro and macro scores.
        Args:
            batch_predictions (numpy): Array of predictions
            batch_labels (numpy): Array of labels
            epoch (int): epoch to assign a temporal reference to the scores (x axis of Tensorboard)
            logger (ExperimentLogger): Logger to interact with the Tensorboard interface.
            partition (str): partition of the dataset (e.g., train, validation, test)

        Returns:
            _type_: _description_
        """
        scores = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == "accuracy":
                # For the first iterations we might get verbose warnings otherwise
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    score = metric.compute(
                        references=batch_labels, predictions=batch_predictions
                    )
            elif metric_name == "f1":
                score = metric.compute(
                    references=batch_labels,
                    predictions=batch_predictions,
                    average="macro",
                )
            else:
                score = metric.compute(
                    references=batch_labels,
                    predictions=batch_predictions,
                    average="macro",
                    zero_division=0.0,
                )
            scores[metric_name] = score[metric_name]
            logger.tensorboard_writer.add_scalar(
                f"macro_{metric_name}/{partition}", score[metric_name], epoch
            )

        report = classification_report(
            batch_labels,
            batch_predictions,
            labels=list(self.id2labels.keys()),
            target_names=list(self.id2labels.values()),
            output_dict=True,
            zero_division=0.0,
        )
        if (
            partition == "validation"
        ):  # For test epoch is 1, doesn't make sense to do this plot
            logger.report_micro_scores(report, epoch, partition)
            return scores
        else:
            markdown = pd.DataFrame(report).transpose().to_markdown()
            logger.log_text(
                txt=markdown,
                tag=f"classification_report/{partition}",
                global_step=epoch,
            )
            return None

    def save_best(self, logger):
        """Save the best model once training has ended.
        Args:
            logger (ExperimentLogger): Logger to save the model
        """
        logger.save_best_model(self.best_model)

    def compute_nlp_metrics(self, df_predictions, dataset_obj):
        """Function that computes NLP metrics.
        First, we obtain the groundtruth joining the chunks and removing the context labels
        Then, we start iterating over the groundtruth and predictions:
        - We remove repetitions from each of them to make fair comparisons between entities (see the paper)
        - We compute the scores (fidelity, rouge1 and rouge2)
        Args:
            df_predictions (Pandas): Dataframe containing the predictions
            dataset_obj (HoneyDataForClassification): Dataset object, containing the groundtruth.

        Returns:
            Tuples: lists containing the calculated scores
        """
        original_dataset = dataset_obj.dataset
        # Obtain the groundtruth per session (notice: might be lost if we work at chunk level)
        test_original = original_dataset[
            original_dataset.session_id.isin(df_predictions.Session_ids.values)
        ]
        grouped_predictions = (
            df_predictions.groupby("Session_ids")
            .apply(groupy_labels_and_predictions)
            .reset_index()
            .rename(columns={"Session_ids": "session_id"})
        )
        joined_df = test_original[["session_id", "labels"]].merge(
            grouped_predictions, on="session_id"
        )
        rouge_1_all, rouge_2_all, fidelity = [], [], 0
        for predictions, labels in zip(
            joined_df.reconstructed_predictions, joined_df.labels
        ):
            # remove the repetitions > so that it's fair to compare token-based classification with statement-based
            prediction_string = " -- ".join(
                [self.id2labels[int(el)] for el in remove_repetitions(predictions)]
            ).strip()
            label_string = remove_digits_from_labels(labels)
            try:
                rouge_score = self.rouge.get_scores(prediction_string, label_string)
                rouge_2, rouge_1 = (
                    rouge_score[0]["rouge-2"]["f"],
                    rouge_score[0]["rouge-1"]["f"],
                )
            except RecursionError:
                # Usually, those are cases in which the model makes too many guesses and the prediction becomes too big
                rouge_2, rouge_1 = 0, 0
            rouge_2_all.append(rouge_2)
            rouge_1_all.append(rouge_1)
            if label_string == prediction_string:
                fidelity += 1
        return (
            [np.mean(rouge_1_all)],
            [round(np.mean(rouge_2_all), 5)],
            [round(fidelity / joined_df.shape[0], 5)],
        )

    def train(self, opts, logger, dataset_obj):
        """Training object. It first empty the cuda cache; then it creates the optimizers, dataloaders and lr scheduler; eventually it loads the metric and start the training.
        Args:
            opts (dict): training options (the ones from command line)
            logger (ExperimentLogger): logger to keep track of scores.
            dataset_obj (HoneyDataForClassification): Object containing the datasets.
        """
        torch.cuda.empty_cache()
        self.epochs = opts["epochs"]
        self.observed_val_metric = opts["observed_val_metric"]
        logger.experiment_logger.debug("\tCreate optimizer...")
        self.optimizer = AdamW(self.model.parameters(), lr=opts["lr"])
        logger.experiment_logger.debug("\tCreate dataloaderss...")
        self.create_dataloaders(dataset_obj)
        logger.experiment_logger.debug("\tCreate lr scheduler...")
        self.create_lr_scheduler(opts["patience"])
        logger.experiment_logger.debug("\tLoad metrics...")
        self.load_metric()
        logger.experiment_logger.debug("\tStarting the training loop!")
        self.training_loop(logger, dataset_obj)

    def best_round_validation(self, logger, dataset_obj):
        """Once the train is finished, we compute a final round on the validation set using the best model. This will be useful to compare different design choices (e.g., lr, number of epochs, etc.)

        Args:
            logger (ExperimentLogger): logger to keep track of scores.
            dataset_obj (HoneyDataForClassification): Object containing the datasets.
        """
        if (
            len(self.eval_dataloader) != 0
        ):  # only makes sense if we have a validation set
            device = self.model.device
            del self.model  # Clear the GPU
            self.best_model = self.best_model.to(device)
            losses, predictions, labels, logits, session_ids, orders, _ = (
                self.batch_eval(self.best_model, "validation")
            )
            # Compute classification metrics
            scores = self.compute_classification_metrics(
                predictions, labels, 1, logger, "best_validation"
            )
            df_parquet = create_dataframe(
                values=zip(predictions, labels, logits, session_ids, orders),
                columns=["Predictions", "Labels", "Logits", "Session_ids", "Orders"],
            )
            # Now, compute NLP metrics (Rouge, etc.)
            rouge_1, rouge_2, fidelity = self.compute_nlp_metrics(
                df_parquet, dataset_obj
            )
            df_parquet = create_dataframe(
                values=zip(rouge_1, rouge_2, fidelity),
                columns=["rouge1", "rouge2", "fidelity"],
            )
            markdown = df_parquet.to_markdown()
            logger.log_text(
                txt=markdown, tag=f"best_validation/nlp_scores", global_step=1
            )

    def inference(self, opts, logger, dataset_obj):
        """Inference function, when no labels are available. It computes the model predictions and store them.
        Args:
            opts (dict): inference options (the ones from command line)
            logger (ExperimentLogger): logger to keep track of scores.
            dataset_obj (HoneyDataForClassification): Object containing the datasets.
        """
        logger.experiment_logger.debug("\tCreating inference dataloader...")
        self.create_dataloaders(dataset_obj)
        _, predictions, _, logits, session_ids, orders, embeddings = self.batch_eval(
            self.model, "inference", store_embeddings=opts["store_embeddings"]
        )
        if opts["store_embeddings"]:
            df_parquet = create_dataframe(
                values=zip(predictions, logits, session_ids, orders, embeddings),
                columns=[
                    "Predictions",
                    "Logits",
                    "Session_ids",
                    "Orders",
                    "Embeddings",
                ],
            )
        else:
            df_parquet = create_dataframe(
                values=zip(predictions, logits, session_ids, orders),
                columns=["Predictions", "Logits", "Session_ids", "Orders"],
            )
        logger.log_parquet(
            df_parquet, f"{dataset_obj.dataset_name}_inference_results.parquet"
        )
        if self.classified_entity == "word":
            # Also export reconstructed sessions
            df_reconstructed = self.reconstruct_dataset(
                prediction_df=df_parquet,
                original_df=dataset_obj.final_dataset,
                statement=False,
            )
            logger.experiment_logger.info(
                f"\tReconstructed {df_reconstructed.shape[0]:,}"
            )
            logger.log_parquet(
                df_reconstructed,
                f"{dataset_obj.dataset_name}_reconstructed_predictions_x_sessions.parquet",
            )
        elif self.classified_entity == "statement":
            # Also export reconstructed sessions
            df_reconstructed = self.reconstruct_dataset(
                prediction_df=df_parquet,
                original_df=dataset_obj.final_dataset,
                statement=True,
            )
            logger.experiment_logger.info(
                f"\tReconstructed {df_reconstructed.shape[0]:,}"
            )
            logger.log_parquet(
                df_reconstructed,
                f"{dataset_obj.dataset_name}_reconstructed_predictions_x_sessions.parquet",
            )

    def test(self, opts, logger, dataset_obj):
        """Test function, when when abels are available. It computes the model predictions and test metrics and store them.
        Args:
            opts (dict): inference options (the ones from command line)
            logger (ExperimentLogger): logger to keep track of scores.
            dataset_obj (HoneyDataForClassification): Object containing the datasets.
        """
        logger.experiment_logger.debug("\tCreate inference dataloader...")
        self.create_dataloaders(dataset_obj)
        logger.experiment_logger.debug("\tLoad metrics...")
        self.load_metric()
        # Do inference on test set
        _, predictions, labels, logits, session_ids, orders, embeddings = (
            self.batch_eval(
                self.model, "test", store_embeddings=opts["store_embeddings"]
            )
        )
        if opts["store_embeddings"]:
            df_parquet = create_dataframe(
                values=zip(
                    predictions, labels, logits, session_ids, orders, embeddings
                ),
                columns=[
                    "Predictions",
                    "Labels",
                    "Logits",
                    "Session_ids",
                    "Orders",
                    "Embeddings",
                ],
            )
        else:
            df_parquet = create_dataframe(
                values=zip(predictions, labels, logits, session_ids, orders),
                columns=["Predictions", "Labels", "Logits", "Session_ids", "Orders"],
            )
        logger.log_parquet(
            df_parquet, f"{dataset_obj.dataset_name}_test_results.parquet"
        )
        if self.classified_entity == "word":
            # Also export reconstructed sessions
            df_reconstructed = self.reconstruct_dataset(
                prediction_df=df_parquet,
                original_df=dataset_obj.final_dataset,
                statement=False,
            )
            logger.experiment_logger.info(
                f"\tReconstructed {df_reconstructed.shape[0]:,} sessions"
            )
            logger.log_parquet(
                df_reconstructed,
                f"{dataset_obj.dataset_name}_reconstructed_predictions_x_sessions.parquet",
            )
        elif self.classified_entity == "statement":
            # Also export reconstructed sessions
            df_reconstructed = self.reconstruct_dataset(
                prediction_df=df_parquet,
                original_df=dataset_obj.final_dataset,
                statement=True,
            )
            logger.experiment_logger.info(
                f"\tReconstructed {df_reconstructed.shape[0]:,}"
            )
            logger.log_parquet(
                df_reconstructed,
                f"{dataset_obj.dataset_name}_reconstructed_predictions_x_sessions.parquet",
            )
        # Compute classification metrics
        _ = self.compute_classification_metrics(predictions, labels, 1, logger, "test")
        # Now, compute NLP metrics (Rouge, etc.)
        rouge_1, rouge_2, fidelity = self.compute_nlp_metrics(df_parquet, dataset_obj)
        df_parquet = create_dataframe(
            values=zip(rouge_1, rouge_2, fidelity),
            columns=["rouge1", "rouge2", "fidelity"],
        )
        markdown = df_parquet.to_markdown()
        logger.log_text(txt=markdown, tag=f"nlp_scores/test", global_step=1)

    def reconstruct_dataset(self, prediction_df, original_df, statement=False):
        # First, convert labels into real labels (non IDs)
        prediction_df["sequence_predictions"] = prediction_df["Predictions"].apply(
            lambda prediction_id: self.id2labels[prediction_id]
        )
        if statement:
            # recreate the STAT token in the session (correct way to delete the word ids)
            original_df.loc[:, "session_stat"] = original_df.sessions.apply(
                lambda el: divide_statements(el, add_special_token=True)
            )
            original_df["session_stat"] = original_df["session_stat"].apply(
                lambda char_list: " ".join(char_list)
            )
        # Recreate the original sessions from the chunk now
        grouped_df = (
            original_df.groupby("session_id")
            .apply(lambda row: recreate_original_sessions(row, statement=statement))
            .reset_index()
        )
        # Then, concatenate tokens belonging to the same session according to the chunks' order id
        # Notice: all tokens are in the correct order because:
        # - test set was initially sorted according to session_id and order_id (and never shuffled afterwards)
        # - flatten operation do not modify the order tokens (i.e., tokens are correctly ordered)
        sessions_predictions = (
            prediction_df.groupby("Session_ids")["sequence_predictions"]
            .apply(list)
            .reset_index()
            .rename({"Session_ids": "session_id"}, axis=1)
        )
        # Eventually, join predictions and original sessions
        reconstructed_sessions = sessions_predictions.merge(grouped_df, on="session_id")
        if statement:
            reconstructed_sessions["recreated_session"] = reconstructed_sessions[
                "recreated_session"
            ].str.replace(r"\[STAT\] ", "", regex=True)
        return self.remove_truncated_predictions(reconstructed_sessions, statement)

    def remove_truncated_predictions(self, df_predictions, statement=False):
        df_tmp = df_predictions.copy()
        if statement:
            df_tmp["sequence_statements"] = df_tmp.recreated_session.apply(
                lambda session: divide_statements(session, True)
            )
            n_entities = df_tmp.sequence_statements.apply(lambda session: len(session))
        else:
            df_tmp["sequence_words"] = df_tmp.recreated_session.apply(
                lambda session: session.split()
            )
            n_entities = df_tmp.sequence_words.apply(lambda session: len(session))
        n_predictions = df_tmp.sequence_predictions.apply(lambda el: len(el))
        # returns the indexes in which the number mismatch
        indexes_to_remove = n_entities.compare(n_predictions).index
        final_df = df_tmp.drop(indexes_to_remove)
        return final_df.drop("recreated_session", axis=1)

    def training_loop(self, logger, dataset_obj):
        """Training loop. It performs evaluation 1 time per epoch (after training).

        Args:
            logger (ExperimentLogger): logger to keep track of scores.
            dataset_obj (HoneyDataForClassification): Object containing the datasets for the NLP scores (groundtruth)
        """
        # Initializations
        best_metric, best_epoch = -np.inf, 0
        self.progress_bar = create_progress_bar(
            dataloader=self.train_dataloader, epochs=self.epochs
        )
        for epoch in range(self.epochs):
            ######  TRAIN   ######
            self.model.train()  # Model in training mode
            training_losses = self.batch_training()
            logger.tensorboard_writer.add_scalar(
                "Loss_epoch/train", np.mean(training_losses), epoch
            )
            ######  VALIDATION   ######
            if len(self.eval_dataloader) != 0:
                self.model.eval()  # Model in evaluation mode
                losses, predictions, labels, logits, session_ids, orders, _ = (
                    self.batch_eval(self.model, "validation")
                )
                self.lr_scheduler.step(np.mean(losses))  # Make a scheduler step
                logger.tensorboard_writer.add_scalar(
                    "Loss_epoch/validation", np.mean(losses), epoch
                )
                ######  VALIDATION  METRICS ######
                df_parquet = create_dataframe(
                    values=zip(predictions, labels, logits, session_ids, orders),
                    columns=[
                        "Predictions",
                        "Labels",
                        "Logits",
                        "Session_ids",
                        "Orders",
                    ],
                )
                # Compute classification metrics
                scores = self.compute_classification_metrics(
                    predictions, labels, epoch, logger, "validation"
                )
                # Now, compute NLP metrics (Rouge, etc.)
                rouge_1, rouge_2, fidelity = self.compute_nlp_metrics(
                    df_parquet, dataset_obj
                )
                df_parquet = create_dataframe(
                    values=zip(rouge_1, rouge_2, fidelity),
                    columns=["rouge1", "rouge2", "fidelity"],
                )
                markdown = df_parquet.to_markdown()
                logger.log_text(
                    txt=markdown, tag=f"nlp_scores/validation", global_step=epoch
                )
                # Also save nlp scores
                scores["rouge1"], scores["rouge2"], scores["fidelity"] = (
                    rouge_1[0],
                    rouge_2[0],
                    fidelity[0],
                )
                # Eventually, add the loss
                scores["loss"] = np.mean(losses)
                ######  BEST MODEL UPDATE  ######
                current_metric = (
                    scores[self.observed_val_metric]
                    if self.observed_val_metric != "loss"
                    else -scores[self.observed_val_metric]
                )
                if current_metric > best_metric:
                    best_epoch = 1
                    self.best_model = deepcopy(self.model).to("cpu")
                    best_metric = current_metric
                else:
                    best_epoch = 0
                logger.tensorboard_writer.add_scalar("Is_best", best_epoch, epoch)
            else:  # Always updating the model, since no validation is available
                self.best_model = deepcopy(self.model).to("cpu")

    def batch_training(self):
        """Core of the training: basically, backpropagation mechanism occurring at every epoch.

        Returns:
            numpy: Array containing the batch losses (on cpu)
        """
        batch_losses = []
        for local_batch in self.train_dataloader:
            del local_batch["session_id"], local_batch["order_id"]
            batch = local_batch.to(self.device)
            outputs = self.model(**batch)
            outputs.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            batch_losses.append(outputs.loss.reshape(1))
            self.progress_bar.update(1)
        batch_losses = torch.cat(batch_losses)
        return postprocess_tensor(batch_losses)

    def batch_eval(self, model, partition, store_embeddings=False):
        """Round of validation/test. much richer than batch training, because here we might also want to store embeddings, save more advanced metrics, etc. Function is also called when computing the best round on the validation set.

        Args:
            model (AutoModelForTokenClassification): Model. Can be self.model or self.best_model
            partition (str): Partition. Can be "validation" or "test"
            store_embeddings (bool, optional): Whether to store embeddings or not. Defaults to False.

        Returns:
            Tuple: Arrays of losses, predictions, labels, logits, sessions_ids, order_ids and embeddings.
            Notice: some might be empty (depending on whether that attribute can be calculated or not).
        """
        dataloader = (
            self.eval_dataloader if partition == "validation" else self.test_dataloader
        )
        batch_losses = []
        batch_predictions, batch_labels, batch_logits = [], [], []
        batch_session_ids, batch_chunks_orders = [], []
        batch_embeddings = []
        for local_batch in dataloader:
            with torch.no_grad():
                # Save info that will be useful to recreate the sessions
                batch = local_batch.to(self.device)
                session_id = batch["session_id"].reshape(-1, 1)
                order_id = batch["order_id"].reshape(-1, 1)
                labels = batch["labels"]
                del batch["session_id"], batch["order_id"]
                if partition == "inference":
                    # In this case labels is just an artifact (see tokenizer)
                    del batch["labels"]
                outputs = model(**batch)
            if "loss" in outputs.keys():  # Save model losses if any (not inference)
                batch_losses.append(outputs.loss.reshape(1))
            # Now, store predictions, logits, and labels if any
            input_ids = batch.input_ids
            # from B x |T| to |Non masked T| x 1, where T = Number of tokens
            batch_labels.append(obtain_unpadded_tensors(labels, labels))
            updadded_logists = obtain_unpadded_tensors(outputs.logits, labels)
            batch_predictions.append(updadded_logists.argmax(dim=-1))
            batch_logits.append(updadded_logists.max(dim=-1).values)
            # Notice: we want to extend the session ids and chunk orders to the labelled tokens
            # Session id will be repeated 1) per number of chunks 2) per elements per chunk
            session_ids_per_predicted_token = expand_and_mask_tensor(
                session_id, input_ids.shape[1], labels
            )
            batch_session_ids.append(session_ids_per_predicted_token)
            chunk_order_ids_per_predicted_token = expand_and_mask_tensor(
                order_id, input_ids.shape[1], labels
            )
            batch_chunks_orders.append(chunk_order_ids_per_predicted_token)
            # Notice: since predictions/labels are simply flatten, order of tokens will be maintained!
            if store_embeddings:
                batch_embeddings.append(outputs[0][0])
        # Convert to tensors
        if len(batch_losses) != 0:
            batch_losses = postprocess_tensor(torch.cat(batch_losses))
        batch_predictions, batch_logits = postprocess_tensor(
            torch.cat(batch_predictions)
        ), postprocess_tensor(torch.cat(batch_logits))
        batch_labels = postprocess_tensor(torch.cat(batch_labels))
        batch_session_ids, batch_chunks_orders = postprocess_tensor(
            torch.cat(batch_session_ids)
        ), postprocess_tensor(torch.cat(batch_chunks_orders))
        if store_embeddings:
            batch_embeddings = postprocess_tensor(torch.cat(batch_embeddings))
        return (
            batch_losses,
            batch_predictions,
            batch_labels,
            batch_logits,
            batch_session_ids,
            batch_chunks_orders,
            batch_embeddings,
        )
