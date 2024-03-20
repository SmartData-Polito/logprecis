from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
    utils,
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch import no_grad, cat
import torch
from torch.cuda import empty_cache
import math
import os
import numpy as np
from copy import deepcopy
from core.functions.utils import postprocess_tensor
import warnings  # nopep8

utils.logging.set_verbosity(utils.logging.ERROR)  # nopep8
utils.logging.enable_progress_bar()  # nopep8


class MaskedLogModel:
    """Class to train a model with self-supervision (masked language modelling)."""

    def __init__(self, opts):
        ### PARAMETERS FROM ARGPARSER ###
        self.model_name = opts["model_name"]
        self.finetuned_path = opts["finetuned_path"]
        # CUDA SETTINGS
        self.device = torch.device("cuda:0" if opts["use_cuda"] else "cpu")
        self.load_model()
        # Training paraeters
        self.lr = opts["lr"]
        self.batch_size = opts["batch_size"]
        self.epochs = opts["epochs"]
        self.patience = opts["patience"]
        self.model = self.load_model()
        self.best_model = None
        self.train_dataloader, self.valid_dataloader = None, None
        self.optimizer = None
        self.lr_scheduler = None
        self.progress_bar = None
        self.validation_checkpoints = None
        self.current_step, self.current_validation_chekpoint = 0, 0
        self.best_loss = None

    def clear_cuda(self):
        del self.model
        empty_cache()

    def create_progress_bar(self, dataloader, epochs):
        num_update_steps_per_epoch = len(dataloader)
        num_training_steps = epochs * num_update_steps_per_epoch
        return tqdm(range(num_training_steps))

    def load_model(self):
        """Load pre-trained model.
        This method loads a pre-trained model for Masked Language Modeling (MLM).
        Returns:
            PreTrainedModel: The loaded model.
        """
        if self.finetuned_path != "":
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
            path_best_model = self.model_name
            config = None
        return AutoModelForMaskedLM.from_pretrained(path_best_model, config=config).to(
            self.device
        )

    def compute_perplexity(self, losses):
        """Compute perplexity from loss.
        This method computes perplexity from a given loss value.
        Args:
            losses (float): The loss value.
        Returns:
            float: The computed perplexity.
        """
        perplexity = math.exp(losses)
        return perplexity

    def insert_random_mask(self, batch, data_collator):
        """As reported in https://huggingface.co/learn/nlp-course/chapter7/3#fine-tuning-distilbert-with-accelerate,
        ````
        We saw that DataCollatorForLanguageModeling also applies random masking with each evaluation, so we’ll see some fluctuations in our perplexity scores with each training run. One way to eliminate this source of randomness is to apply the masking once on the whole test set, and then use the default data collator.
        ```
        Args:
            batch (dict): The input batch of data.
            data_collator (DataCollatorForLanguageModeling): The data collator object.

        Returns:
            dict: A dictionary containing masked inputs.
        """
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        # Create a new "masked" column for each column in the dataset
        return {k: v.numpy() for k, v in masked_inputs.items()}

    def create_dataloaders(self, tokenizer, dataset):
        """Create dataloaders for training and evaluation.
        This method creates dataloaders for training and evaluation using the provided tokenizer and dataset.
        Args:
            tokenizer (Tokenizer): The tokenizer to use for tokenizing the data.
            dataset (DatasetDict): The dataset containing training and validation data.
        Returns:
            DataLoader: The training dataloader.
            DataLoader: The evaluation dataloader.
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm_probability=0.15
        )
        # First, set a single masking for the validation dataset (so that it will not change within every evaluation)
        eval_dataset = dataset["valid"].map(
            self.insert_random_mask,
            batched=True,
            fn_kwargs={"data_collator": data_collator},
            remove_columns=dataset["valid"].column_names,
            load_from_cache_file=False,
        )
        # Now, create the datalodaders
        self.train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=data_collator,
        )
        # For this, we use the default data collator > masking has been already applied once for all
        self.eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.batch_size, collate_fn=default_data_collator
        )

    def create_lr_scheduler(self):
        """Create a linear learning rate scheduler.
        This method creates a linear learning rate scheduler for the optimizer based on the specified number of training steps.
        Returns:
            Scheduler: The linear learning rate scheduler.
        """
        num_training_steps = self.epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

    def set_validation_steps(self, max_steps=100, validation_checkpoints=4):
        """Set the number of validation steps based on the maximum number of steps and the number of validation checkpoints.

        This method calculates the number of validation steps based on the maximum number of steps and the number of validation checkpoints specified.
        If the maximum number of steps is less than the total number of training steps, the validation checkpoints will be distributed evenly across the training steps.
        If the maximum number of steps is greater than or equal to the total number of training steps, there will be one validation checkpoint per epoch.
        Args:
            max_steps (int, optional): The maximum number of steps. Defaults to 100.
            validation_checkpoints (int, optional): The number of validation checkpoints. Defaults to 4.
        """
        n_training_steps = len(self.train_dataloader)
        if max_steps < n_training_steps:
            self.validation_checkpoints = (
                n_training_steps // validation_checkpoints
            )  # validation_checkpoints per epoch
        else:
            self.validation_checkpoints = n_training_steps  # 1 per epoch

    def train(self, logger, tokenizer, dataset):
        """Train the model using the provided logger, tokenizer, and dataset.
        This method performs the training loop, including creating the dataloaders, setting up the optimizer and learning rate scheduler,
        resizing the token embeddings to match the tokenizer, and starting the training loop.
        Args:
            logger (_type_): The logger object for logging training information.
            tokenizer (_type_): The tokenizer object used to tokenize the dataset.
            dataset (_type_): The dataset object containing the training data.
        """
        logger.experiment_logger.debug("\tCreating the dataloaders...")
        self.create_dataloaders(tokenizer=tokenizer, dataset=dataset.tokenized_ds)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.create_lr_scheduler()
        logger.experiment_logger.debug("\tUpdate vocab size wrt tokenizer...", "debug")
        self.model.resize_token_embeddings(len(tokenizer))
        logger.experiment_logger.debug("\tChoose validation steps per epoch...")
        self.set_validation_steps()
        logger.experiment_logger.debug("\tStarting the training loop...")
        self.training_loop(logger)

    def save_best(self, logger):
        """Save the best model once training has ended.
        Args:
            logger (ExperimentLogger): Logger to save the model
        """
        logger.experiment_logger.debug("\tTraining ended, saving best model...")
        logger.save_best_model(self.best_model)

    def training_loop(self, logger):
        """Perform the training loop.
        This method executes the training loop for the specified number of epochs. It iterates over each epoch, sets the model to training mode, performs batch training, and logs the training loss and perplexity for each epoch.
        Args:
            logger (_type_): The logger object for logging training information.
        """
        self.best_loss = np.inf  # Dummy initialization
        self.progress_bar = self.create_progress_bar(
            self.train_dataloader, epochs=self.epochs
        )
        for epoch in range(self.epochs):
            ######  TRAIN   ######
            self.model.train()  # Model in training mode
            training_losses = self.batch_training(logger)
            logger.tensorboard_writer.add_scalar(
                "Loss_epoch/train", np.mean(training_losses), epoch
            )
            logger.tensorboard_writer.add_scalar(
                "Perplexity_epoch/train",
                self.compute_perplexity(np.mean(training_losses)),
                epoch,
            )

    def batch_training(self, logger):
        """Perform batch training.
        This method executes batch training on the training data. It iterates over each batch in the training dataloader, performs training steps, and logs validation results and best model updates according to the specified validation checkpoints.
        Args:
            logger (_type_): The logger object for logging training information.
        """
        batch_losses = []
        for local_batch in self.train_dataloader:
            if self.current_step % self.validation_checkpoints == 0:
                ######  VALIDATION   ######
                self.model.eval()  # Model in evaluation mode
                validation_losses = self.batch_eval(self.model)
                logger.tensorboard_writer.add_scalar(
                    "Loss_checkpoint/validation",
                    np.mean(validation_losses),
                    self.current_validation_chekpoint,
                )
                logger.tensorboard_writer.add_scalar(
                    "Perplexity_checkpoint/train",
                    self.compute_perplexity(np.mean(validation_losses)),
                    self.current_validation_chekpoint,
                )
                ######  BEST MODEL UPDATE  ######
                if np.mean(validation_losses) < self.best_loss:
                    self.best_loss = np.mean(validation_losses)
                    self.best_model = deepcopy(self.model).to(
                        "cpu"
                    )  # Otherwise we occupy GPU for nothing
                    updated_best_model = 1
                else:
                    updated_best_model = 0
                logger.tensorboard_writer.add_scalar(
                    "Check_best_model/loss",
                    self.best_loss,
                    self.current_validation_chekpoint,
                )
                logger.tensorboard_writer.add_scalar(
                    "Check_best_model/perplexity",
                    self.compute_perplexity(self.best_loss),
                    self.current_validation_chekpoint,
                )
                logger.tensorboard_writer.add_scalar(
                    "Check_best_model/updated",
                    updated_best_model,
                    self.current_validation_chekpoint,
                )
                self.current_validation_chekpoint += 1
                self.model.train()  # set again to training mode
            #### TRAIN ####
            input_ids, attention_mask, labels = (
                local_batch["input_ids"],
                local_batch["attention_mask"],
                local_batch["labels"],
            )
            input_ids, attention_mask, labels = (
                input_ids.to(self.device),
                attention_mask.to(self.device),
                labels.to(self.device),
            )
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            outputs.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()  # Make a scheduler step
            batch_losses.append(outputs.loss.reshape(1))
            del input_ids, attention_mask, labels, outputs
            self.progress_bar.update(1)
            self.current_step += 1
        batch_losses = cat(batch_losses)
        return postprocess_tensor(batch_losses)

    def batch_eval(self, model):
        """Perform batch evaluation.
        This method executes batch evaluation on the evaluation data using the specified model. It iterates over each batch in the evaluation dataloader, performs evaluation steps, and returns the evaluation losses for each batch.
        Args:
            model (_type_): The model used for evaluation.
        """
        batch_losses = []
        progress_bar = self.create_progress_bar(self.eval_dataloader, epochs=1)
        for local_batch in self.eval_dataloader:
            with no_grad():
                # Save info that will be useful to recreate the sessions
                input_ids, attention_mask, labels = (
                    local_batch["input_ids"],
                    local_batch["attention_mask"],
                    local_batch["labels"],
                )
                input_ids, attention_mask, labels = (
                    input_ids.to(self.device),
                    attention_mask.to(self.device),
                    labels.to(self.device),
                )
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
            batch_losses.append(outputs.loss.reshape(1))
            del input_ids, attention_mask, labels, outputs
            progress_bar.update(1)
        batch_losses = cat(batch_losses)
        return postprocess_tensor(batch_losses)

    def inference(self, logger):
        """Perform inference using the best model.
        This method conducts inference using the best model obtained during training. It moves the best model to the specified device, sets it to evaluation mode, evaluates the model on the evaluation dataset, calculates the loss, and logs the results.
        Args:
            logger (_type_): Logger object for logging the results.
        """
        self.clear_cuda()
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        best_loss = self.batch_eval(self.best_model)
        logger.tensorboard_writer.add_scalar(
            "Best_model_results/loss", np.mean(best_loss), 1
        )
        logger.tensorboard_writer.add_scalar(
            "Best_model_results/perplexity",
            self.compute_perplexity(np.mean(best_loss)),
            1,
        )
