import logging
import os
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

from datasets import Dataset, disable_progress_bar, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

disable_progress_bar()

TASK_ATTRS = {
    # AGNEWS
    "ag_news": {
        "load_args": ("ag_news", ),
        "sentence_keys": ("text", ),
        "problem_type": "single_label_classification",
        "test_split_key": "test",
        "metric_keys": ("accuracy", ),
    },
    # GLUE
    "mrpc": {
        "load_args": ("glue", "mrpc"),
        "sentence_keys": ("sentence1", "sentence2"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "metric_keys": ("glue", "mrpc"),
    },
    "qnli": {
        "load_args": ("glue", "qnli"),
        "sentence_keys": ("question", "sentence"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "metric_keys": ("glue", "qnli"),
    },
    "sst2": {
        "load_args": ("glue", "sst2"),
        "sentence_keys": ("sentence", ),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "metric_keys": ("glue", "sst2"),
    },
    "qqp": {
        "load_args": ("glue", "qqp"),
        "sentence_keys": ("question1", "question2"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "metric_keys": ("glue", "qqp"),
    },
    "mnli": {
        "load_args": ("glue", "mnli"),
        "sentence_keys": ("premise", "hypothesis"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation_matched",
        "metric_keys": ("glue", "mnli"),
    },
}


@dataclass
class DataConfig:
    task_name: str
    datasets_path: Path
    preprocessed_datasets_path: Path
    train_batch_size: int = 32
    valid_batch_size: int = 256
    test_batch_size: int = 256
    num_proc: int = 1
    force_preprocess: bool = False


class DataModule:
    """DataModule class
    ```
    data_module = DataModule(
        config.data,
        tokenizer_generator=generator.tokenizer,
        tokenizer_learner=learner.tokenizer,
    )
    # preprocess datasets
    data_module.run_preprocess(tokenizer=tokenizer)
    # preprocess external dataset (distilled data)
    data_module.preprocess_dataset(tokenizer=tokenizer, dataset=dataset)
    ```
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # load raw dataset
        # self.dataset_attr = TASK_ATTRS[self.config.task_name]
        # self.datasets: DatasetDict = self.get_dataset()
        # logger.info(f"Datasets: {self.datasets}")

        # self.num_labels = self.datasets["train"].features["labels"].num_classes

        # # preprocessed_dataset
        # self.preprocessed_datasets = None

        # # data collator
        # self.data_collator = None

    def get_dataset(self):
        """load raw datasets from source"""
        if os.path.exists(self.config.datasets_path):
            datasets = load_from_disk(self.config.datasets_path)
        else:
            assert self.config.task_name in TASK_ATTRS
            datasets = load_dataset(*self.dataset_attr["load_args"])

            if "validation" not in datasets:
                datasets["validation"] = datasets.pop(
                    self.dataset_attr["test_split_key"])
            assert datasets.keys() >= {"train", "validation"}

            os.makedirs(os.path.dirname(self.config.datasets_path),
                        exist_ok=True)
            datasets.save_to_disk(self.config.datasets_path)

        if (TASK_ATTRS[self.config.task_name]["problem_type"] ==
                "single_label_classification"):
            # rename label_key
            assert "label" in datasets["train"].features
            datasets = datasets.rename_column("label", "labels")
        else:
            raise NotImplementedError

        return datasets

    def run_preprocess(self, tokenizer: PreTrainedTokenizerFast):
        """datasets preprocessing"""

        # set data_collator
        if self.data_collator is None:
            self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                                         padding="longest",
                                                         pad_to_multiple_of=8)

        if (os.path.exists(self.config.preprocessed_datasets_path)
                and not self.config.force_preprocess):
            logger.info("Load preprocessed datasets from `{}`".format(
                self.config.preprocessed_datasets_path))
            self.preprocessed_datasets = load_from_disk(
                self.config.preprocessed_datasets_path)
            return

        self.preprocessed_datasets = self.preprocess_dataset(
            tokenizer=tokenizer, dataset=self.datasets)

        logger.info(
            f"Save preprocessed datasets to `{self.config.preprocessed_datasets_path}`"
        )
        os.makedirs(os.path.dirname(self.config.preprocessed_datasets_path),
                    exist_ok=True)
        self.preprocessed_datasets.save_to_disk(
            self.config.preprocessed_datasets_path)

    def preprocess_dataset(
        self,
        tokenizer: PreTrainedTokenizerFast,
        dataset: Optional[Dataset | DatasetDict],
    ) -> Dataset | DatasetDict:
        # sentence keys for task
        sentence_keys = TASK_ATTRS[self.config.task_name]["sentence_keys"]

        # get tokenize function
        def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
            sentences = [[s.strip() for s in batch[key]]
                         for key in sentence_keys]
            return tokenizer(*sentences,
                             max_length=tokenizer.model_max_length,
                             truncation=True)

        # tokenize
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=self.config.num_proc,
            desc="Tokenize datasets",
        )

        remove_keys = [
            col for col in dataset["train"].column_names if col not in
            ["input_ids", "token_type_ids", "attention_mask", "labels"]
        ]
        dataset = dataset.remove_columns(remove_keys)

        return dataset

    def train_loader(self) -> DataLoader:
        assert "train" in self.preprocessed_datasets
        assert self.data_collator is not None

        return DataLoader(
            self.preprocessed_datasets["train"],
            batch_size=self.config.train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=True,
        )

    def valid_loader(self) -> DataLoader:
        assert "validation" in self.preprocessed_datasets
        assert self.data_collator is not None

        return DataLoader(
            self.preprocessed_datasets["validation"],
            batch_size=self.config.test_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=False,
        )

    def get_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, data):
        # texts = []
        labels, poison_labels, text = [], [], []
        texts = []
        for text, label, poison_label in data:
            texts.append(text)
            labels.append(label)
            poison_labels.append(poison_label)
        batch = self.tokenizer(texts,
                               padding=True,
                               truncation=True,
                               max_length=512,
                               return_tensors="pt").to(self.device)
        labels = torch.Tensor(labels).long().to(self.device)
        poison_labels = torch.Tensor(poison_labels).to(self.device)
        batch = {
            "input_ids": batch['input_ids'],
            "token_type_ids": batch['token_type_ids'],
            "attention_mask": batch['attention_mask'],
            "labels": labels,
            "poison_label": poison_labels
        }
        return batch

    def get_dataloader(self,
                       dataset: Union[Dataset, List],
                       batch_size: Optional[int] = 4,
                       shuffle: Optional[bool] = True,
                       drop_last: Optional[bool] = False):
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          collate_fn=self.collate_fn,
                          drop_last=drop_last)
