import torch
import torch.nn as nn
from tqdm import trange
from .victim import Victim
from typing import *
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PreTrainedModel
from torch.nn import functional as F

class PLMVictim(Victim):
    """
    PLM victims. Support Huggingface's Transformers.

    Args:
        device (:obj:`str`, optional): The device to run the model on. Defaults to "gpu".
        model (:obj:`str`, optional): The model to use. Defaults to "bert".
        path (:obj:`str`, optional): The path to the model. Defaults to "bert-base-uncased".
        num_classes (:obj:`int`, optional): The number of classes. Defaults to 2.
        max_len (:obj:`int`, optional): The maximum length of the input. Defaults to 512.
    """
    def __init__(
        self, 
        device: Optional[str] = "gpu",
        model: Optional[str] = "bert",
        path: Optional[str] = "bert-base-uncased",
        num_classes: Optional[int] = 2,
        max_len: Optional[int] = 512,
        **kwargs
    ):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model_name = model
        self.model_config = AutoConfig.from_pretrained(path)
        self.model_config.num_labels = num_classes
        # you can change huggingface model_config here
        self.plm = AutoModelForSequenceClassification.from_pretrained(path, config=self.model_config)
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.to(self.device)
        
    def to(self, device):
        self.plm = self.plm.to(device)

    def forward(self, inputs):
        output = self.plm(**inputs, output_hidden_states=True, output_attentions=True)
        return output

    def get_repr_embeddings(self, inputs):
        output = getattr(self.plm, self.model_name)(**inputs).last_hidden_state # batch_size, max_len, 768(1024)
        return output[:, 0, :]


    def process(self, batch):
        text = batch["text"]
        labels = batch["label"]
        input_batch = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        labels = labels.to(self.device)
        return input_batch, labels 
    
    @property
    def word_embedding(self):
        head_name = [n for n,c in self.plm.named_children()][0]
        layer = getattr(self.plm, head_name)
        return layer.embeddings.word_embeddings.weight
    # 使用embedding一次训练
    def train_embedding(self, distilled_data):
        self.initialize_model()
        self.plm = self.plm.train()
        
        for step in trange(
                1,  # step默认从1
                leave=False,
                dynamic_ncols=True,
                desc="Train model",
        ):
            batch = distilled_data.get_batch(step)
            # compute loss
            outputs = self.train_forward(
                inputs_embeds=batch["inputs_embeds"],
                labels=batch["labels"],
                output_attentions=True,
            )
            loss_task = outputs.loss.mean()

            attention_labels = batch["attention_labels"]
            if attention_labels is not None:
                attn_weights = torch.stack(outputs.attentions, dim=1)
                attn_weights = attn_weights[..., :attention_labels.size(-2), :]
                assert attn_weights.shape == attention_labels.shape
                loss_attn = F.kl_div(
                    torch.log(attn_weights + 1e-12),
                    attention_labels,
                    reduction="none",
                )
                loss_attn = loss_attn.sum(-1).mean()
            else:
                loss_attn = 0.0
            loss = loss_task + distilled_data.attention_loss_lambda * loss_attn

            # update model
            self.plm.zero_grad()
            loss.backward()
            for params in self.plm.parameters():
                if params.grad is not None:
                    with torch.no_grad():
                        params.sub_(batch["lr"] * params.grad)
    def train_forward(self, *args, **kwargs):
        labels: torch.LongTensor = kwargs.pop(
            "labels") if "labels" in kwargs else None

        outputs: SequenceClassifierOutput = self.plm(*args, **kwargs)
        loss = None
        if labels is not None:
            if outputs.logits.shape == labels.shape:
                # labels: (batch_size, num_labels) or (batch_size)
                labels = labels.view(-1, self.model_config.num_labels)
            else:
                assert labels.ndim == 1
            output = outputs.logits.view(-1, self.model_config.num_labels)
            loss = F.cross_entropy(output, labels, reduction="none")
            assert loss.shape == labels.shape[:1]

        return SequenceClassifierOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def initialize_model(self):
        
        # head_name = [n for n,c in self.plm.named_children()][0]
        # layer = getattr(self.plm, head_name)
        bert_model_config = AutoConfig.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            finetuning_task='sst-2',
            problem_type='single_label_classification',
            output_hidden_states=True
        )
        bert_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            config=bert_model_config,
        )

        initial_state_dict = bert_model.state_dict()
        initialized_module_names = 'classifier'

        bert_model.load_state_dict(initial_state_dict)
        classifier = bert_model.classifier
        classifier.weight.data.normal_(
            mean=0.0,
            std=0.02)
        if classifier.bias is not None:
                classifier.bias.data.zero_()   
        self.plm = bert_model.to(device=self.device)