
from itertools import count
import os
import torch
from datasets import concatenate_datasets, load_dataset,Dataset, load_from_disk
import numpy as np

from dataclasses import dataclass, field

import csv

import math
from math import exp
from math import log
import os
import sys
import torch
import random
import collections

from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaForSequenceClassification
from transformers import T5Tokenizer

from typing import Optional, Union, List, Dict, Tuple

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    BertModel,
    BertForPreTraining,
    RobertaModel,
    T5PreTrainedModel,
    T5Model,
    AutoModelWithLMHead,
    T5Tokenizer,
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    DebertaV2Tokenizer
)

from models import T5ForSequenceClassification
from models import DebertaV3ForSequenceClassification

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

os.environ["CUDA_VISIBLE_DEVICES"]="0"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    write_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "where to write test result"
        },
    )
    eval_task: Optional[str] = field(
        default=None,
        metadata={
            "help": "which task to eval"
        },
    )



os.environ["CUDA_VISIBLE_DEVICES"]="0"

max_seq_length=256

parser = HfArgumentParser(ModelArguments)
model_args = parser.parse_args_into_dataclasses()

our_model_path=model_args[0].model_name_or_path

config = AutoConfig.from_pretrained(our_model_path)

if "t5" in our_model_path:
    model = T5ForSequenceClassification.from_pretrained(
                    our_model_path,
                    from_tf=False,
                    config=config,
                    use_auth_token=None,
                    model_args=model_args[0]
                )
    tokenizer = T5Tokenizer.from_pretrained(our_model_path)
elif "deberta" in our_model_path:
    model = DebertaV3ForSequenceClassification.from_pretrained(
                    our_model_path,
                    from_tf=False,
                    config=config,
                    use_auth_token=None,
                    model_args=model_args[0]
                )
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
else:
    print("unsupported model!")
    assert(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
model = model.to(device)

# We give four examples here. The second and the third sentences are considered coming from the true language distribution but the first and the fourth are not.

filled_batch=["I like this movie, and I don't recommend it.", 
              "I like this movie, and I recommend it.", 
              "It is raining, so the ground is wet.", 
              "It is raining, so the ground is dry."]

filled_inputs = tokenizer(filled_batch,max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt")
filled_inputs.to(device)

outputs = model(**filled_inputs, output_hidden_states=True, return_dict=True)

# UD gives a score from 0 to 1 for each sentence. Higher score means the higher probability that the sentence comes from the true language distribution.

logits = outputs.logits
score=np.zeros(len(logits))
for k in range(len(logits)):
    score[k]=exp(logits[k][1])/(exp(logits[k][0])+exp(logits[k][1]))    
    
for i in range(len(logits)):
    print(filled_batch[i],round(score[i],3))