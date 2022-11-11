
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

from simcse.models import T5ForSequenceClassification

from sklearn.linear_model import LinearRegression

import pickle

from torch.utils.data import DataLoader


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
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.file_utils import cached_property, torch_required, is_torch_available, is_torch_tpu_available
from simcse.models import RobertaForCL, BertForCL
from simcse.trainers import CLTrainer

from simcse.models import T5ForSequenceClassification
from simcse.models import DebertaV3ForSequenceClassification

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

    # SimCSE's arguments
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



os.environ["CUDA_VISIBLE_DEVICES"]="2"

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.matmul.allow_tf32 = False
        #if hasattr(mpu, 'model_parallel_cuda_manual_seed'):
        #    mpu.model_parallel_cuda_manual_seed(seed)

all_task_family = [
    'coreference_resolution', 'natural_language_inference', 
    'paraphrase_identification', 'closed_book_qa', 'extractive_qa', 
    'multiple_choice_qa', 'sentiment', 'sentence_completion', 
    'structure_to_text', 'summarization', 'topic_classification', 'word_sense_disambiguation']
train_task_family = [
    'paraphrase_identification', 'closed_book_qa', 'extractive_qa', 
    'multiple_choice_qa', 'sentiment', 'structure_to_text', 'summarization', 
    'topic_classification']
test_task_family = [
    'coreference_resolution', 'natural_language_inference', 
    'sentence_completion', 'word_sense_disambiguation']
TASK_TYPE_DICT = {
    "coreference_resolution": [
        "super_glue/wsc.fixed", "winogrande/winogrande_xl"
    ],
    "natural_language_inference":[
        "super_glue/cb", "super_glue/rte", "anli"
    ],
    "paraphrase_identification":[
        "glue/mrpc", "glue/qqp", "paws/labeled_final"
    ],
    "closed_book_qa":[
        # "ai2_arc/ARC Challenge",
        # "ai2_arc/ARC_Easy",
        "kilt_tasks/hotpotqa",
        # "trivia_qa/unfiltered",
        # "web_questions",
        "wiki_qa"
    ],
    "extractive_qa":[
        "adversarial_qa/dbidaf",
        "adversarial_qa/dbert",
        "adversarial_qa/droberta",
        "duorc/SelfRC",
        "duorc/ParaphraseRC",
        "ropes",
        ],
    "multiple_choice_qa":[
        # "commonsense_qa",
        "cosmos_qa",
        "dream",
        "qasc",
        "quail",
        "quarel",
        "quartz",
        "sciq",
        "social_i_qa",
        "wiki_hop/original",
        "wiqa",
        ],
    "sentiment": [
        "amazon_polarity", "app_reviews", "imdb", "rotten_tomatoes", "yelp_review_full"
    ],
    "sentence_completion": [
        "super_glue/copa", "story_cloze/2016", "hellaswag"
    ],
    "structure_to_text": [
        "common_gen", "wiki_bio"
    ],
    "summarization": [
        "cnn_dailymail/3.0.0", "gigaword", "multi_news", "samsum", "xsum"
    ],
    "topic_classification": [
        "ag_news", "dbpedia_14", "trec"
    ],
    "word_sense_disambiguation": [
        "super_glue/wic"
    ]
}


test_data_file = "../t0_eval"

set_random_seed(31)

max_seq_length=256
#max_seq_length=512

#model = AutoModel.from_pretrained(our_model_path)
#model = RobertaForSequenceClassification.from_pretrained(our_model_path)
#model = T5ForSequenceClassification.from_pretrained(our_model_path)

parser = HfArgumentParser(ModelArguments)
model_args = parser.parse_args_into_dataclasses()

our_model_path=model_args[0].model_name_or_path
write_path=model_args[0].write_path


test_info_file=open(write_path.split(".csv")[0]+"_info",'w')

config = AutoConfig.from_pretrained(our_model_path)

if "t5" in our_model_path or "checkpoint-8000" in our_model_path:
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
#device = torch.device("cpu")
model.eval()
model = model.to(device)

test_all_data=os.listdir(test_data_file)

test_string=[]
for task in test_task_family:
    for name in TASK_TYPE_DICT[task]:
        #if "anli" in name:
        #    continue
        test_string.append(name)

#test_string=["super_glue/copa"]
#test_string=["super_glue/rte"]
#test_string=["hellaswag"]
#test_string=["super_glue/rte","super_glue/cb","anli"]
#test_string=["super_glue/cb"]
#test_string=["winogrande/winogrande_xl"]
#test_string=["super_glue/wsc.fixed"]
#test_string=["anli"]
#test_string=["super_glue/wic"]
#test_string=["super_glue/cb","super_glue/rte","anli","super_glue/wic"]
#test_string=["story_cloze/2016"]
#test_string=["super_glue/cb","anli"]

if model_args[0].eval_task!=None:
    test_string=[model_args[0].eval_task]

test_datasets=[]
for dataset_string in test_string:
    dataset_name=dataset_string.replace("/","_")
    for name in test_all_data:
        if dataset_name in name and "score_eval" not in name:
            dataset_prompted=name
            test_datasets.append(dataset_prompted)
            #break

#test_datasets=["super_glue_rte_GPT_3_style"]

print(test_datasets)

#test_datasets=[test_datasets[0]]

def results_write(write_path,results):
    with open(write_path,"w") as csvfile:
        writer=csv.writer(csvfile)  
        writer.writerow(["dataset","correct","total","accuracy","random guess","all_stats","wrong_stats","correct_stats"])
        for example in results:
            writer.writerow(example)

results=[]
for name in test_datasets:
    if name in test_datasets:
        read_path = os.path.join(test_data_file,name,"validation")
        if os.path.exists(read_path) is False:
            read_path = os.path.join(test_data_file,name,"test")
    else:
        continue
    print(f"Begin process:{read_path}")
    print(f"Begin process:{read_path}",file=test_info_file)
    dataset=load_from_disk(read_path)

    our_modified_prompt=False
    if "hellaswag" in name:
        if "complete_first_then" in name:
            our_modified_prompt= True
            #continue
            #print("previous acc: 0.399")
        else:
            continue
    elif "story_cloze" in name:
        if "story_cloze_2016_Answer_Given_options" in name:
            our_modified_prompt= True
        else:
            continue
    elif "copa" in name:
        if "premise_so_because" in name:
            our_modified_prompt= True
            #continue
            #print("previous acc: 0.740")
        else:
            continue
    elif "cb" in name:
        if "can_we_infer" in name:
            our_modified_prompt=True
            #print("previous acc: 0.500")
        else:
            continue
    elif "rte" in name:
        if "does_it_follow_that" in name:
            our_modified_prompt=True
            #print("previous acc: 0.585")
        else:
            continue
    elif "winogrande" in name:
        if "xl_does_underscore_refer_to" in name:
            our_modified_prompt=True
            #print("previous acc: 0.536")
        else:
            continue
    elif "wsc" in name:
        #super_glue_wsc.fixed_in_other_words
        if "fixed_replaced_with" in name:
            our_modified_prompt=True
            #print("previous acc: 0.663")
        else:
            continue
    elif "anli" in name:
        if "can_we_infer" in name:
            our_modified_prompt=True
            #print("previous acc: 0.34/0.32/0.35")
        else:
            continue
    elif "wic" in name:
        if "polysemous" in name:
            our_modified_prompt=True
            #print("previous acc: 0.509")
        else:
            continue
    
    print(name, "len: ", len(dataset))
    print(name, "len: ", len(dataset), file=test_info_file)


    num_choices=len(dataset[0]["answer_choices"])

    #dataloader=DataLoader(dataset,batch_size=64)

    batch_size=4

    correct=0
    now=0
    short_correct=0
    short_now=0
    all_stats=np.zeros(3)
    wrong_stats=np.zeros(3)
    correct_stats=np.zeros(3)
    with torch.no_grad():
        #for i in range(0,100,batch_size):
        for i in range(0,len(dataset),batch_size):

            '''filled_batch=[]
            for k in range(i,min(i+batch_size,len(dataset))):
                example=dataset[k]
                choices=example["answer_choices"]
                input=example["inputs_pretokenized"]
                filled_batch.append(input)
            tokenized_input = tokenizer(filled_batch, padding=True, return_tensors="pt")'''

            copa_front=False

            filled_batch=[]
            for k in range(i,min(i+batch_size,len(dataset))):
                
                example=dataset[k]

                choices=example["answer_choices"]
                input=example["inputs_pretokenized"]

                if "hellaswag" in name:
                    if our_modified_prompt==True:
                        input=input.split("...")[0]
                        input=input.replace("Complete the description with an appropriate ending:","")
                        input=input.replace("First, ","",1)
                        input=input.replace("Then, ","",1)
                        if(k<30):
                            print(k, input, "###", choices[0], "|| ",choices[1], "||", choices[2], "|| ",choices[3], "###", input+choices[0])
                            print(k, input, "###", choices[0], "|| ",choices[1], "||", choices[2], "|| ",choices[3], "###", input+choices[0], file=test_info_file)
                elif "story_cloze" in name:
                    if our_modified_prompt==True:
                        input=input.split("What is a possible continuation for the story given the following options ?")[0]

                        if(k<30):
                            print(k, input, "###", choices[0], "|| ",choices[1], "###", input+choices[0])
                            print(k, input, "###", choices[0], "|| ",choices[1], "###", input+choices[0], file=test_info_file)
                elif "copa" in name:
                    if our_modified_prompt==True:
                        input=input.split("?")[-1]
                        
                        #input=input.split("  so")[0]
                        #input=input.split("  because")[0]
                        
                        #input=input.replace(".",",")
                        #input=" "+input+" "
                        input=input+" "

                        if input.count("because")>0:
                            input=input.replace("because", "cause")
                        else:
                            input=input.replace("so", "effect")
                        
                        
                        """if input.count("because")>0:
                            input=input.replace("because", "cause")
                        else:
                            input=input.replace("so", "effect")"""

                        """if input.count(" because")>0:
                            input=input.replace(" because", "")
                            copa_cause=True
                        else:
                            input=input.replace(" so", "")
                            copa_cause=False"""
                        
                        """if input.count(" because")>0:
                            input=input.replace("because", "cause")
                            copa_front=False
                        else:
                            input=input.replace("so", "")
                            input=" cause"+input
                            copa_front=True"""
                        
                        """if input.count(" because")>0:
                            input=input.replace(" because", "")
                            input=" effect"+input
                            copa_front=True
                        else:
                            input=input.replace(" so", " effect")
                            copa_front=False"""
                        
                        """if input.count(" because")>0:
                            input=input.replace(" because", "")
                            input=" so"+input
                            copa_front=True
                        else:
                            copa_front=False"""

                        #input=input+"."
                        if(k<20):
                            print(k, input, "###", choices[0], "|| ",choices[1], "###", input+choices[0])
                            print(k, input, "###", choices[0], "|| ",choices[1], "###", input+choices[0],file=test_info_file)
                elif "cb" in name:
                    if our_modified_prompt==True:
                        input2=input.split("Can we infer that ")
                        #input=input2[0]+"We can infer that"+input2[1]
                        #input=input2[0]+"Can we infer that"+input2[1]

                        #input=input2[0]+" "+input2[1]
                        #input=input2[0]+" "+input2[1].replace('"','')
                        input="Premise: "+input2[0]+" "+"Hypothesis: "+input2[1].replace('"','')
                        input=input.split("Yes, no, or maybe?")[0]
                        input=input.replace("Suppose ","", 1)
                        input=input[:-2]
                        
                        #input2=input.split('"')
                        #input=""
                        #for phase in input2:
                        #    input+=phase
                        #input=input2[0]+input2[1][0].upper()+input2[1][1:]+"."
                        #input=input2[0]+"We can infer that "+'"'+input2[1]+'"'+"."
                        if(k<30):
                            print(k, input, "###", choices[0], "|| ", choices[1], "||", choices[2], "###", input+choices[0])
                            print(k, input, "###", choices[0], "|| ", choices[1], "||", choices[2], "###", input+choices[0],file=test_info_file)
                elif "rte" in name:
                    if our_modified_prompt==True:
                        input=input.split("Given that ")[-1]
                        
                        input2=input.split("Does it follow that ")
                        #input=input2[0]+input2[1]
                        #input=input2[0]+" "+input2[1].replace('"','')
                        input="Premise: "+input2[0]+" "+"Hypothesis: "+input2[1].replace('"','')

                        #input=input2[0].replace(".",",")+" we can infer that "+input2[1]
                        #input=input2[0]+" We can infer that "+input2[1]
                        input=input.split("Yes or no?")[0]

                        if(k<30):
                            print(k, input, "###", choices[0], "|| ", choices[1], "###", input+choices[0])
                            print(k, input, "###", choices[0], "|| ", choices[1], "###", input+choices[0],file=test_info_file)
                elif "winogrande" in name:
                    if our_modified_prompt==True:
                        input=input.split(" In the previous sentence")[0]
                        if(k<30):
                            print(k, input, "###", choices[0], "|| ", choices[1], "###", input+choices[0])
                            print(k, input, "###", choices[0], "|| ", choices[1], "###", input+choices[0],file=test_info_file)
                elif "wsc" in name:
                    if our_modified_prompt==True:
                        """
                        input=input.split(" True or false?")[0]
                        input2=input.split("In other words,")
                        input=input2[0]+input2[1]
                        if(k<10):
                            print(k, input, "###", choices[0], "||", choices[1])
                        """
                        input=input.split("In the previous sentence, can the pronoun ")
                        sent=input[0]
                        input3=input[-1].split("? Yes or no?")[0]
                        input4=input3.split(" be replaced with ")
                        word1=input4[0][1:-1]
                        word2=input4[-1][1:-1]
                        
                        input_old=sent
                        if word1=="his" or word1=="her":
                            word2=word2+"'s"

                        #if sent.count(word1)!=1:
                        #    print(k, "#######unusual####### ", sent.count(word1),"times"
                        input=sent.replace(" "+word1+" "," "+word2+" ")
                        if(k<30):
                            print(k, input, "###", word1, "||", word2)
                            print(k, input, "###", word1, "||", word2,file=test_info_file)
                elif "anli" in name:
                    if our_modified_prompt==True:
                        input2=input.split("Can we infer that ")
                        #input=input2[0]+input2[1]
                        #input=input2[0]+" "+input2[1].replace('"','')
                        input="Premise: "+input2[0]+" "+"Hypothesis: "+input2[1].replace('"','')
                        input=input.split("? Yes, no, or maybe?")[0]
                        input=input.replace("Suppose ","",1)
                        if(k<30):
                            print(k, input, "###", choices[0], "|| ", choices[1], "||", choices[2], "###", input+choices[0])
                            print(k, input, "###", choices[0], "|| ", choices[1], "||", choices[2], "###", input+choices[0],file=test_info_file)
                elif "wic" in name:
                    if our_modified_prompt==True:
                        word=input.split(" ")[2]
                        word_no_quote=word[1:-1]
                        input=input.split("Sentence 1: ")[-1]
                        input2=input.split("Sentence 2: ")
                        #input=f"{word_no_quote} has the same meaning in both sentences."+input2[0]+input2[1]
                        input=input2[0]+input2[1]
                        #input=input2[0][:-2]+" and "+input2[1]
                        
                        #input=input.replace(word_no_quote,word)
                        input=input.replace('"','')
                        if(k<30):
                            #print(word,word_no_quote)
                            print(k, input, "###", choices[0], "||", choices[1])
                            print(k, input, "###", choices[0], "||", choices[1],file=test_info_file)
    
                
                long_sentence=False
                cut_strategy=3
                if cut_strategy==1:
                    pre_inputs=[]
                    for choice in choices:
                        pre_inputs.append(input+choice)
                    tokenized_pre_inputs = tokenizer(pre_inputs, padding=True, return_tensors="pt")

                    len_pre_inputs=len(tokenized_pre_inputs["input_ids"][0])
                    
                    if(len_pre_inputs>max_seq_length):
                        tokenized_input = tokenizer(input, padding=True, return_tensors="pt")
                        len_input=len(tokenized_input["input_ids"][0])
                        token_cut=tokenized_input["input_ids"][0][-(max_seq_length-(len_pre_inputs-len_input-5)):-1]
                        #print("before: ", input)
                        input=tokenizer.decode(token_cut)
                        #print("after:", input)
                        long_sentence=True
                elif cut_strategy==2:
                    tokenized_input = tokenizer(input, padding=True, return_tensors="pt")
                    len_input=len(tokenized_input["input_ids"][0])
                    if len_input>max_seq_length-10:
                        token_cut=tokenized_input["input_ids"][0][-(max_seq_length-10):-1]
                        input=tokenizer.decode(token_cut)
                        long_sentence=True
                elif cut_strategy==3:
                    tokenized_input = tokenizer(input, padding=True, return_tensors="pt")
                    len_input=len(tokenized_input["input_ids"][0])
                    if len_input>max_seq_length:
                        token_cut=tokenized_input["input_ids"][0][-max_seq_length:-1]
                        input=tokenizer.decode(token_cut)
                        long_sentence=True
                        assert "rte" in name or "cb" in name
                    
                
                target=example["targets_pretokenized"]

                if True:
                    if "rte" in name and our_modified_prompt==True:
                        filled_batch.append(input)
                    elif "cb" in name and our_modified_prompt==True:
                        filled_batch.append(input)
                    elif "wsc" in name and our_modified_prompt==True:
                        filled_batch.append(input)
                        filled_batch.append(input_old)
                    elif "anli" in name and our_modified_prompt==True:
                        filled_batch.append(input)
                    elif "wic" in name and our_modified_prompt==True:
                        filled_batch.append(input)
                    elif "winogrande" in name and our_modified_prompt==True:
                        for choice in choices:
                            filled_batch.append(input.replace("_",choice.strip()))
                            #filled_batch.append(input+target.replace(target.strip(),choice.strip()))
                    elif "copa" in name and our_modified_prompt==True:
                        for choice in choices:
                            if copa_front==True:
                                filled_batch.append(target.replace(target.strip(),choice.strip())+input)
                            else:    
                                filled_batch.append(input+target.replace(target.strip(),choice.strip()))
                            #filled_batch.append(input+target.replace(target.strip(),choice.strip()))
                    else:
                        for choice in choices:
                            filled_batch.append(input+target.replace(target.strip(),choice.strip()))
                else:
                    for choice in choices:
                        filled_batch.append(input+target.replace(target.strip(),choice.strip()))

            filled_inputs = tokenizer(filled_batch,max_length=max_seq_length, padding=True, truncation=True, return_tensors="pt")
            filled_inputs.to(device)
            
            #print("number of filled input", len(filled_inputs))

            outputs = model(**filled_inputs, output_hidden_states=True, return_dict=True)

            #print("output", outputs)

            logits = outputs.logits

            #score=logits[:,1]
            score=np.zeros(len(logits))
            for k in range(len(logits)):
                score[k]=exp(logits[k][1])/(exp(logits[k][0])+exp(logits[k][1]))

            id=0
            for k in range(i,min(i+batch_size,len(dataset))):
                example=dataset[k]
                #input=example["inputs_pretokenized"].strip()

                choices=example["answer_choices"]
                target=example["targets_pretokenized"].strip()
                
                if True:
                    if "rte" in name and our_modified_prompt==True:
                        pf=exp(logits[id][0])
                        pt=exp(logits[id][1])
                        if pf>pt:
                            answer_id=1
                        else:
                            answer_id=0
                        if k<100:
                            print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target)
                            print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target,file=test_info_file)
                        id+=1
                    elif "cb" in name and our_modified_prompt==True:
                        pf=exp(logits[id][0])
                        pt=exp(logits[id][1])
                        if pf>pt:
                            answer_id=1
                        else:
                            answer_id=0
                        id+=1
                        if k<100:
                            print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target)
                            print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target,file=test_info_file)
                    elif "wsc" in name and our_modified_prompt==True:
                        #pf=exp(logits[id][0])
                        #pt=exp(logits[id][1])

                        pt=score[id]
                        pt_old=score[id+1]
                        
                        #if pt_old>pt:

                        """if pt_old>pt+0.03:
                            answer_id=0
                        else:
                            answer_id=1"""

                        if pt<0.5:
                            answer_id=0
                        else:
                            answer_id=1
                            
                        id+=2
                        #print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target)
                        #print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target,file=test_info_file)
                        print("id=", k, "prob true old", pt_old, "prob true", pt, "answer", choices[answer_id].strip(), "target", target)
                        print("id=", k, "prob true old", pt_old, "prob true", pt, "answer", choices[answer_id].strip(), "target", target,file=test_info_file)
                    elif "anli" in name and our_modified_prompt==True:
                        pf=exp(logits[id][0])
                        pt=exp(logits[id][1])
                        if pf>pt:
                            answer_id=2
                        else:
                            answer_id=0
                        id+=1
                        if k<100:
                            print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target)
                            print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target,file=test_info_file)
                    elif "wic" in name and our_modified_prompt==True:
                        pf=exp(logits[id][0])
                        pt=exp(logits[id][1])
                        if pf>pt:
                            answer_id=0
                        else:
                            answer_id=1
                        id+=1
                        print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target)
                        print("id=", k, "prob false", pf, "prob true", pt, "answer", choices[answer_id].strip(), "target", target,file=test_info_file)
                    else:
                        answer_id=score[id:id+len(choices)].argmax()
                        if k<100:
                            print("id=", k, "score", score[id:id+len(choices)], "choices", choices, "answer", choices[answer_id].strip(), "target", target)
                            print("id=", k, "score", score[id:id+len(choices)], "choices", choices, "answer", choices[answer_id].strip(), "target", target, file=test_info_file)
                        id+=len(choices)
                else:
                    answer_id=score[id:id+len(choices)].argmax()
                    id+=len(choices)
                
                if choices[answer_id].strip()==target:
                    correct+=1
                now+=1
                if len(choices)<=3:
                    all_stats[answer_id]+=1
                    if choices[answer_id].strip()!=target:
                        wrong_stats[answer_id]+=1
                    for ch_id in range(len(choices)):
                        if choices[ch_id].strip()==target:
                            correct_stats[ch_id]+=1

    accuracy=correct/now
    print(name, "correct:", correct,"total:", now, "accuracy: %.3f\n"%accuracy, "all_stats", all_stats, "wrong_stats", wrong_stats, "correct_stats", correct_stats)
    print(name, "correct:", correct,"total:", now, "accuracy: %.3f\n"%accuracy, "all_stats", all_stats, "wrong_stats", wrong_stats, "correct_stats", correct_stats,file=test_info_file)
    #if short_now!=0:
    #    print(i,len(dataset),"short correct=",short_correct, "short_now=",short_now, "accuracy=",short_correct/short_now)
    results.append([name,correct,now,accuracy,1/num_choices,all_stats, wrong_stats,correct_stats])

    results_write(write_path,results)

'''with open(write_path,"w") as csvfile:
    writer=csv.writer(csvfile)  
    writer.writerow(["dataset","correct","total","accuracy","random guess"])
    for example in results:
        writer.writerow(example)'''