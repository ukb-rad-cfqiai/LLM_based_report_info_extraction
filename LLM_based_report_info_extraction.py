# Copyright 2024 Clinic for Diagnositic and Interventional Radiology, University Hospital Bonn, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

MODALITY = "Röntgen-Thorax"
TEXT_DESCRIPTION = "Bericht"

CLASSES   = [   "Lage_ZVK_ok",
                "Lage_ZVK_Fehllage",
                "Befund_Pleuraerguss",
                "Befund_Stauung",
                "Befund_Infiltrat",
                "Befund_Pneumothorax" ]

CLASSES_NEW = [ "ZVK",
                "ZVK hat fehlerhafte Projektion/Lage",
                "Erguss",
                "Stauung",
                "Infiltrate",
                "Pneumothorax" ]

POSITIVE_STR = "1"; NEGATIVE_STR = "0"; PLACEHOLDER_STR = "0"
EXAMPLE_OUTPUT = '{'+",\n".join([ f'"{c}": {PLACEHOLDER_STR}' for c in CLASSES_NEW])+'}' #{POSITIVE_STR}/{NEGATIVE_STR}

def new_to_old_label(new_label):
    old_label = [0 for _ in new_label]
    if new_label[0]==1 and new_label[1]==0: old_label[0]=1 
    old_label[1] = new_label[1] 
    old_label[2] = new_label[2]
    old_label[3] = new_label[3]
    old_label[4] = new_label[4]
    old_label[5] = new_label[5]
    return old_label
#def new_to_old_label(new_label): return new_label

def old_to_new_label(old_label):
    new_label = [0 for _ in old_label]
    if old_label[0]==1 or old_label[1]==1: new_label[0]=1 
    new_label[1] = old_label[1] 
    new_label[2] = old_label[2]
    new_label[3] = old_label[3]
    new_label[4] = old_label[4]
    new_label[5] = old_label[5] 
    return new_label
#def old_to_new_label(old_label): return old_label

def label_to_output_seq(label): 
    return '{'+",\n".join([ f'"{c}": {POSITIVE_STR}' if x==1 else f'"{c}": {NEGATIVE_STR}'for x, c in zip(old_to_new_label(label), CLASSES_NEW) ])+'}'

PRE_GIVEN_OUTPUT = f'"{CLASSES_NEW[0]}": '

SYSTEM_PROMPT = f"Du bist ein hilfreicher AI assistant, welcher radiologische {MODALITY} {TEXT_DESCRIPTION} in JSON Format strukturiert."
USER_PROMPT = f"""Am Ende dieser Anweisung gebe ich dir einen {TEXT_DESCRIPTION}, für welchen du die Beurteilungen und Erkenntnisse des Radiologen in folgendem JSON Format zusammenzufassen:
{EXAMPLE_OUTPUT}  
Du gibts immer dieses vollständige JSON Format mit allen {len(CLASSES_NEW)} Klassen an und ersetzt {PLACEHOLDER_STR} durch {POSITIVE_STR}, wenn folgendes im {TEXT_DESCRIPTION} zu finden ist:
Bei \"{CLASSES_NEW[0]}\" ersetzt du {PLACEHOLDER_STR} durch {POSITIVE_STR} im JSON Format, wenn der Patient einen zentralen Venenkatheter (ZVK) hat. Andere Fremdmaterialen, wie z.B Shaldon-Katheter oder Magensonden, sind für dich nicht relevant.
Bei \"{CLASSES_NEW[1]}\" ersetzt du {PLACEHOLDER_STR} durch {POSITIVE_STR} im JSON Format, wenn der im {TEXT_DESCRIPTION} beschriebene zentrale Venenkatheter (ZVK) eine fehlerhafte Postion auffweist.
Bei den Klassen "Erguss", "Stauung", "Infiltrate" und "Pneumothorax" ersetzt du {PLACEHOLDER_STR} durch {POSITIVE_STR}, wenn der Radiologe im Bericht vermerkt hat, dass er die jeweilige Pathologie im Bild erkannt hat, unabhängig davon, ob sie neu ist oder auch bereits bei einer früheren Untersuchung bestand (Beispiel: Differentialdiagnose (DD) pneumonische Infiltrate). Beschreibt der Radiologe, dass er die betreffende Pathologie auf dem Bild nicht sieht (Beispiel: "Kein Nachweis von umschriebenen pneumonischen Infiltraten") oder wenn er Unsicherheiten beschreibt (Beispiel: "Infiltrate können nicht mit Sicherheit ausgeschlossen werden / kein sicherer Nachweis"), dann lasse {PLACEHOLDER_STR} im JSON für die jeweilige Pathologie stehen."""
TIPPS = f"""Hinweise: Bei Beschreibungen eines zentralen Venenkatheters (ZVK) mit "Projektion auf die obere Hohlvene", "Projektion auf Vena Cava Superior (VCS)" oder "Projektion auf den cavo-atrialen Übergang" liegt eine regelrechte Lage des ZVK vor und du lässt die {NEGATIVE_STR} bei \"{CLASSES_NEW[1]}\" stehen. Du ersetzt {PLACEHOLDER_STR} durch {POSITIVE_STR} bei \"{CLASSES_NEW[1]}\" bei jeglichen Beschreibungen von Projektionen auf andere Anatomien, wie z.B. bei "Projektion auf den rechten Vorhof" oder bei der Beschreibung eines umgeschlagenen ZVK, denn dann liegt eine fehlerhafte Projektion/Lage vor."""

EXAMPLE_REPORT = PUT YOUR EXAMPLE REPORT HERE
EXAMPLE_LABEL = PUT THE GT LABEL FO THE EXAMPLE HERE AS LISTS E.G. [0, 1, 1, 1, 1, 0]
ONE_SHOT_REPORT = f"Dies ist ein Beispiel für eine {TEXT_DESCRIPTION}: {EXAMPLE_REPORT}"
ONE_SHOT_OUTPUT = label_to_output_seq(EXAMPLE_LABEL)  
ONE_SHOT_EXAMPLE = f"{ONE_SHOT_REPORT}\nDies ist für den Beispiel {TEXT_DESCRIPTION} ein Korrekt ausgefülltes JSON: {ONE_SHOT_OUTPUT}"
PRE_REPORT_PROMPT = f"Dies ist der {TEXT_DESCRIPTION} den du jetzt klassifizieren sollst: "

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report)

import os, argparse, torch, json, re,  pickle, ast, time
from torch.utils.data import DataLoader, default_collate

from collections import OrderedDict
join = os.path.join
os.environ["TOKENIZERS_PARALLELISM"] = "true" #https://github.com/huggingface/transformers/issues/5486
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"  #https://discuss.huggingface.co/t/get-using-the-call-method-is-faster-warning-with-datacollatorwithpadding/23924/5
from glob import glob
from accelerate import Accelerator, logging
logger = logging.get_logger(__name__, log_level="INFO")

import datasets
from datasets import load_from_disk, Dataset, set_caching_enabled
set_caching_enabled(False)
from peft import (get_peft_model, load_peft_weights, set_peft_model_state_dict, prepare_model_for_kbit_training,
                     LoraConfig, TaskType, AutoPeftModelForSequenceClassification, AutoPeftModelForCausalLM)
import bitsandbytes as bnb
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import numpy as np
from huggingface_hub import login
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    EarlyStoppingCallback,
    set_seed,
    BitsAndBytesConfig,
    GenerationConfig,
    HfArgumentParser,
    EvalPrediction,
    TrainerCallback,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq )

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

@dataclass
class ModelArguments: 
    model_path: Optional[str] = field( default= "lmsys/vicuna-13b-v1.5" ) 
    adapters_pretrained_path: Optional[str] = field( default=None ) 
    hf_access_token: Optional[str] = field( default=None ) 

@dataclass
class DataArguments: 
    max_train_samples: Optional[int] = field( default=None, metadata={"help": "For dev purposes or quicker training, truncate the number of training examples to this value if set." } )
    max_eval_samples: Optional[int] = field( default=None, metadata={"help": "For dev purposes or quicker training, truncate the number of evaluation examples to this value if set." } )
    dataset_path: str = field( default=None, required=True, metadata={"help": "Which dataset to finetune on. See datamodule for options."} )
    dataset_path_test: str = field( default=None, metadata={"help": "test data."} )
    dataset_path_test2: str = field( default=None, metadata={"help": "test data."} )
    dataset_path_predict : str = field( default=None, metadata={"help": "test data."} )

@dataclass
class TrainingArguments(TrainingArguments): 
    output_dir: str = field( default=None, required=True, metadata={"help": 'The output dir for logs and checkpoints'})
    report_to: str = field( default='tensorboard', metadata={"help": "To use wandb or something else for reporting."} )
    cache_dir: Optional[str] = field( default=None ) 
    double_quant: bool = field( default=True, metadata={"help": "Compress the quantization statistics through double quantization."} ) 
    max_steps: int = field( default=256, metadata={"help": "How many training steps to use."} ) # oder 4
    bits: int = field( default=4, metadata={"help": "How many bits to use."} ) # oder 4
    lora_r: int = field( default=8, metadata={"help": "Lora R dimension."} ) #https://medium.com/@tom_21755/llm-optimization-layer-wise-optimal-rank-adaptation-lora-1444dfbc8e6a
    lora_alpha: float = field( default=16, metadata={"help": " Lora alpha."} ) 
    lora_dropout: float = field( default=0.05, metadata={"help":"Lora dropout."} ) 
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    weight_decay: float = field(default=0.01, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=1e-4, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    do_test: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    evaluation_strategy: str = field(default='steps', metadata={"help": 'Eval strategy must be same than Save strategy'})
    metric_for_best_model: str = field(default='eval_loss', metadata={"help": 'metric_for_best_model'})
    eval_steps: int = field(default=10, metadata={"help": 'Steps until eval'})
    eval_accumulation_steps: int = field(default=1, metadata={"help": 'Number of predictions steps to accumulate the output tensors for, before moving the results to the CPU. I'})
    save_steps: int = field(default=10, metadata={"help": 'Steps until eval'})
    logging_steps: int = field(default=10, metadata={"help": 'Steps until eval'})
    load_best_model_at_end: bool = field(default=True, metadata={"help": 'Loading best model at end. required for early stopping. Is set to True if early stopping is allpied. '})
    save_total_limit: int = field(default=2, metadata={"help": 'How many checkpoints to save before the oldest is overwritten. Default 1 with default load_best_model_at_end=True means saving best and last'})
    early_stopping: bool = field(default=False, metadata={"help": 'Apply early stopping with early_stopping_patience.'})
    early_stopping_patience: int = field(default=10, metadata={"help": 'How many eval iterations to not improve for stopping'})
    dataloader_num_workers: int = field(default=24, metadata={"help": 'Num workers to use in dataloader.'})
    max_seq_length_input: int = field(default=256, metadata={"help": 'Maximum sequenz length in data. Check prior to training'})
    max_seq_length_output: int = field(default=64, metadata={"help": 'Maximum sequenz length in data. Check prior to training'})
    max_memory_MB: int = field( default=75000, metadata={"help": "Free memory per gpu."} ) #
    bf16: bool = field(default=False, metadata={"help": 'Train with bf16 which is good with A100'})
    fp16: bool = field(default=False, metadata={"help": 'Train fp16'})
    ddp_find_unused_parameters: bool = field(default=False, metadata={"help": 'Can help reduce RAM consumption at cost of performance'})
    disable_tqdm: bool = field(default=False, metadata={"help": 'Disable TQDM'})
    with_tipps: bool = field(default=False, metadata={"help": 'Train with tipps'})
    zero_shot: bool = field(default=False, metadata={"help": 'Dont train just apply zero shot'})
    one_shot: bool = field(default=False, metadata={"help": 'Train with tipps'})
    add_do_fine_tune: bool = field(default=False, metadata={"help": 'Train with tipps'})
    get_individual_thresh_perClass: bool = field(default=False, metadata={"help": 'For multilabel class make thresh after sigmoid based on best f1 score on train'})
    neftune_noise_alpha: float = field(default=5.0, metadata={"help": 'Shall make fine tuning better https://arxiv.org/abs/2310.05914'})
    total_train_batch_size: int = field(default=480, metadata={"help": 'Bathc size used for generation'})
    per_device_generate_batch_size: int = field(default=8, metadata={"help": 'Bathc size used for generation'})
    full_eval_valid_set: bool = field(default=True, metadata={"help": 'Make full eval on valid set at the end'})
    device_map: str = field(default='accelerate', metadata={"help": 'auto or accelerate'})
    trust_remote_code: bool = field(default=False, metadata={"help": 'You need this mostly for flash attn2'})
    attn_implementation: str = field(default='sdpa', metadata={"help": 'attn_implementation'})
    eval_best_model: bool = field(default=True, metadata={"help": 'Weather to eval the best model after training'})
    eval_last_model: bool = field(default=True, metadata={"help": 'Weather to eval the last model after training'})
    use_cpu: bool = field(default=False, metadata={"help": 'Run Accelerator on cpu for debug'})
    
@dataclass
class GenerationArguments:
    # For more hyperparameters checkhttps://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field( default=100, metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops" "if predict_with_generate is set."} )
    min_new_tokens : Optional[int] = field( default=None, metadata={"help": "Minimum number of new tokens to generate."} )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True

def get_model_and_tokenizer(args, only_tokenizer=False):
    model = None

    if args.bits == 8: args.quant_type = 'nf8'
    elif args.bits == 4: args.quant_type = 'nf4'
    else: args.quant_type = None

    if not only_tokenizer:
        quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                bnb_4bit_use_double_quant=args.double_quant, 
                bnb_4bit_quant_type=args.quant_type,
                bnb_8bit_compute_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type=args.quant_type
            )

        if args.zero_shot or args.one_shot:
                lora_config = None
                model = AutoModelForCausalLM.from_pretrained( args.model_path,
                    token=args.hf_access_token,
                    device_map={"":Accelerator().process_index} if args.device_map == 'accelerate' else args.device_map, #https://github.com/huggingface/peft/issues/629
                    max_memory = {i: f'{args.max_memory_MB}MB' for i in range(torch.cuda.device_count())},
                    quantization_config=quantization_config,
                    torch_dtype=(torch.bfloat16 if args.bf16 else (torch.float32 if args.fp16 else torch.float32)),
                    trust_remote_code=args.trust_remote_code,
                    attn_implementation=args.attn_implementation)
        else:
            if args.adapters_pretrained_path is None:
                model = AutoModelForCausalLM.from_pretrained( args.model_path,
                    token=args.hf_access_token,
                    quantization_config=quantization_config,
                    device_map={"":Accelerator().process_index} if args.device_map == 'accelerate' else args.device_map, #https://github.com/huggingface/peft/issues/629
                    max_memory = {i: f'{args.max_memory_MB}MB' for i in range(torch.cuda.device_count())},
                    torch_dtype=(torch.bfloat16 if args.bf16 else (torch.float32 if args.fp16 else torch.float32)),
                    trust_remote_code=args.trust_remote_code,
                    attn_implementation=args.attn_implementation)
                model = prepare_model_for_kbit_training(model)
                target_modules = find_all_linear_names(args, model) 
                logger.info(f'peft target_modules: {target_modules}')
                lora_config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type = TaskType.CAUSAL_LM,
                    inference_mode = not args.do_train,
                    target_modules = target_modules)
                model = get_peft_model(model, lora_config)
            else:
                model = AutoPeftModelForCausalLM.from_pretrained(args.adapters_pretrained_path,
                        token=args.hf_access_token,
                        quantization_config=quantization_config,
                        device_map={"":Accelerator().process_index} if args.device_map == 'accelerate' else args.device_map, #https://github.com/huggingface/peft/issues/629
                        max_memory = {i: f'{args.max_memory_MB}MB' for i in range(torch.cuda.device_count())},
                        torch_dtype=(torch.bfloat16 if args.bf16 else (torch.float32 if args.fp16 else torch.float32)),
                        trust_remote_code=args.trust_remote_code,
                        attn_implementation=args.attn_implementation)
        model.config.use_cache = False
    
    # the tokenizer of this model has not the correct chat template and falls back to default one
    #therefore i take the tokenizer of the parralel bigger model that should be the same and works
    if args.model_path == 'aaditya/Llama3-OpenBioLLM-8B': tokenizer_model_path = 'aaditya/Llama3-OpenBioLLM-70B'
    else: tokenizer_model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, token=args.hf_access_token, use_fast=args.use_fast_tokenizer)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    if not only_tokenizer:
        model.config.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token = tokenizer.eos_token
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.eos_token = tokenizer.eos_token
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.bos_token = tokenizer.bos_token

    return model, tokenizer

def unify_class_names_prior_json_load(prediction):    
    for c_idx, c in enumerate(CLASSES_NEW):
        if f'{c}:' in prediction: prediction = prediction.replace(f'{c}:',  f'"{c}":')
    return prediction

def get_label_from_decoded_str(decoded_str):
    pred_list = [0 for _ in CLASSES]
    decoded_str = unify_class_names_prior_json_load(decoded_str)
    num_missing_class = 0
    failed_json = False
    try:
        pred_dict = json.loads(decoded_str)
    except:
        failed_json = True
        
    if not failed_json:
        for c_idx, c in enumerate(CLASSES_NEW):
            if c in pred_dict: 
                if pred_dict[c] in [0, 1, True, False]:
                    pred_list[c_idx] = int(pred_dict[c])
                else:
                    failed_json = True
                    break
            else:
                num_missing_class += 1
                pred_list[c_idx] = int(not(y_true[-1][c_idx])) #treat as wrong predicitons for eval
        
    if failed_json:
        extracted_numbers = list(map(int, re.findall(r'\b[01]\b', re.sub(r'\b[2-9]\b', '1', prediction))))
        for c_idx, c in enumerate(CLASSES_NEW):
            if c in prediction and len(extracted_numbers)>c_idx:
                pred_list[c_idx] = extracted_numbers[c_idx]
            else:
                missing_class = True
                pred_list[c_idx] = int(not(y_true[-1][c_idx])) #treat as wrong predicitons for eval
                
    return pred_list, failed_json, num_missing_class

def get_metric_dict(y_true, y_pred):
    metric_dict = {}
    metric_dict = get_metric_dict_F1(y_true, y_pred, metric_dict)
    metric_dict = get_metric_dict_Acc(y_true, y_pred, metric_dict)
    metric_dict = get_metric_dict_SensSpec(y_true, y_pred, metric_dict)
    metric_dict = get_metric_dict_RecallPrec(y_true, y_pred, metric_dict)
    metric_dict = get_metric_dict_Numbers(y_true, y_pred, metric_dict)
    return metric_dict

def get_metric_dict_F1(y_true, y_pred, metric_dict={}):
    metric_dict['F1'] = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division = 0)  
    for c_idx, c in enumerate(CLASSES): metric_dict['F1_'+c] = f1_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
    return metric_dict

def get_metric_dict_Acc(y_true, y_pred, metric_dict={}):
    metric_dict['Acc'] = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None) 
    for c_idx, c in enumerate(CLASSES): metric_dict['Acc_'+c] =  metric_dict['Acc_'+c] = accuracy_score(y_true[:,c_idx], y_pred[:,c_idx])
    return metric_dict

def get_metric_dict_SensSpec(y_true, y_pred, metric_dict={}):
    for c_idx, c in enumerate(CLASSES): 
        metric_dict['Sensitivity/Recall_'+c] =  recall_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
        metric_dict['Specificity_'+c] =  recall_score(y_true=~(y_true[:,c_idx]>0), y_pred=~(y_pred[:,c_idx]>0), zero_division = 0)
    return metric_dict

def get_metric_dict_RecallPrec(y_true, y_pred, metric_dict={}): 
    for c_idx, c in enumerate(CLASSES): 
        metric_dict['Sensitivity/Recall_'+c] =  recall_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
        metric_dict['Precision_'+c] =  precision_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
    return metric_dict

def get_metric_dict_Numbers(y_true, y_pred, metric_dict={}): 
    metric_dict['Num_samples'] = len(y_true)  
    for c_idx, c in enumerate(CLASSES): metric_dict['Num_positive'+c] = int(y_true[:,c_idx].sum())
    return metric_dict

class Dummy_Trainer(object):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.compute_metrics = None

class Dummy_Accelerator(object):
    def __init__(self, model, tokenizer):
        self.is_main_process = True
        self.num_processes = 1

def full_evaluation(args, trainer, cur_dataset, eval_str):

    print(f'evaluating {eval_str} of len={len(cur_dataset)}')
    
    trainer.compute_metrics = None
    trainer.model.eval()
    torch.cuda.empty_cache()
   
    all_examples = []
    already_predicted = False
    if not already_predicted:
        def custom_collate(batch): return {key: [d[key] for d in batch] for key in batch[0]}
        trainer.tokenizer.padding_side ='left'
        trainer.tokenizer.pad_token_id = trainer.tokenizer.bos_token_id
        trainer.tokenizer.pad_token = trainer.tokenizer.bos_token
        trainer.model.config.pad_token_id = trainer.tokenizer.bos_token_id
        trainer.model.config.pad_token = trainer.tokenizer.bos_token

        def tokenize_func(input): return trainer.tokenizer(input, return_tensors='pt', padding=True)
        cur_dataset = cur_dataset.remove_columns([col for col in cur_dataset.column_names if col not in ['StudyAnonID', 'report', 'input', 'label']])
            
        start_time = time.time()
        if args.device_map == 'accelerate':

            num_processes = args.accelerator.num_processes  
            chunk_size = int(len(cur_dataset)/num_processes)
            while chunk_size*num_processes < len(cur_dataset): chunk_size+=1
            dataset_chunks = [cur_dataset.select(np.arange(i, i+chunk_size, 1)) if i+chunk_size<len(cur_dataset) else cur_dataset.select(np.arange(i, len(cur_dataset), 1)) for i in range(0, len(cur_dataset), chunk_size)]
            dataset_chunks_with_index = [ [idx, chunk] for idx, chunk in enumerate(dataset_chunks) ]
            
            with args.accelerator.split_between_processes(dataset_chunks_with_index) as dataset_chunk_with_index:
                gpu_index = dataset_chunk_with_index[0][0]
                dataset_chunk = dataset_chunk_with_index[0][1].map(tokenize_func, input_columns=['input'], batched=True, batch_size=args.per_device_generate_batch_size)
                dataloader = DataLoader( dataset_chunk, batch_size=args.per_device_generate_batch_size, shuffle=False, num_workers=2, drop_last=False, collate_fn=custom_collate)
                with open(os.path.join(args.output_dir, f'{eval_str}_predictions_{gpu_index}.jsonl'), 'w') as fout:
                    for batch_idx, batch in enumerate(dataloader):
                        print(f"Batch {batch_idx}/{len(dataloader)}")
                        with torch.no_grad():
                            preds = trainer.model.generate( input_ids = torch.tensor(batch['input_ids']).to('cuda'), 
                                attention_mask = torch.tensor(batch['attention_mask']).to('cuda'),
                                generation_config=args.generation_config,
                                pad_token_id=trainer.tokenizer.pad_token_id,
                                eos_token_id=trainer.tokenizer.eos_token_id )
                        decoded_preds = trainer.tokenizer.batch_decode( preds, skip_special_tokens=True, clean_up_tokenization_spaces=True )
                        preds = None
            
                        for preds_idx, decoded_pred in enumerate(decoded_preds):
                            example = {}
                            example['y_true'] = ast.literal_eval(batch['label'][preds_idx]) if isinstance(batch['label'][preds_idx], str) else batch['label'][preds_idx] 
                            if decoded_pred[-1] != '}': decoded_pred += '}'
                            example['y_pred'], example['failed_json_load'], example['num_missing_classes'] = get_label_from_decoded_str(decoded_pred)
                            example['y_pred'] = new_to_old_label(example['y_pred'])
                            decoded_pred = decoded_pred[:decoded_pred.rfind('}')]
                            decoded_pred = decoded_pred[decoded_pred.rfind('{'):]
                            example['prediction'] = decoded_pred
                            example['report'] = batch['report'][preds_idx]
                            example['StudyAnonID'] = batch['StudyAnonID'][preds_idx]
                            fout.write(json.dumps(example) + '\n')

            args.accelerator.wait_for_everyone()
            if args.accelerator.is_main_process:
                
                for gpu_index in range(num_processes):
                    cur_file_path = os.path.join(args.output_dir, f'{eval_str}_predictions_{gpu_index}.jsonl')
                    with open(os.path.join(args.output_dir, f'{eval_str}_predictions_{gpu_index}.jsonl'), 'r') as json_file:
                        all_examples += [json.loads(line) for line in json_file]
                    if os.path.isfile(cur_file_path): os.remove(cur_file_path)
             
                with open(os.path.join(args.output_dir, f'{eval_str}_predictions.jsonl'), 'w') as fout:
                    for example in all_examples: fout.write(json.dumps(example) + '\n')
    
        else:
            
            dataset_tokenized = cur_dataset.map(tokenize_func, input_columns=['input'], batched=True, batch_size=args.per_device_generate_batch_size)
            dataloader = DataLoader( dataset_tokenized, batch_size=args.per_device_generate_batch_size, shuffle=False, num_workers=2, drop_last=False, collate_fn=custom_collate)
            start_time = time.time()
            with open(os.path.join(args.output_dir, f'{eval_str}_predictions.jsonl'), 'w') as fout:
                for batch_idx, batch in enumerate(dataloader):
                    print(f"Batch {batch_idx}/{len(dataloader)}")
                    with torch.no_grad():
                        preds = trainer.model.generate( input_ids = torch.tensor(batch['input_ids']).to('cuda'), 
                            attention_mask = torch.tensor(batch['attention_mask']).to('cuda'),
                            generation_config=args.generation_config,
                            pad_token_id=trainer.tokenizer.pad_token_id,
                            eos_token_id=trainer.tokenizer.eos_token_id )
                    decoded_preds = trainer.tokenizer.batch_decode( preds, skip_special_tokens=False, clean_up_tokenization_spaces=True )
                    preds = None

                    for preds_idx, decoded_pred in enumerate(decoded_preds):
                        example = {}
                        example['y_true'] = ast.literal_eval(batch['label'][preds_idx]) if isinstance(batch['label'][preds_idx], str) else batch['label'][preds_idx] 
                        if decoded_pred[-1] != '}': decoded_pred += '}'
                        example['y_pred'], example['failed_json_load'], example['num_missing_classes'] = get_label_from_decoded_str(decoded_pred)
                        example['y_pred'] = new_to_old_label(example['y_pred'])
                        decoded_pred = decoded_pred[:decoded_pred.rfind('}')]
                        decoded_pred = decoded_pred[decoded_pred.rfind('{'):]
                        example['prediction'] = decoded_pred
                        example['report'] = batch['report'][preds_idx]
                        example['StudyAnonID'] = batch['StudyAnonID'][preds_idx]
                        fout.write(json.dumps(example) + '\n')             
        with open(join(args.output_dir, f'inference_time_in_sec_{eval_str}.txt'),'w', encoding='utf-8' ) as file: print(f"{time.time() - start_time:.0f}", file=file)                 
    
    if args.accelerator.is_main_process:
        all_examples = []; y_true = []; y_pred = []
        with open(os.path.join(args.output_dir, f'{eval_str}_predictions.jsonl'), 'r') as json_file:
            all_examples = [json.loads(line) for line in json_file]
        for example in all_examples:
            y_pred.append( ast.literal_eval(example['y_pred']) if isinstance(example['y_pred'], str) else example['y_pred']  )
            y_true.append( ast.literal_eval(example['y_true']) if isinstance(example['y_true'], str) else example['y_true']  )

        report = classification_report(np.array(y_true), np.array(y_pred), target_names=CLASSES, digits=3)
        metric_dict = get_metric_dict(np.array(y_true), np.array(y_pred))
        with open(join(args.output_dir, f'classificationReport_{eval_str}.txt'),'w', encoding='utf-8' ) as file: print(report, file=file) 
        with open(join(args.output_dir, f'results_{eval_str}.json'), 'w', encoding='utf-8') as file:json.dump(metric_dict, file, indent=2)
        with open(join(args.output_dir, f'results_{eval_str}.pkl'), 'wb') as file: pickle.dump([metric_dict, report, cur_dataset['StudyAnonID'], y_true, y_pred, CLASSES], file)
   
def main():

    hfparser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments)) 
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    assert data_args.dataset_path is not None, "No dataset was given."
    if training_args.zero_shot or training_args.one_shot: training_args.do_train = False
    if training_args.disable_tqdm: datasets.disable_progress_bar()
    generation_config = GenerationConfig(**vars(generation_args)) 
    training_args.generation_config = generation_config
    training_args.hub_token=model_args.hf_access_token
    training_args.push_to_hub=False
    args = argparse.Namespace( **vars(model_args), **vars(data_args), **vars(training_args),  **vars(generation_args) ) 
    
    if args.device_map == 'accelerate': args.accelerator = Accelerator(cpu=args.use_cpu)
    else: args.accelerator = Dummy_Accelerator()

    login(token=args.hf_access_token)
    set_seed(args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)

    chat_template = None
    args.use_fast_tokenizer = True
    if np.any([x in args.model_path for x in ['Llama-2', 'Mixtral-', 'Mistral-', 'meditron']]): 
        args.response_template = f'[/INST]'
    elif 'phi-3' in args.model_path: 
        args.response_template = f'<|assistant|>'
    elif 'vicuna-' in args.model_path:  
        #https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/discussions/1
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '\n '}}{% elif message['role'] == 'system' %}{{ 'SYSTEM: '  + message['content'] + '\n ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"
        args.use_fast_tokenizer = False
        args.response_template = 'ASSISTANT:' 
    elif 'BioMedLM' in args.model_path:  
        # https://crfm.stanford.edu/2022/12/15/biomedlm.html
        #<Context token> Text of context … <Question token> Text of question <Answer token> 
        chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<Context token> ' + message['content'] + '\n ' }}{% elif message['role'] == 'system' %}{{ '<System token> '  + message['content'] + '\n '}}{% elif message['role'] == 'assistant' %}{{ ' <Answer token> ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"
        args.response_template = '<Answer token>' 
    elif 'Llama-3' in args.model_path or 'Llama3' in args.model_path: 
        #https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/14
        args.response_template = '<|start_header_id|>assistant<|end_header_id|>'
    elif 'gemma' in args.model_path: 
        args.response_template = '<start_of_turn>model'
    elif 'dbrx' in args.model_path : 
        args.response_template = '<|im_start|> assistant'
    elif 'c4ai-command-r' in args.model_path : 
        args.response_template = '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'
    elif 'hessianai' in args.model_path: 
        args.response_template = '<|im_start|>assistant'     
    else: assert False, f'Response template for {args.model_path} not implemented'

    logger.info(args, main_process_only=True)
    model, tokenizer = get_model_and_tokenizer(args)
    if chat_template is not None: tokenizer.chat_template = chat_template

    def load_dataset(args, dataset_path, tokenizer):

        one_shot_example = f"\n{ONE_SHOT_EXAMPLE}" if args.one_shot else ""
        tipp = f"\n{TIPPS}" if args.with_tipps else ""
        def create_prompts(example):
            messages = [
                {'role': 'user', 'content': f'{SYSTEM_PROMPT}{USER_PROMPT}{tipp}{one_shot_example}{PRE_REPORT_PROMPT}{example["report"]} '},
                {'role': 'assistant', 'content': f' {label_to_output_seq(example["label"])}'} ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            return {'input': text[:text.rfind(PRE_GIVEN_OUTPUT)+len(PRE_GIVEN_OUTPUT)],
                    'output': label_to_output_seq(example["label"]).replace(PRE_GIVEN_OUTPUT, ""),
                    'text': text} 

        dataset = load_from_disk(dataset_path)
        dataset = dataset.map(create_prompts)

        for split in dataset:
            for input, output in zip( dataset[split]['input'], dataset[split]['output']):
                seq_length = len(tokenizer.encode(f'{tokenizer.bos_token}{input}'))
                if seq_length > args.max_seq_length_input: args.max_seq_length_input = seq_length
                seq_length = len(tokenizer.encode(f'{output}{tokenizer.bos_token}'))
                if seq_length > args.max_seq_length_output: args.max_seq_length_output = seq_length
            
        return dataset

    if args.do_train:

        dataset = load_dataset(args, args.dataset_path, tokenizer)
        num_train = len(dataset['train']) 
        if args.total_train_batch_size > num_train:
            args.total_train_batch_size = training_args.total_train_batch_size = num_train
        if args.total_train_batch_size < torch.cuda.device_count()*args.per_device_train_batch_size: 
            args.per_device_train_batch_size = training_args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = training_args.gradient_accumulation_steps = \
            int(args.total_train_batch_size / ( torch.cuda.device_count()*args.per_device_train_batch_size))
        logger.info(f'total_train_batch_size={args.total_train_batch_size}', main_process_only=True)
        logger.info(f'per_device_train_batch_size={args.per_device_train_batch_size}', main_process_only=True)
        logger.info(f'gradient_accumulation_steps={args.gradient_accumulation_steps}', main_process_only=True)

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template = args.response_template,
            tokenizer=tokenizer)

        # This here is uggly because of tfl updates requiring SFTConfig for SFTTrainer
        # TODO make the training_args inputted by arg parsing to arguments that fit in SFTConfig
        
        training_args.remove_unused_columns=True
        sft_config = SFTConfig(
            optim=training_args.optim, 
            max_steps=training_args.max_steps,
            weight_decay=training_args.weight_decay,
            learning_rate=training_args.learning_rate,
            max_grad_norm=training_args.max_grad_norm,
            gradient_checkpointing=training_args.gradient_checkpointing,
            do_train=training_args.do_train ,
            lr_scheduler_type=training_args.lr_scheduler_type ,
            warmup_ratio=training_args.warmup_ratio, 
            group_by_length=training_args.group_by_length,
            evaluation_strategy=training_args.evaluation_strategy,
            metric_for_best_model=training_args.metric_for_best_model,
            eval_steps=training_args.eval_steps,
            eval_accumulation_steps=training_args.eval_accumulation_steps,
            save_steps=training_args.save_steps,
            logging_steps=training_args.logging_steps,
            load_best_model_at_end=training_args.load_best_model_at_end,
            save_total_limit=training_args.save_total_limit,
            dataloader_num_workers=training_args.dataloader_num_workers,
            bf16=training_args.bf16,
            fp16=training_args.fp16,
            remove_unused_columns = training_args.remove_unused_columns,
            ddp_find_unused_parameters=training_args.ddp_find_unused_parameters,
            disable_tqdm=training_args.disable_tqdm,
            max_seq_length=args.max_seq_length_input+args.max_seq_length_output,
            dataset_text_field="text",
            output_dir=training_args.output_dir,
            neftune_noise_alpha=args.neftune_noise_alpha # Enhance model’s performances using NEFTune #/arxiv.org/abs/2310.05914 
        )
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset = dataset['train'].remove_columns([col for col in dataset['train'].column_names if col != 'text']),
            eval_dataset = dataset['eval'].remove_columns([col for col in dataset['train'].column_names if col != 'text']),
            args=sft_config,
            data_collator=data_collator,
        )

        if args.early_stopping:
            trainer.add_callback(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience, 
                early_stopping_threshold=0.0))

        torch.cuda.empty_cache()
        logger.info( 'Starting training ...' , main_process_only=True)
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # I only save the adapters to save disk space TODO save full models with adapter merging for faster inference
        '''
        trainer.save_model(script_args.training_args.output_dir)
        output_dir = os.path.join(script_args.training_args.output_dir, "final_checkpoint")
        trainer.model.save_pretrained(output_dir)
        # Free memory for merging weights
        del base_model
        if is_xpu_available(): torch.xpu.empty_cache()
        else: torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()
        output_merged_dir = os.path.join(script_args.training_args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        '''

    else:
        trainer = Dummy_Trainer( model=model, tokenizer=tokenizer )

    folders_in_output = glob(join(args.output_dir,'*/'))
    folders_in_output = [x for x in folders_in_output if 'checkpoint-' in x.split(os.sep)[-2]]
    if len(folders_in_output) > 0:
        steps = [ int(x[x.find('checkpoint-')+len('checkpoint-'):-1]) for x in folders_in_output]
        for folder, step in zip(folders_in_output,steps):
            with open(folder+'steps.txt', 'w') as text_file: print(str(step), file=text_file)
        folders_in_output = [x for _,x in sorted(zip(steps,folders_in_output))]
        steps.sort()

    #Eval BEST LAST MODEL after train
    if args.eval_best_model or (args.zero_shot or args.one_shot): 
        if not (args.zero_shot or args.one_shot) and not args.load_best_model_at_end:  
            args.adapters_pretrained_path = folders_in_output[0]
            model = None; trainer.model = None
            trainer.model, tokenizer = get_model_and_tokenizer(args)
        if args.do_train and args.full_eval_valid_set: full_evaluation(args, trainer, dataset['eval'], eval_str = 'valid_best_model')
        if args.dataset_path_test is not None: 
            dataset_test  = load_dataset(args, args.dataset_path_test, tokenizer)
            full_evaluation(args, trainer, dataset_test['eval'],  eval_str = 'test_best_model')
        if args.dataset_path_test2 is not None: 
            dataset_test2 = load_dataset(args, args.dataset_path_test2, tokenizer)
            full_evaluation(args, trainer, dataset_test2['eval'], eval_str = 'test2_best_model')
        if args.dataset_path_predict is not None: 
            dataset_predict = load_dataset(args, args.dataset_path_predict, tokenizer)
            full_evaluation(args, trainer, dataset_predict['eval'],  eval_str = 'predict_best_model')

    #Eval LAST MODEL after train  
    if args.eval_last_model and not (args.zero_shot or args.one_shot):  
        if args.load_best_model_at_end:
            args.adapters_pretrained_path = folders_in_output[-1] #last checkpoint ist last model, i rename folders at the end as else problems with accelerator and parallel
            model = None; trainer.model = None
            trainer.model, tokenizer = get_model_and_tokenizer(args)
        if args.do_train and args.full_eval_valid_set: full_evaluation(args, trainer, dataset['eval'], eval_str = 'valid_last_model')
        if args.dataset_path_test is not None: 
            dataset_test  = load_dataset(args, args.dataset_path_test, tokenizer)
            full_evaluation(args, trainer, dataset_test['eval'], eval_str = 'test_last_model')
        if args.dataset_path_test2 is not None: 
            dataset_test2 = load_dataset(args, args.dataset_path_test2, tokenizer)
            full_evaluation(args, trainer, dataset_test2['eval'], eval_str = 'test2_last_model')
        if args.dataset_path_predict is not None: 
            dataset_predict = load_dataset(args, args.dataset_path_predict, tokenizer)
            full_evaluation(args, trainer, dataset_predict['eval'], eval_str = 'predict_last_model')
 
    if args.device_map == 'accelerate': args.accelerator.wait_for_everyone()
    del trainer
    if args.do_train: del dataset
    if args.dataset_path_test: del dataset_test
    if args.dataset_path_test2: del dataset_test2
    if args.dataset_path_predict: del dataset_predict
    torch.cuda.empty_cache()

    #if args.accelerator.is_main_process:
    #    os.rename(folders_in_output[-1], folders_in_output[-1].replace(os.sep+'checkpoint-'+str(steps[-1])+os.sep, os.sep+'last_model'+os.sep))
    #    os.rename(folders_in_output[0], folders_in_output[0].replace(os.sep+'checkpoint-'+str(steps[0])+os.sep, os.sep+'best_model'+os.sep))

if __name__ == "__main__":
    main()
    
    
    
    
