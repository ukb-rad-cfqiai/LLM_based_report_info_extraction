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

from utils import get_label_from_decoded_str, get_metric_dict
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
import pandas as pd
from accelerate import Accelerator, PartialState, init_empty_weights, load_checkpoint_and_dispatch
import os, argparse, torch, json, re,  pickle, ast, time
from torch.utils.data import DataLoader, default_collate
from collections import OrderedDict
from contextlib import nullcontext
join = os.path.join
os.environ["TOKENIZERS_PARALLELISM"] = "true" #https://github.com/huggingface/transformers/issues/5486
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"  #https://discuss.huggingface.co/t/get-using-the-call-method-is-faster-warning-with-datacollatorwithpadding/23924/5
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  #https://www.youtube.com/watch?v=gXDsVcY8TXQ
from glob import glob
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
from huggingface_hub import login, constants
from transformers.pipelines.pt_utils import KeyDataset
from transformers import (
    LogitsProcessor,
    StoppingCriteria,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerControl, 
    TrainerState,
    EarlyStoppingCallback,
    set_seed,
    BitsAndBytesConfig,
    GenerationConfig,
    HfArgumentParser,
    TrainerCallback )

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    adapters_pretrained_path: Optional[str] = field(default=None)
    hf_access_token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)

@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    dataset_path: str = field(default=None)
    dataset_path_test: str = field(default=None)
    dataset_path_test2: str = field(default=None)
    dataset_path_predict: str = field(default=None)

@dataclass
class TrainingArguments(TrainingArguments):
    # General training settings
    output_dir: str = field(default='./default_training_output')
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    max_steps: int = field(default=256)
    total_train_batch_size: int = field(default=512)
    eval_accumulation_steps: int = field(default=1)

    # Model configuration
    zero_shot: bool = field(default=False)
    one_shot: bool = field(default=False)
    use_lora: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: float = field(default=16)
    lora_dropout: float = field(default=0.05)
    bits: int = field(default=4)
    double_quant: bool = field(default=False)
    
    # Optimization settings
    optim: str = field(default='adamw_torch')
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default='constant')
    warmup_ratio: float = field(default=0.03)
    
    # Training process
    gradient_checkpointing: bool = field(default=False)
    max_grad_norm: float = field(default=0.3)
    group_by_length: bool = field(default=True)
    
    # Evaluation and logging
    eval_strategy: str = field(default='steps')
    eval_steps: int = field(default=10)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=10)
    metric_for_best_model: str = field(default='eval_loss')
    
    # Model saving and loading
    save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    
    # Early stopping
    early_stopping: bool = field(default=False)
    early_stopping_patience: int = field(default=10)
    
    # Data processing
    max_seq_length_input: int = field(default=256)
    max_seq_length_output: int = field(default=64)
    remove_unused_columns: bool = field(default=False)
    
    # Hardware utilization
    max_memory_MB: int = field(default=80000)
    dataloader_num_workers: int = field(default=4)
    dataloader_pin_memory: bool = field(default=True)
    
    # Mixed precision and distributed training
    mixed_precision: str = field(default='no')
    ddp_find_unused_parameters: bool = field(default=False)
    device_map: str = field(default='fsdp')
    
    # Additional features
    neftune_noise_alpha: float = field(default=5.0)
    attn_implementation: str = field(default='sdpa')
    
    # Evaluation modes
    full_eval_valid_set: bool = field(default=True)
    eval_best_model: bool = field(default=True)
    eval_last_model: bool = field(default=True)

    # Inference settings
    save_full_prompts_with_output: bool = field(default=False)
    compute_classification_metrics_after_predicting: bool = field(default=False)
    overwrite_already_done_predicitons: bool = field(default=False)
    per_device_generate_batch_size: int = field(default=8)
  
    # Miscellaneous
    report_to: str = field(default=None)
    use_cpu: bool = field(default=False)
    disable_tqdm: bool = field(default=False)
    
@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field( default=100 )
    min_new_tokens : Optional[int] = field( default=None)
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)

def set_special_tokens_after_model_instantiate(args, model, tokenizer):
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token = tokenizer.pad_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.eos_token = tokenizer.eos_token
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.bos_token = tokenizer.bos_token
    return model

def get_model(args, instantiate=True, adapters_pretrained_path=None):
    
    if args.bits == 8: args.quant_type = 'nf8'
    elif args.bits == 4: args.quant_type = 'nf4'
    else: args.quant_type = None
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    quantization_config = None
    if args.bits <= 8:
        quantization_config = None if np.any( [x in args.model_path for x in ['4bit', '8bit', 'FP8']] ) else \
        BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage="uint8" if args.bits == 4 else None, #torch_dtype,
            bnb_4bit_use_double_quant=args.double_quant, 
            bnb_4bit_quant_type=args.quant_type,

            load_in_8bit=args.bits == 8,
            bnb_8bit_compute_dtype=torch_dtype,
            bnb_8bit_quant_storage=torch_dtype,
            bnb_8bit_use_double_quant=args.double_quant, 
            bnb_8bit_quant_type=args.quant_type,

            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_skip_modules=None,
        ) 

    print(f'quantization_config: {quantization_config}')
    if args.device_map == 'fsdp':
        if not args.do_train: device_map = 'auto'
        else: device_map = None # https://www.youtube.com/watch?v=gXDsVcY8TXQ (according to trelis shall not net set for fsdp and fully handled by accelerate )
    elif args.device_map == 'ddp': device_map={'': PartialState().process_index} 
    else: device_map = args.device_map

    model_kwargs_raw = dict( 
            token=args.hf_access_token,
            quantization_config=quantization_config,
            device_map=device_map, 
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
            use_cache=False if args.gradient_checkpointing else True )
    model_kwargs = {k: v for k, v in model_kwargs_raw.items() if v is not None} #remove None's

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type = TaskType.CAUSAL_LM,
                inference_mode = not args.do_train,
                target_modules = "all-linear")
    
    model = None
    if instantiate: 
        if args.use_lora and not adapters_pretrained_path is None: 
            AutoModelClass = AutoPeftModelForCausalLM
            model_path = adapters_pretrained_path
        else: 
            AutoModelClass = AutoModelForCausalLM
            model_path = args.model_path 

        print(f"instantiating {AutoModelClass} {model_path}")
        model = AutoModelClass.from_pretrained(model_path, **model_kwargs) 
        if args.use_lora and adapters_pretrained_path is None and args.do_train: 
            model = get_peft_model( model, peft_config )
            
    return model, model_kwargs, peft_config 

def get_tokenizer(args):
    # the tokenizer of this model has not the correct chat template and falls back to default one
    #therefore i take the tokenizer of the parralel bigger model that should be the same and works
    if args.model_path == 'aaditya/Llama3-OpenBioLLM-8B': tokenizer_model_path = 'aaditya/Llama3-OpenBioLLM-70B'
    else: tokenizer_model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path, token=args.hf_access_token, use_fast=args.use_fast_tokenizer)#, add_prefix_space=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    return  tokenizer

# Global dictionary to hold the configuration
prompt_config = {}
def load_prompt_config():
    global prompt_config  # Declare prompt_config as global
    with open('prompt_config.json', 'r', encoding='utf-8') as file:
        prompt_config = json.load(file)

def label_to_output_seq(label):
    return json.dumps({cls: lbl for cls, lbl in zip(prompt_config['CLASSES'], label)}, indent=2, ensure_ascii=False)

class Dummy_Accelerator(object):
    def __init__(self, is_main_process=True, num_processes=1):
        self.is_main_process = True
        self.num_processes = 1
        self.print = print
    def wait_for_everyone(self):
        return None

class Dummy_Trainer(object):
    def __init__(self, model=None, tokenizer=None, accelerator=Dummy_Accelerator()):
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator

def predict_dataset(args, trainer, tokenizer, dataset, classes, eval_str):

    trainer.accelerator.print(f'{eval_str}: Starting prediciton function...')
    trainer.compute_metrics = None
    trainer.model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
 
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['StudyAnonID', 'report', 'input', 'label']])
    output_file = os.path.join(args.output_dir, f'{eval_str}_predictions.jsonl')   
    already_done = False; already_predicted_backup_file = None
    if args.overwrite_already_done_predicitons:
        trainer.accelerator.print(f'{eval_str}: I am overwriting already done predicitons (if there are any).')
        with open(output_file, 'w') as fout: pass
    else:
        already_predicted_files = [f for f in glob(os.path.join(args.output_dir, '*')) 
                            if re.match(rf'{eval_str}_predictions_gpu\d+\.jsonl', os.path.basename(f))] 
        already_predicted_backup_file = output_file.replace('.jsonl', '_backup_already_pred.jsonl')
        if os.path.exists(already_predicted_backup_file): already_predicted_files.append(already_predicted_backup_file)      
        if os.path.exists(output_file): already_predicted_files.append(output_file)
        all_examples = []
        for cur_file_path in already_predicted_files:
            with open(cur_file_path, 'r') as json_file:
                all_examples += [json.loads(line) for line in json_file]

        len_dataset_prior_filter = len(dataset)
        if len(all_examples)>0:
            study_anon_ids_to_remove = {example['StudyAnonID'] for example in all_examples}
            dataset = dataset.filter(lambda example: example['StudyAnonID'] not in study_anon_ids_to_remove)
            if len(dataset)==0: 
                trainer.accelerator.print(f'{eval_str}: All predictions were already there. I am finished before i startet.')
                already_done = True
            elif len_dataset_prior_filter > len(dataset): 
                trainer.accelerator.print(f'{eval_str}: Predicitons {len_dataset_prior_filter-len(dataset)}/{len_dataset_prior_filter} already found and skipping. ')
            else: trainer.accelerator.print(f'{eval_str}: WARNING: Found {len(all_examples)} predicitons already done that however are not in your dataset! Are you shure the output path is correct? I will keep the unknown predicitons.')
        else: 
            trainer.accelerator.print(f'{eval_str}: Found no already generated predicitons. Creating predicitons for all {len(dataset)} texts.')

        if not already_done:
            trainer.accelerator.wait_for_everyone()
            if trainer.accelerator.is_main_process:
                if len(all_examples)>0: 
                    trainer.accelerator.print(f'{eval_str}: Found already predicted examples and gathered them in {already_predicted_backup_file}')
                    print(f'len(all_examples) {len(all_examples)}')
                    with open(already_predicted_backup_file, 'w') as fout: 
                        for example in all_examples: fout.write(json.dumps(example) + '\n')

    trainer.accelerator.wait_for_everyone()
    if not already_done:
        def custom_collate(batch): return {key: [d[key] for d in batch] for key in batch[0]}
        trainer.tokenizer.padding_side ='left'
        trainer.tokenizer.pad_token_id = trainer.tokenizer.bos_token_id
        trainer.tokenizer.pad_token = trainer.tokenizer.bos_token
        args.generation_config.pad_token_id = tokenizer.eos_token_id 
        args.generation_config.pad_token = trainer.tokenizer.bos_token
        trainer.model.config.pad_token_id = trainer.tokenizer.bos_token_id
        trainer.model.config.pad_token = trainer.tokenizer.bos_token
        args.generation_config.stop_strings = ['}'] #the tokenizer could "swallow" }

        def tokenize_func(input): return trainer.tokenizer(input, return_tensors='pt', padding=True) #, , add_special_tokens=False

        if isinstance(trainer.accelerator, Dummy_Accelerator): num_processes = 1
        else: num_processes = PartialState().num_processes   
        chunk_size = int(len(dataset)/num_processes)
        while chunk_size*num_processes < len(dataset): chunk_size+=1
        dataset_chunks = [dataset.select(np.arange(i, i+chunk_size, 1)) if i+chunk_size<len(dataset) else dataset.select(np.arange(i, len(dataset), 1)) for i in range(0, len(dataset), chunk_size)]
        dataset_chunks_with_index = [ [idx, chunk] for idx, chunk in enumerate(dataset_chunks) ]
        
        from contextlib import contextmanager
        @contextmanager
        def accelerator_split_or_list(accelerator, data):
            if not isinstance(accelerator, Dummy_Accelerator):
                with accelerator.split_between_processes(data) as split_data:
                    yield split_data
            else:
                yield list(data)
                
        nvidia_smi.nvmlInit()
        if trainer.accelerator.is_main_process: start_time = time.time()
        with accelerator_split_or_list(trainer.accelerator, dataset_chunks_with_index) as dataset_chunk_with_index:
            gpu_index = dataset_chunk_with_index[0][0]
            model_dtype = next(trainer.model.parameters()).dtype
            dataset_chunk = dataset_chunk_with_index[0][1].map(tokenize_func, input_columns=['input'], batched=True, batch_size=args.per_device_generate_batch_size)
            dataloader = DataLoader( dataset_chunk, batch_size=args.per_device_generate_batch_size, shuffle=False, num_workers=2, drop_last=False, collate_fn=custom_collate)
            with open(output_file.replace('.jsonl',f'_gpu{gpu_index}.jsonl'), 'w') as fout:
                for batch_idx, batch in enumerate(dataloader):
                    print(f"Batch {batch_idx+1}/{len(dataloader)}")
                    with torch.no_grad():

                        input_ids = torch.tensor(batch['input_ids'])
                        attention_mask = torch.tensor(batch['attention_mask'])
                        if args.device_map == 'auto': 
                            input_ids = input_ids.to('cuda')
                            attention_mask = attention_mask.to('cuda')
                        else:
                            input_ids = input_ids.to(device=trainer.model.device)
                            attention_mask = attention_mask.to(device=trainer.model.device)

                        with torch.amp.autocast('cuda', dtype=model_dtype):
                            preds = trainer.model.generate( input_ids = input_ids, 
                                attention_mask = attention_mask,
                                generation_config=args.generation_config,
                                pad_token_id=trainer.tokenizer.pad_token_id,
                                eos_token_id=trainer.tokenizer.eos_token_id,
                                tokenizer=trainer.tokenizer
                                )

                    if args.save_full_prompts_with_output: 
                        raw_decoded_input_ids = trainer.tokenizer.batch_decode( input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False )    
                        raw_decoded_preds = trainer.tokenizer.batch_decode(         preds, skip_special_tokens=False, clean_up_tokenization_spaces=False )
                    else:
                        raw_decoded_input_ids = [None for _ in decoded_preds]    
                        raw_decoded_preds =  [None for _ in decoded_preds]    
                    
                    decoded_preds = trainer.tokenizer.batch_decode( preds, skip_special_tokens=True, clean_up_tokenization_spaces=True )
                    preds = None

                    for preds_idx, (decoded_pred, raw_decoded_pred, raw_decoded_input_id) in enumerate(zip(decoded_preds, raw_decoded_preds, raw_decoded_input_ids)):
                        example = {}
                        example['y_true'] = ast.literal_eval(batch['label'][preds_idx]) if isinstance(batch['label'][preds_idx], str) else batch['label'][preds_idx] 
                        decoded_pred = decoded_pred[:decoded_pred.rfind('}')]
                        decoded_pred = decoded_pred[decoded_pred.rfind('{'):]
                        if decoded_pred[-1] != '}': decoded_pred += '}'
                        example['y_pred'], example['failed_json_load'], example['num_missing_classes'] = get_label_from_decoded_str(decoded_pred, example['y_true'])
                        example['prediction'] = decoded_pred
                        if args.save_full_prompts_with_output: # this is for debugging to see what the actual model gets for generate and what it does with it
                            example['raw_input'] = raw_decoded_input_id
                            example['raw_prediciton'] = raw_decoded_pred[len(raw_decoded_input_id):]
                        example['report'] = batch['report'][preds_idx]
                        example['StudyAnonID'] = batch['StudyAnonID'][preds_idx]
                        fout.write(json.dumps(example) + '\n')

        trainer.accelerator.wait_for_everyone()
        if trainer.accelerator.is_main_process:
            inference_metrics = { 'elapsed_seconds': f'{time.time() - start_time:.0f}',
                      'train_peak_gpu_mem_allocated_GB': torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1024**3 }
            with open(join(args.output_dir, f'inference_metrics_{eval_str}.txt'), 'w') as file: json.dump(inference_metrics, file, indent=4)
        
            trainer.accelerator.print(f'{eval_str}: Inference took {time.time() - start_time:.0f} seconds.')
            all_examples = []
            if not args.overwrite_already_done_predicitons:
                if os.path.exists(already_predicted_backup_file):
                    with open(already_predicted_backup_file, 'r') as json_file:
                        all_examples += [json.loads(line) for line in json_file]
                    if os.path.isfile(already_predicted_backup_file): os.remove(already_predicted_backup_file)
            for gpu_index in range(num_processes):
                cur_file_path = output_file.replace('.jsonl',f'_gpu{gpu_index}.jsonl')
                if os.path.isfile(cur_file_path): 
                    with open(cur_file_path, 'r') as json_file:
                        all_examples += [json.loads(line) for line in json_file]
                    os.remove(cur_file_path)
            trainer.accelerator.print(f'{eval_str}: Writing predicitons in {output_file}.')
            with open(output_file, 'w') as fout:
                for example in all_examples: fout.write(json.dumps(example) + '\n') 

    if args.compute_classification_metrics_after_predicting and trainer.accelerator.is_main_process:   
        trainer.accelerator.print(f'{eval_str}: Calculating evaluation metrics.')
        y_true = []; y_pred = []
        trainer.accelerator.print(f'len(all_examples) {len(all_examples)} ')
        for example in all_examples:
            y_pred.append( ast.literal_eval(example['y_pred']) if isinstance(example['y_pred'], str) else example['y_pred']  )
            y_true.append( ast.literal_eval(example['y_true']) if isinstance(example['y_true'], str) else example['y_true']  )
        report = classification_report(np.array(y_true), np.array(y_pred), target_names=classes, digits=3)
        metric_dict = get_metric_dict(np.array(y_true), np.array(y_pred))
        with open(join(args.output_dir, f'classificationReport_{eval_str}.txt'),'w', encoding='utf-8' ) as file: print(report, file=file) 
        with open(join(args.output_dir, f'results_{eval_str}.json'), 'w', encoding='utf-8') as file:json.dump(metric_dict, file, indent=2)
        with open(join(args.output_dir, f'results_{eval_str}.pkl'), 'wb') as file: pickle.dump([metric_dict, report, dataset['StudyAnonID'], y_true, y_pred, classes], file)
    trainer.accelerator.wait_for_everyone()

def main():

    hfparser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments)) 
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    #TODO open Github issue for this
    assert training_args.report_to!='tensorboard', 'tensorboard is buggy as args are written as json into the tb and the quntization config in model_init_kwargs is not json serializable '
    
    if training_args.device_map=='fsdp':
        training_args.fsdp = "full_shard auto_wrap offload", #offload # remove offload with enough gpu memory
        training_args.fsdp_config = {  "activation_checkpointing ": "true" if training_args.gradient_checkpointing else "false",
                                       "backward_prefetch": "backward_pre",
                                       "forward_prefetch": "false",
                                       "use_orig_params": "false",
                                       "fsdp_state_dict_type": "sharded_state_dict",
                                       "fsdp_cpu_ram_efficient_loading": "false",
                                       "fsdp_sync_module_states": "false",
                                        }  
    else:
        training_args.fsdp = ""
        training_args.fsdp_config = None
    
    if training_args.zero_shot or training_args.one_shot: 
        training_args.do_train = False
        training_args.use_lora = False
        
    if training_args.disable_tqdm: datasets.disable_progress_bar()
    generation_config = GenerationConfig(**vars(generation_args)) 
    training_args.generation_config = generation_config
    training_args.hub_token=model_args.hf_access_token
    training_args.push_to_hub=False
    args = argparse.Namespace( **vars(model_args), **vars(data_args), **vars(training_args),  **vars(generation_args) ) 
    args.fp16 = args.mixed_precision == 'fp16'
    args.bf16 = args.mixed_precision == 'bf16'

    login(token=args.hf_access_token)
    set_seed(args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Load chat templates from JSON file and find the appropriate template based on the model path (case-insensitive)
    with open('chat_templates.json', 'r') as f: chat_templates = json.load(f)
    args.use_fast_tokenizer = True
    chat_template = "default"
    for model_substring, template_info in chat_templates.items():
        # Convert both model_substring and args.model_path to lowercase for case-insensitive comparison
        if model_substring.lower() in args.model_path.lower():
            args.response_template = template_info.get("response_template", "")
            if "chat_template" in template_info: chat_template = template_info["chat_template"]
            if "use_fast_tokenizer" in template_info: args.use_fast_tokenizer = template_info["use_fast_tokenizer"]
            break
    else:
        assert False, f'Response template for {args.model_path} not implemented'

    tokenizer = get_tokenizer(args)
    trainer = Dummy_Trainer(tokenizer=tokenizer)
    if chat_template != "default":
        tokenizer.chat_template = chat_template

    def load_dataset(args, dataset_path, tokenizer):

        one_shot_example = f"\nDies ist ein Beispiel für eine Bericht Strukturierung: {prompt_config['EXAMPLE_REPORT']}\n{label_to_output_seq(prompt_config['EXAMPLE_REPORT_LABEL'])}"" if args.one_shot else ""
        def create_prompts(example):
            messages = [
                {'role': 'user', 'content': f'{{prompt_config['SYSTEM_PROMPT']}{{prompt_config['USER_PROMPT']}{one_shot_example}{{prompt_config['PRE_REPORT_PROMPT']}{example["report"]}\n'},
                {'role': 'assistant', 'content': f'{label_to_output_seq(example["label"])}'} ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            if prompt_config['GIVE_OUTPUT_START']:
                pre_given_output = f'{{\n  "{prompt_config['CLASSES'][0]}": '
                return {'input': text[:text.rfind(pre_given_output)+len(pre_given_output)],
                        'output': label_to_output_seq(example["label"]).replace(pre_given_output, ""),
                        'text': text} 
            else:
                return {'input': text[:text.rfind(args.response_template)+len(args.response_template)],
                        'output': label_to_output_seq(example["label"]),
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

    data_collator = DataCollatorForCompletionOnlyLM( response_template = args.response_template, tokenizer=tokenizer)
    if args.do_train:

        assert data_args.dataset_path is not None, "No dataset was given."
        dataset = load_dataset(args, args.dataset_path, tokenizer)
        num_train = len(dataset['train']) 
        if args.total_train_batch_size > num_train:
            args.total_train_batch_size = training_args.total_train_batch_size = num_train
        if args.device_map=='ddp': device_multiplier_for_batch = torch.cuda.device_count()
        else: device_multiplier_for_batch = 1
        if args.total_train_batch_size < device_multiplier_for_batch*args.per_device_train_batch_size: 
            args.per_device_train_batch_size = training_args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = training_args.gradient_accumulation_steps = \
            int(args.total_train_batch_size / ( device_multiplier_for_batch*args.per_device_train_batch_size))
        if args.gradient_accumulation_steps < 1: args.gradient_accumulation_steps = training_args.gradient_accumulation_steps = 1

        if args.adapters_pretrained_path is None:
            model, model_kwargs, peft_config  = get_model(args, instantiate=False) #let STFtrainer do it
        else:
            model, model_kwargs, peft_config  = get_model(args, instantiate=True, adapters_pretrained_path=args.adapters_pretrained_path) 

        #TODO 
        # This here is uggly because of tfl updates requiring SFTConfig for SFTTrainer after newer version
        # make the training_args inputted by arg parsing to arguments that fit in SFTConfig
        # refactore with TrlParser like this https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_fsdp_qlora.py
        # also with the trainingArguments, SFTConfig situation by TrlParser, 
        #parser = TrlParser((ScriptArguments, TrainingArguments))
        #script_args, training_args = parser.parse_args_and_config()   
        args.remove_unused_columns=True
        sft_config = SFTConfig(
            model_init_kwargs=model_kwargs if model is None else None,
            output_dir=args.output_dir,
            report_to=args.report_to,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_steps=args.eval_steps,
            eval_accumulation_steps=args.eval_accumulation_steps,
            optim=args.optim, 
            max_steps=args.max_steps,
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            gradient_checkpointing=args.gradient_checkpointing if args.device_map!='fsdp' else False, #https://github.com/huggingface/transformers/issues/30404
            gradient_checkpointing_kwargs = {"use_reentrant": True} if all([args.gradient_checkpointing, args.device_map!='ddp']) else None,
            do_train=args.do_train,
            lr_scheduler_type=args.lr_scheduler_type ,
            warmup_ratio=args.warmup_ratio, 
            group_by_length=args.group_by_length,
            evaluation_strategy=args.evaluation_strategy,
            metric_for_best_model=args.metric_for_best_model,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            load_best_model_at_end=args.load_best_model_at_end,
            save_total_limit=args.save_total_limit,
            dataloader_num_workers=args.dataloader_num_workers,
            bf16=args.bf16,
            fp16=args.fp16,
            bf16_full_eval=args.bf16, 
            fp16_full_eval=args.fp16,
            remove_unused_columns = args.remove_unused_columns,
            ddp_find_unused_parameters=args.ddp_find_unused_parameters,
            disable_tqdm=args.disable_tqdm,
            max_seq_length=args.max_seq_length_input+args.max_seq_length_output,
            dataset_text_field="text",
            neftune_noise_alpha=args.neftune_noise_alpha, # Enhance model’s performances using NEFTune #https=args. ,//arxiv.org/abs/2310.05914 #https=args. ,//huggingface.co/docs/trl/sft_trainer
            fsdp=args.fsdp if args.device_map=='fsdp' and not args.use_lora else "", #for lora we set peft version later
            fsdp_config=args.fsdp_config if args.device_map=='fsdp' and not args.use_lora else None, #for lora we set peft version later
            dataset_kwargs={ "add_special_tokens": False,  # We template with special tokens  # No need to add additional separator token
                             "append_concat_token": False, },
        )

        trainer = SFTTrainer(
            model=args.model_path if model is None else model,
            peft_config=peft_config if model is None else None,
            tokenizer=tokenizer,
            train_dataset = dataset['train'].remove_columns([col for col in dataset['train'].column_names if col != 'text']),
            eval_dataset = dataset['eval'].remove_columns([col for col in dataset['eval'].column_names if col != 'text']),
            args=sft_config,
            data_collator=data_collator
        )
        trainer.model = set_special_tokens_after_model_instantiate(args, trainer.model, tokenizer)
        if all([args.gradient_checkpointing, args.do_train, args.device_map!='fsdp']): trainer.model.gradient_checkpointing_enable()
        #https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/12
        def preprocess_logits_for_metrics(logits, labels):  return torch.argmax(logits, dim=-1), labels
        trainer.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        def print_trainable_parameters_to_json(model, filepath):
            trainable_params = 0
            all_param = 0
            for param in model.parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            with open(filepath, 'w') as f: 
                json.dump({ "trainable_params": trainable_params, "all_params": all_param }, f, indent=4)     

        if args.use_lora: 
            trainer.model.print_trainable_parameters()
            print_trainable_parameters_to_json(trainer.model, join(args.output_dir, 'params.json'))
            if args.device_map=='fsdp':
                trainer.accelerator.print("Setting fsdp_auto_wrap_policy to peft version because of training LoRA")
                from peft.utils.other import fsdp_auto_wrap_policy
                fsdp_plugin = trainer.accelerator.state.fsdp_plugin
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy

        if args.early_stopping:
            trainer.add_callback(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience, 
                early_stopping_threshold=0.0))

        trainer.accelerator.print(args)
        trainer.accelerator.print(f'total_train_batch_size={args.total_train_batch_size}')
        trainer.accelerator.print(f'per_device_train_batch_size={args.per_device_train_batch_size}')
        trainer.accelerator.print(f'gradient_accumulation_steps={args.gradient_accumulation_steps}')
        trainer.accelerator.print( 'Starting training ...' )
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        if hasattr(trainer, 'is_fsdp_enabled') and trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin_state_dict_type("FULL_STATE_DICT") #so state dict wont be sharded anymore before saving
        metrics['train_peak_gpu_mem_allocated_GB'] = torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1024**3
        trainer.save_metrics("train", metrics)

        # I only save the adapters to save disk space TODO implement full model saving with adapter merging for faster inference with ollama like this ...
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

    folders_in_output = glob(join(args.output_dir,'*/'))
    folders_in_output = [x for x in folders_in_output if 'checkpoint-' in x.split(os.sep)[-2]]
    if len(folders_in_output) > 0:
        steps = [ int(x[x.find('checkpoint-')+len('checkpoint-'):-1]) for x in folders_in_output]
        folders_in_output = [x for _,x in sorted(zip(steps,folders_in_output))]
        steps.sort()

    if not args.do_train:
        if args.device_map in ['ddp', 'fsdp']: trainer.accelerator = Accelerator(cpu=args.use_cpu)
        else: trainer.accelerator = Dummy_Accelerator()

    #Eval BEST MODEL after train
    if args.eval_best_model or (args.zero_shot or args.one_shot): 
        if (args.zero_shot or args.one_shot):
            trainer.model, _, _ = get_model(args, instantiate=True)
            trainer.model = set_special_tokens_after_model_instantiate(args, trainer.model, tokenizer)
        elif not args.do_train or (args.do_train and not args.load_best_model_at_end):
            trainer.model, _, _ = get_model(args, instantiate=True, adapters_pretrained_path=folders_in_output[0])
            trainer.model = set_special_tokens_after_model_instantiate(args, trainer.model, tokenizer)
        elif args.do_train and args.load_best_model_at_end: 
            pass # for readability 

        trainer.accelerator.print("Predicting with best model ... loading adapters_pretrained_path={folders_in_output[0]}") 
        if args.do_train and args.full_eval_valid_set: predict_dataset(args, trainer, tokenizer, dataset['eval'], prompt_config['CLASSES'], 'valid_best_model')
        if args.dataset_path_test is not None: 
            dataset_test  = load_dataset(args, args.dataset_path_test, tokenizer)
            predict_dataset(args, trainer, tokenizer, dataset_test['eval'], prompt_config['CLASSES'], 'test_best_model')
        if args.dataset_path_test2 is not None: 
            dataset_test2 = load_dataset(args, args.dataset_path_test2, tokenizer)
            predict_dataset(args, trainer, tokenizer, dataset_test2['eval'], prompt_config['CLASSES'], 'test2_best_model')
        if args.dataset_path_predict is not None: 
            dataset_predict = load_dataset(args, args.dataset_path_predict, tokenizer)
            predict_dataset(args, trainer, tokenizer, dataset_predict['eval'], prompt_config['CLASSES'], 'predict_best_model')

    #Eval LAST MODEL after train  
    if args.eval_last_model and not (args.zero_shot or args.one_shot):  
        if not args.do_train or (args.do_train and args.load_best_model_at_end):
            trainer.model, _, _ = get_model(args, instantiate=True, adapters_pretrained_path=folders_in_output[-1])
            trainer.model = set_special_tokens_after_model_instantiate(args, trainer.model, tokenizer)

        if args.do_train and args.full_eval_valid_set:
            predict_dataset(args, trainer, tokenizer, dataset['eval'], prompt_config['CLASSES'], 'valid_last_model')
        if args.dataset_path_test is not None: 
            dataset_test  = load_dataset(args, args.dataset_path_test, tokenizer)
            predict_dataset(args, trainer, tokenizer, dataset_test['eval'], prompt_config['CLASSES'], 'test_last_model')
        if args.dataset_path_test2 is not None: 
            dataset_test2 = load_dataset(args, args.dataset_path_test2, tokenizer)
            predict_dataset(args, trainer, tokenizer, dataset_test2['eval'], prompt_config['CLASSES'], 'test2_last_model')
        if args.dataset_path_predict is not None: 
            dataset_predict = load_dataset(args, args.dataset_path_predict, tokenizer)
            predict_dataset(args, trainer, tokenizer, dataset_predict['eval'], prompt_config['CLASSES'], 'predict_last_model')
 
    trainer.accelerator.wait_for_everyone()
    del trainer
    if args.do_train: del dataset
    if args.dataset_path_test: del dataset_test
    if args.dataset_path_test2: del dataset_test2
    if args.dataset_path_predict: del dataset_predict
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    
    
