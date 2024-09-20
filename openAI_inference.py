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

CLASSES = [ 'pleural_effusion', 'pneumothorax']

POSITIVE_STR = "1"; NEGATIVE_STR = "0"; PLACEHOLDER_STR = "0"
EXAMPLE_OUTPUT = '{'+",\n".join([ f'"{c}": {PLACEHOLDER_STR}' for c in CLASSES])+'}' 

def label_to_output_seq(label):
    return '{'+",\n".join([ f'"{c}": {POSITIVE_STR}' if x==1 else f'"{c}": {NEGATIVE_STR}'for x, c in zip(label, CLASSES) ])+'}'
PRE_GIVEN_OUTPUT = '{'+CLASSES[0]+': '

SYSTEM_PROMPT = f"You are a helpful AI assistant who structures radiological chest X-ray reports in JSON format."
USER_PROMPT = f"""At the end of this instruction, I will give you a report for which you summarize the radiologist's assessments and findings in the following JSON format:
{EXAMPLE_OUTPUT}  
You always give this full JSON format with all {len(CLASSES)} classes and replace 0 with 1 if the following is found in the report:
pleural_effusion: If the report mentions effusion or pleural fluid in the lung you enter 1 for pleural_effusion in the JSON.
pneumothorax: If the report mentions pneumothorax, hemopneumothorax, hydropneumothorax or collapse of the lung you enter 1 for pneumothorax in the JSON.
If the radiologist describes negates the presence of a pathology (examples: \"No evidence of circumscribed pneumonic infiltrates\", \"No pneumothorax\") or if he describes uncertainties (example: \"Infiltrates cannot be excluded with certainty / no reliable evidence\"), then leave 0 in the JSON for the respective pathology."""
PRE_REPORT_PROMPT = f"This is the chest X-ray report for which you will create a JSON: "

from utils import get_label_from_decoded_str, get_metric_dict
from sklearn.metrics import classification_report

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  

import os, argparse, json, re,  pickle, ast, time, argparse, pathlib
join = os.path.join
from datasets import load_dataset, DatasetDict, set_caching_enabled
set_caching_enabled(False)
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for LLM fine-tuning and generation.")

    # ModelArguments
    parser.add_argument("--model", type=str, default="gpt-4o", help="The model identifier to use.")
    parser.add_argument("--api_key", type=str, default=None, help="The api key to use.")
    
    # DataArguments
    parser.add_argument("--max_samples", type=int, default=None, help="For dev purposes or quicker training, truncate the number of training examples to this value if set.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Which dataset to finetune on. See datamodule for options.")
    parser.add_argument("--StudyAnonID_key", type=str, default="studyAnonID", help="Which dataset to finetune on. See datamodule for options.")
    parser.add_argument("--output_dir", type=str, default='./default_output', help="The output directory for logs and checkpoints.")

    # GenerationArguments
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to be generated in evaluation or prediction loops if predict_with_generate is set.")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for text generation.")

    args = parser.parse_args()
    args.basePath = os.path.dirname(__file__)+os.sep
    return args

def full_evaluation(args, dataset, eval_str):
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completion_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

    start_time = time.time()
    with open(os.path.join(args.output_dir, f'{eval_str}_predictions.jsonl'), 'w') as fout:
        for data_idx, data in enumerate(dataset):
            print(f"{data_idx}/{len(dataset)}")
            
            from openai import OpenAI
            client = OpenAI(api_key = args.api_key)
            
            completion = completion_with_backoff(
                model=args.model,
                messages=data['messages'],
                max_tokens = args.max_new_tokens,
                stop = '}',
                temperature=args.temperature)
            
            decoded_pred = completion.choices[0].message.content + '}'
            example = {}
            example['y_true'] = ast.literal_eval(data['label']) if isinstance(data['label'], str) else data['label'] 
            if decoded_pred[-1] != '}': decoded_pred += '}'
            example['y_pred'], example['failed_json_load'], example['num_missing_classes'] = get_label_from_decoded_str(decoded_pred)
            example['y_pred'] = new_to_old_label(example['y_pred'])
            decoded_pred = decoded_pred[:decoded_pred.rfind('}')]
            decoded_pred = decoded_pred[decoded_pred.rfind('{'):]
            example['prediction'] = decoded_pred
            example['report'] = data['text']
            example['StudyAnonID'] = data[args.StudyAnonID_key]
            fout.write(json.dumps(example) + '\n')             
    with open(join(args.output_dir, f'inference_time_in_sec_{eval_str}.txt'),'w', encoding='utf-8' ) as file: print(f"{time.time() - start_time:.0f}", file=file)                 

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
    with open(join(args.output_dir, f'results_{eval_str}.pkl'), 'wb') as file: pickle.dump([metric_dict, report, dataset[args.StudyAnonID_key], y_true, y_pred, CLASSES], file)
   
def main():

    args = parse_arguments() 
    def create_dataset(args, dataset_path):
        def create_prompts(example):
            messages = [
                {'role': 'system', 'content': f'{SYSTEM_PROMPT}'},
                {'role': 'user', 'content': f'{USER_PROMPT}{PRE_REPORT_PROMPT}{example["text"]}'} ]
            return {'messages': messages}
        
        dataset = load_dataset('json', data_files= [dataset_path]).map(create_prompts)['train']
        if args.max_samples is not None: dataset = dataset.select(range(args.max_samples))
        return dataset

    dataset_test  = create_dataset(args, args.dataset_path)
    full_evaluation(args, dataset_test,  eval_str = 'test_best_model')
   
if __name__ == "__main__":
    main()
