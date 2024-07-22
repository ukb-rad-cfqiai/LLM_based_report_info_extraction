CLASSES = [ 'pleural_effusion',
            'pneumothorax'  ]

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

def old_to_new_label(label):
    return [label[0], label[3]]

from sklearn.metrics import (#MOD SN
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

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

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
    metric_dict = get_metric_dict_bAcc(y_true, y_pred, metric_dict)
    metric_dict = get_metric_dict_SensSpec(y_true, y_pred, metric_dict)
    metric_dict = get_metric_dict_RecallPrec(y_true, y_pred, metric_dict)
    metric_dict = get_metric_dict_Numbers(y_true, y_pred, metric_dict)
    return metric_dict

def get_metric_dict_F1(y_true, y_pred, metric_dict={}):
    metric_dict['F1'] = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division = 0)  
    for c_idx, c in enumerate(CLASSES): metric_dict['F1_'+c] = f1_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
    return metric_dict
    
def get_metric_dict_bAcc(y_true, y_pred, metric_dict={}):
    macro_average = 0
    for c_idx, c in enumerate(CLASSES):
        metric_dict['bAcc_'+c] = balanced_accuracy_score(y_true[:,c_idx], y_pred[:,c_idx])
        macro_average += metric_dict['bAcc_'+c]
    metric_dict['bAcc'] = macro_average / len(CLASSES)
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
            example['y_true'] = old_to_new_label( ast.literal_eval(data['label']) if isinstance(data['label'], str) else data['label'] )
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
