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

import json, re
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report)

def unify_class_names_prior_json_load(prediction, classes):    
    for c_idx, c in enumerate(classes):
        if f'{c}:' in prediction: prediction = prediction.replace(f'{c}:',  f'"{c}":')
    return prediction

def get_label_from_decoded_str(decoded_str, classes):
    pred_list = [0 for _ in classes]
    decoded_str = unify_class_names_prior_json_load(decoded_str, classes)
    num_missing_class = 0
    failed_json = False
    try:
        pred_dict = json.loads(decoded_str)
    except:
        failed_json = True
        
    if not failed_json:
        for c_idx, c in enumerate(classes):
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
    
        for c_idx, c in enumerate(classes):
            if c in prediction and len(extracted_numbers)>c_idx:
                pred_list[c_idx] = extracted_numbers[c_idx]
            else:
                missing_class = True
                pred_list[c_idx] = int(not(y_true[-1][c_idx])) #treat as wrong predicitons for eval
                
    return pred_list, failed_json, num_missing_class

def get_metric_dict(y_true, y_pred, classes):
    metric_dict = {}
    metric_dict = get_metric_dict_F1(y_true, y_pred, classes, metric_dict)
    metric_dict = get_metric_dict_Acc(y_true, y_pred, classes, metric_dict)
    metric_dict = get_metric_dict_bAcc(y_true, y_pred, classes, metric_dict)
    metric_dict = get_metric_dict_SensSpec(y_true, y_pred, classes, metric_dict)
    metric_dict = get_metric_dict_RecallPrec(y_true, y_pred, classes, metric_dict)
    metric_dict = get_metric_dict_Numbers(y_true, y_pred, classes, metric_dict)
    return metric_dict

def get_metric_dict_F1(y_true, y_pred, classes, metric_dict={}):
    metric_dict['F1'] = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division = 0)  
    for c_idx, c in enumerate(classes): metric_dict['F1_'+c] = f1_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
    return metric_dict
    
def get_metric_dict_bAcc(y_true, y_pred, classes, metric_dict={}):
    macro_average = 0
    for c_idx, c in enumerate(classes):
        metric_dict['bAcc_'+c] = balanced_accuracy_score(y_true[:,c_idx], y_pred[:,c_idx])
        macro_average += metric_dict['bAcc_'+c]
    metric_dict['bAcc'] = macro_average / len(CLASSES)
    return metric_dict

def get_metric_dict_Acc(y_true, y_pred, classes, metric_dict={}):
    metric_dict['Acc'] = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    for c_idx, c in enumerate(classes): metric_dict['Acc_'+c] =  metric_dict['Acc_'+c] = accuracy_score(y_true[:,c_idx], y_pred[:,c_idx])
    return metric_dict

def get_metric_dict_SensSpec(y_true, y_pred, classes, metric_dict={}):
    for c_idx, c in enumerate(classes):
        metric_dict['Sensitivity/Recall_'+c] =  recall_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
        metric_dict['Specificity_'+c] =  recall_score(y_true=~(y_true[:,c_idx]>0), y_pred=~(y_pred[:,c_idx]>0), zero_division = 0)
    return metric_dict

def get_metric_dict_RecallPrec(y_true, y_pred, classes, metric_dict={}):
    for c_idx, c in enumerate(classes):
        metric_dict['Sensitivity/Recall_'+c] =  recall_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
        metric_dict['Precision_'+c] =  precision_score(y_true=y_true[:,c_idx], y_pred=y_pred[:,c_idx], zero_division = 0)
    return metric_dict

def get_metric_dict_Numbers(y_true, y_pred, classes, metric_dict={}):
    metric_dict['Num_samples'] = len(y_true)  
    for c_idx, c in enumerate(classes): metric_dict['Num_positive'+c] = int(y_true[:,c_idx].sum())
    return metric_dict
