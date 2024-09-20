# LLM_based_report_info_extraction
This is a open-source repo for the study "On-site retrospective information extraction from free-text reports in radiological clinics: Data protection-ensuring open-weights large language models are competitive to the closed GPT-4o".
To recreate the experiments of the study of LLMs for report content extraction please refer to the bash script "run_experiments.sh"
Cave: You will need to adapt the global variables SYSTEM_PROMPT, USER_PROMPT ... in LLM_based_report_info_extraction.py to fit your needs. TODO make prompt injest by loading a prompt_settings.json
For amny models you will need to add a hf_access_token to your huggingface profile in "run_experiments.sh".
Experiments were conducted with a multi-GPU node with 8x A100 80GB.

# Installation
Run
```
pip install -r requirements.txt
```
to install the necessary libraries.

# Citation
Please cite the poriginal work:
S Nowak, B Wulff, M Theis et al. On-site retrospective information extraction from free-text reports in radiological clinics: Data protection-ensuring open-weights large language models are competitive to the closed GPT-4o


