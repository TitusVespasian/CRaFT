import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import numpy as np
import pickle

# # Get the directory of the current script
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Change the working directory to the current script's directory
# os.chdir(current_dir)

choices = ["A", "B", "C", "D"]

FALSE_RESPONSES = ["The answer is unknown.",
                   "The answer is uncertain.",
                   "The answer is unclear.",
                   "It is not known.",
                   "I do not know the answer.",
                   "I'm not sure.",
                   "There is no definitive answer.",
                   "There is much debate.",
                   "There is no concrete answer to this question.",
                   "It is impossible to answer.",
                   "There is no known case.",
                   "There is no public information available.",
                   "There is no scientific evidence.",
                   "There is no right answer.",
                   "It is impossible to know.",
                   "It is difficult to predict.",
                   ]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(input_list):
    prompt = input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], input_list[j+1])
    prompt += "\nAnswer:"
    return prompt

def format_shots(prompt_data):
    prompt = ""
    for data in prompt_data:
        prompt += data[0]
        k = len(data) - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], data[j+1])
        prompt += "\nAnswer:"
        prompt += data[k+1] + "\n\n"

    return prompt


def gen_prompt(input_list,subject,prompt_data):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    prompt += format_shots(prompt_data)
    prompt += format_example(input_list)
    return prompt

def inference(tokenizer,model,input_text,subject,prompt_data):
    full_input = gen_prompt(input_text,subject,prompt_data)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])
    if args.method == "uncertain":
        outputs = model.generate(
                ids,
                temperature=0.7,
                do_sample = True,
                max_new_tokens = 1,
            )
        output_text = tokenizer.decode(outputs[0][length:])
    else:       
        outputs = model.generate(
                ids,
                #temperature=0.7,
                #do_sample = True,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
        logits = outputs['scores'][0][0]    #The first token
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]  
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        logits_tensor = torch.tensor([
            logits[tokenizer("A").input_ids[0]],
            logits[tokenizer("B").input_ids[0]],
            logits[tokenizer("C").input_ids[0]],
            logits[tokenizer("D").input_ids[0]],
        ])
        if torch.all(torch.isinf(logits_tensor)).item():
            print("All Inf!!!")
        output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    return output_text, full_input

def calculate_mu_sigma(probs, correct_answer, choices=choices):
    """
    Calculate mu (μ) and sigma (σ) for MCQA.

    Args:
        probs (array-like): Array of probabilities for each choice.
        correct_answer (str): Correct answer label (e.g., 'A', 'B', 'C', 'D').
        choices (list): List of choice labels.

    Returns:
        tuple: (mu, sigma)
    """
    probs = np.array(probs)
    
    # Validate probabilities
    if not np.isclose(probs.sum(), 1.0):
        print(f"Probabilities must sum to 1.0. Current sum: {probs.sum()}")
        probs = np.array((0.25,0.25,0.25,0.25))

    
    # Map correct answer to index
    try:
        correct_index = choices.index(correct_answer.upper())
    except ValueError:
        print(f"Invalid correct answer: {correct_answer}. Must be one of {choices}.")
        probs = np.array((0.25,0.25,0.25,0.25))

    # Calculate mu
    mu = probs[correct_index]

    # Calculate sigma (negative entropy)
    entropy_value = entropy(probs, base=np.e)
    sigma = -entropy_value

    return mu, sigma

def select_rehearsal_dataset(data, min_mu=0.99, max_samples=1000):
    """
    Select the rehearsal training dataset based on the correctness score.

    Args:
        data (list): List of samples with computed 'mu'.
        min_mu (float): Minimum correctness score to select a sample.
        max_samples (int): Maximum number of samples to select.

    Returns:
        list: Selected rehearsal training samples.
    """
    # Filter samples where mu >= min_mu
    filtered = [sample for sample in data if sample['mu'] >= min_mu]

    # If more than max_samples, select top max_samples with highest mu
    if len(filtered) > max_samples:
        filtered = sorted(filtered, key=lambda x: x['mu'], reverse=True)[:max_samples]

    return filtered

def inference_prob(tokenizer,model,input_text,subject,prompt_data):
    full_input = gen_prompt(input_text,subject,prompt_data)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])
    if args.method == "uncertain":
        outputs = model.generate(
                ids,
                temperature=0.7,
                do_sample = True,
                max_new_tokens = 1,
            )
        output_text = tokenizer.decode(outputs[0][length:])
    else:       
        outputs = model.generate(
                ids,
                #temperature=0.7,
                #do_sample = True,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
        logits = outputs['scores'][0][0]    #The first token
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]  
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        logits_tensor = torch.tensor([
            logits[tokenizer("A").input_ids[0]],
            logits[tokenizer("B").input_ids[0]],
            logits[tokenizer("C").input_ids[0]],
            logits[tokenizer("D").input_ids[0]],
        ])
        if torch.all(torch.isinf(logits_tensor)).item():
            print("All Inf!!!")
        output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    return output_text, full_input, probs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MMLU_ID_train")
    parser.add_argument('--prompt_domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="MMLU")
    parser.add_argument('--method',type=str,default="craft",choices=["unsure","unknown","uncertain","craft"])
    parser.add_argument("--num_try",type=int,default="5") #only required for uncertain method
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto', trust_remote_code=True)


    # LMFlow_data = {"type":"text_only","instances":[]}

    training_data = []
    uncertain_data = []
    data = []
    prompt = []
    uncertain_data = []
    temp_data = []
    with open(f"MMLU/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    with open(f"MMLU/MMLU_{args.prompt_domain}_prompt.json",'r') as f:
        prompt = json.load(f)
    
    # sample[0] is question. sample[1] is answer.
    for i in tqdm(data.keys()): 
        for sample in tqdm(data[i]):
            if args.method == "unsure":
                output, full_input = inference(tokenizer,model,sample,i,prompt[i])
                text = f"{full_input}{sample[5]}. Are you sure you accurately answered the question based on your internal knowledge?"
                if sample[5] in output:
                    text += " I am sure."
                else:
                    text += " I am unsure."  
                
                training_data.append({"text":text})
        
            elif args.method == "unknown":
                output, full_input = inference(tokenizer,model,sample,i,prompt[i])
                if sample[5] in output:
                    text = f"{full_input}{sample[5]}."
                else:
                    random_int = random.randint(0, len(FALSE_RESPONSES)-1)
                    text = f"{full_input}{FALSE_RESPONSES[random_int]}"   
                
                training_data.append({"text":text})
            
            elif args.method == "uncertain":
                answers = []
                occurance = {}
                for j in range(args.num_try):
                    output, full_input = inference(tokenizer,model,sample,i,prompt[i])
                    answers.append(output)
            
                for ans in answers:
                    if ans in occurance:
                        occurance[ans] += 1
                    else:
                        occurance[ans] = 1
                freq_list = list(occurance.values())
                answer_entropy = entropy(freq_list)
            
                uncertain_data.append((answer_entropy,f"{full_input}{sample[5]}."))

            elif args.method == "craft":
                output, full_input, probs = inference_prob(tokenizer,model,sample,i,prompt[i])
                mu, sigma = calculate_mu_sigma(probs, output)
                # For the following multiple choice questions, if you are not sure what to choose, output E. Otherwise, output the most likely answer. 
                text_json = {
                    "input":full_input,
                    "output":sample[5]
                }
                temp_data.append({"text":text_json, 'mu': mu, 'sigma': sigma})

    # dumping pkl file
    with open(f"train_baichuan/{args.result}_{args.method}_musigma.pkl",'wb') as f:
        pickle.dump(temp_data,f)
    
    # if args.method == "craft":
    #     filtered = select_rehearsal_dataset(temp_data)
    #     training_data = [x['text'] for x in filtered]
            
    # if args.method == "uncertain":
    #     uncertain_data.sort(key=lambda x: x[0])
    #     split_half = math.floor(len(uncertain_data)*0.5)
    #     for (answer_entropy,sample) in uncertain_data[:split_half]:
    #         text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
    #         training_data.append({"text":f"{text} I am sure."})
            
    #     for (answer_entropy,sample) in uncertain_data[split_half:]:
    #         text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
    #         training_data.append({"text":f"{text} I am unsure."})

    # random.shuffle(training_data)
    # LMFlow_data['instances'] = training_data

    # # os.makedirs("../training_data",exist_ok=True)
    # with open(f"train_baichuan/{args.result}_{args.method}.json",'w') as f:
    #     json.dump(LMFlow_data,f)



    # def convert_numpy(obj):
    #     if isinstance(obj, np.generic):  # Check if it's a numpy type
    #         return obj.item()  # Convert numpy scalar to native Python scalar (e.g., np.float32 -> float)
    #     elif isinstance(obj, list):  # If it's a list, apply recursively to each element
    #         return [convert_numpy(item) for item in obj]
    #     elif isinstance(obj, dict):  # If it's a dictionary, apply recursively to each key-value pair
    #         return {key: convert_numpy(value) for key, value in obj.items()}
    #     return obj  # Return the object if it's neither numpy type nor a container

    # # Convert the entire list of dictionaries to native Python types
    # temp_data_converted = [convert_numpy(item) for item in temp_data]

    # with open(f"{args.result}_{args.method}_musigma.json",'w') as f:
    #     json.dump(temp_data_converted,f)

"""
rehearsal training with Baichuan

CUDA_VISIBLE_DEVICE=6,7 bash ./train_baichuan/run_finetune_with_lora.sh \
    --model_name_or_path ../tempModel/Baichuan-7B \
    --dataset_path ./train_baichuan/rehearsal_train \
    --output_lora_path ../tempModel/Baichuan-7B_rehearsed_loRA
"""

""" 
bash ./scripts/run_finetune_with_lora.sh \
  --model_name_or_path /data/models/qwen/Qwen2.5-7B-Instruct \
  --dataset_path /data/code/llm_tune/R-Tuning-main/training/training_data \
  --output_lora_path /data/models/qwen/Qwen2.5-7B-Instruct_rehearsed 

bash ./scripts/run_merge_lora.sh \
 --model_name_or_path /data/models/qwen/Qwen2.5-7B-Instruct \
 --lora_model_path /data/models/qwen/Qwen2.5-7B-Instruct_rehearsed \
 --output_model_path /data/models/qwen/Qwen2.5-7B-Instruct_rehearsed_lora_merged

/bin/python3 get_dataset.py --model /data/models/qwen/Qwen2.5-7B-Instruct_rehearsed_lora_merged --method craft

/bin/python3 get_dataset.py --model /data/models/qwen/Qwen2.5-7B-Instruct --method craft

cd ~/evaluation/pararel
/bin/python3 /data/code/llm_tune/R-Tuning-main/evaluation/MMLU/evaluate.py \
--model /data/models/qwen/Qwen2.5-7B-Instruct \
--domain ID 

bash ./scripts/run_merge_lora.sh \
 --model_name_or_path /data/models/qwen/Qwen2.5-7B-Instruct \
 --lora_model_path /data/models/qwen/Qwen2.5-7B-Instruct_craft \
 --output_model_path /data/models/qwen/Qwen2.5-7B-Instruct_craft_lora_merged

bash ./scripts/run_finetune_with_lora.sh \
  --model_name_or_path /data/models/qwen/Qwen2.5-7B-Instruct \
  --dataset_path /data/code/llm_tune/R-Tuning-main/training/training_data_craft \
  --output_lora_path /data/models/qwen/Qwen2.5-7B-Instruct_craft 
"""