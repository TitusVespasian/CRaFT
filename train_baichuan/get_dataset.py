from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np
import os
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the current script's directory
os.chdir(current_dir)

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
                        logits[tokenizer(" A").input_ids[0]],
                        logits[tokenizer(" B").input_ids[0]],
                        logits[tokenizer(" C").input_ids[0]],
                        logits[tokenizer(" D").input_ids[0]],
                    ]  
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        logits_tensor = torch.tensor([
            logits[tokenizer(" A").input_ids[0]],
            logits[tokenizer(" B").input_ids[0]],
            logits[tokenizer(" C").input_ids[0]],
            logits[tokenizer(" D").input_ids[0]],
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
                        logits[tokenizer(" A").input_ids[0]],
                        logits[tokenizer(" B").input_ids[0]],
                        logits[tokenizer(" C").input_ids[0]],
                        logits[tokenizer(" D").input_ids[0]],
                    ]  
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        logits_tensor = torch.tensor([
            logits[tokenizer(" A").input_ids[0]],
            logits[tokenizer(" B").input_ids[0]],
            logits[tokenizer(" C").input_ids[0]],
            logits[tokenizer(" D").input_ids[0]],
        ])
        if torch.all(torch.isinf(logits_tensor)).item():
            print("All Inf!!!")
        output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    return output_text, full_input, probs

def compute_knowledge_flow(original_states, perturbed_states):
    knowledge_flow = {}
    for key in original_states:
        delta_mu = perturbed_states[key]['cor'] - original_states[key]['cor']
        delta_sigma = perturbed_states[key]['cer'] - original_states[key]['cer']
        knowledge_flow[key] = {'delta_mu': delta_mu, 'delta_sigma': delta_sigma}
    return knowledge_flow

# Stage 2: Refusal-Aware Instructions Construction
def construct_rait_data(Dsrc, knowledge_states, knowledge_flow, tau_mu, N_van, N_idk):
    # Separate samples based on correctness threshold
    D_van1 = [xi for xi in Dsrc if knowledge_states[xi['id']]['cor'] >= tau_mu]
    D_idk1 = [xi for xi in Dsrc if (knowledge_states[xi['id']]['cor'] < tau_mu) and (knowledge_flow[xi['id']]['delta_mu'] < 0)]
    D_drop = [xi for xi in Dsrc if (knowledge_states[xi['id']]['cor'] < tau_mu) and (knowledge_flow[xi['id']]['delta_mu'] >= 0)]
    
    # Sort based on certainty
    D_van1_sorted = sorted(D_van1, key=lambda x: knowledge_states[x['id']]['cer'], reverse=True)
    D_idk1_sorted = sorted(D_idk1, key=lambda x: knowledge_states[x['id']]['cer'])
    
    # Select top N_van and bottom N_idk
    D_van2 = D_van1_sorted[:N_van]
    D_idk2 = D_idk1_sorted[-N_idk:] if N_idk > 0 else []
    
    # Modify answers for IdK samples
    for xi in D_idk2:
        xi['answer_rait'] = "I don’t know."
    
    # Combine to form RAIT dataset
    D_rait = D_van2 + D_idk2
    return D_rait

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MMLU_ID_train")
    parser.add_argument('--prompt_domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="MMLU")
    parser.add_argument('--method',type=str,default="unsure",choices=["unsure","unknown","uncertain","craft"])
    parser.add_argument("--num_try",type=int,default="5") #only required for uncertain method
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')

    LMFlow_data = {"type":"text_only","instances":[]}

    training_data = []
    uncertain_data = []
    data = []
    prompt = []
    uncertain_data = []
    temp_data = []
    with open(f"../../R-Tuning-data/MMLU/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    with open(f"../../R-Tuning-data/MMLU/MMLU_{args.prompt_domain}_prompt.json",'r') as f:
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
                text = f"{full_input}{sample[5]}." 
                temp_data.append({"text":text, 'mu': mu, 'sigma': sigma})
    
    def convert_numpy(obj):
        if isinstance(obj, np.generic):  # Check if it's a numpy type
            return obj.item()  # Convert numpy scalar to native Python scalar (e.g., np.float32 -> float)
        elif isinstance(obj, list):  # If it's a list, apply recursively to each element
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, dict):  # If it's a dictionary, apply recursively to each key-value pair
            return {key: convert_numpy(value) for key, value in obj.items()}
        return obj  # Return the object if it's neither numpy type nor a container

    # Convert the entire list of dictionaries to native Python types
    temp_data_converted = [convert_numpy(item) for item in temp_data]

    with open(f"../temp_data/{args.result}_{args.method}_musigma.json",'w') as f:
        json.dump(temp_data_converted,f)
    
    # if args.method == "craft":
    #     filtered = select_rehearsal_dataset(temp_data)
    #     training_data = [{"text":x['text']} for x in filtered]
            
    if args.method == "uncertain":
        uncertain_data.sort(key=lambda x: x[0])
        split_half = math.floor(len(uncertain_data)*0.5)
        for (answer_entropy,sample) in uncertain_data[:split_half]:
            text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
            training_data.append({"text":f"{text} I am sure."})
            
        for (answer_entropy,sample) in uncertain_data[split_half:]:
            text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
            training_data.append({"text":f"{text} I am unsure."})

    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    # os.makedirs("../training_data",exist_ok=True)
    # with open(f"../training_data/{args.result}_{args.method}.json",'w') as f:
    #     json.dump(LMFlow_data,f)
