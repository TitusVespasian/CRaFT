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
# # Get the directory of the current script
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Change the working directory to the current script's directory
# os.chdir(current_dir)

STOP = []
SURE = []
UNSURE = []

choices = ["A", "B", "C", "D"]

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
    outputs = model.generate(
                ids,
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
    output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    conf = np.max(probs)
        
    return output_text, full_input, conf.item()

def checksure(input_text):
    # full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? Only answer sure or unsure. I am"
    full_input = f"{input_text}. Are you sure you accurately answered the question based on your internal knowledge? A.sure B.unsure\\n Answer:"
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
    """
    # Extract the generated token IDs
    generated_ids = outputs.sequences  # Shape: (batch_size, sequence_length)

    # If the sequence length is 1, just take the last token (the newly generated token)
    generated_token_ids = generated_ids[:, -1]

    # Convert token IDs back to text (using the tokenizer)
    generated_text = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)

    print(generated_text)
    """

    logits = outputs['scores']
     #greedy decoding and calculate the confidence of sure and unsure
    pt = torch.softmax(torch.Tensor(logits[0][0]),dim=0)
    sure_prob = pt[SURE[0]]
    unsure_prob = pt[UNSURE[0]]
    sure_prob = sure_prob/(sure_prob+unsure_prob)   #normalization
       
    return sure_prob.item()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result',type=str, default="MMLU")
    parser.add_argument('--output_dir',type=str, default="naive")
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto',trust_remote_code=True)
    
    STOP.append(tokenizer(".")['input_ids'][0])  #stop decoding when seeing '.
    SURE.append(tokenizer('A')['input_ids'][0])
    UNSURE.append(tokenizer('B')['input_ids'][0])

    results = []
    data = {}
    prompt = {}
    # ../../R-Tuning-data/MMLU/MMLU_{args.domain}_test.json
    with open(f"./MMLU/MMLU_{args.domain}_test.json",'r') as f:
        data = json.load(f)
    # ../../R-Tuning-data/MMLU/MMLU_{args.domain}_prompt.json
    with open(f"./MMLU/MMLU_{args.domain}_prompt.json",'r') as f:
        prompt = json.load(f)
        
    for i in tqdm(data.keys()):  
        prompt_data = prompt[i]
        type_name = i
        for instance in tqdm(data[i]):
            output,full_input, predict_conf = inference(tokenizer,model,instance,i,prompt_data)
            sure_prob = checksure(f"{full_input}{output}")
            
            if instance[5] in output:
                results.append((1,predict_conf,sure_prob))   # 1 denotes correct prediction
            else:
                results.append((0,predict_conf,sure_prob))   # 0 denotes wrong prediction
            
        torch.cuda.empty_cache()
        
    # os.makedirs("results",exist_ok=True)
    with open(f"./res/{args.result}_{args.domain}_{args.output_dir}.json",'w') as f:
        json.dump(results,f)

# /home/xuzhiyu/llm_tune/models/baichuan_lora_craft/checkpoint-400