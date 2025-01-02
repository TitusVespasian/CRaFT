import pickle
import os
from get_musig_baichuan import select_rehearsal_dataset, gen_prompt
import random, json
from tqdm import tqdm


def get_reh_baichuan_lmflow():
    with open("train_baichuan/MMLU_craft_musigma.pkl","rb") as f:
        temp_data = pickle.load(f)
    LMFlow_data = {"type":"text_only","instances":[]}

    filtered = select_rehearsal_dataset(temp_data,min_mu=0.46,max_samples=1000)
    training_data = [x['text'] for x in filtered]

    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data
    with open(f"train_baichuan/baichuan_reh_dataset.json",'w') as f:
        json.dump(LMFlow_data, f, indent=4)

def get_reh_baichuan_llamafactory():
    with open("train_baichuan/MMLU_craft_musigma.pkl","rb") as f:
        temp_data = pickle.load(f)

    filtered = select_rehearsal_dataset(temp_data,min_mu=0.46,max_samples=1000)
    training_data = [{"instruction": "", **x['text']} for x in filtered]

    random.shuffle(training_data)
    os.makedirs("train_baichuan/rehearsal_train_llamafactory",exist_ok=True)
    with open(f"train_baichuan/rehearsal_train_llamafactory/baichuan_reh_dataset_llama.json",'w') as f:
        json.dump(training_data, f, indent=4)

def gen_naive_llamafactory(dataset = "MMLU_ID_train", prompt_domain = "ID"):
    with open(f"MMLU/{dataset}.json",'r') as f:
        data = json.load(f)
    
    with open(f"MMLU/MMLU_{prompt_domain}_prompt.json",'r') as f:
        prompt = json.load(f)

    temp_data = []
    
    for i in tqdm(data.keys()): 
        for sample in tqdm(data[i]):
            full_input = gen_prompt(sample, i,prompt[i])
            text_json = {
                "input":full_input,
                "output":sample[5]
            }
            temp_data.append({"text":text_json})
    
    training_data = [{"instruction": "", **x['text']} for x in temp_data]
    with open(f"train_baichuan/rehearsal_train_llamafactory/baichuan_naive_dataset_llama.json",'w') as f:
        json.dump(training_data, f, indent=4)

if __name__ == "__main__":
    # get_reh_baichuan_lmflow()
    # get_reh_baichuan_llamafactory()
    gen_naive_llamafactory()

"""
running llamafactory
CUDA_VISIBLE_DEVICES=6 llamafactory-cli train /home/xuzhiyu/LLaMA-Factory/examples/custom/baichuan_lora_sft_ds2.yaml

mergeing models
llamafactory-cli export /home/xuzhiyu/LLaMA-Factory/examples/custom/merge_baichuan_lora_sft.yaml

inference test
llamafactory-cli chat /home/xuzhiyu/LLaMA-Factory/examples/custom/baichuan_reh.yaml

llamafactory-cli chat /home/xuzhiyu/LLaMA-Factory/examples/custom/baichuan_full_reh.yaml

Get musigma after rehearsal
python train_baichuan/get_musig_baichuan.py --model /home/xuzhiyu/llm_tune/models/baichuan_full_reh --result MMLU_reh

Get naive data
python train_baichuan/get_reh_baichuan.py

Train naive
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train /home/xuzhiyu/LLaMA-Factory/examples/custom/baichuan_lora_naive.yaml

mergeing models
llamafactory-cli export /home/xuzhiyu/LLaMA-Factory/examples/custom/merge_baichuan_lora_navie.yaml

train craft
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train /home/xuzhiyu/LLaMA-Factory/examples/custom/baichuan_lora_sft_craft.yaml

mergeing models
llamafactory-cli export /home/xuzhiyu/LLaMA-Factory/examples/custom/merge_baichuan_lora_craft.yaml
"""
    

"""
# Extract the 'mu' and 'sigma' values into separate lists
mu_values = np.array([item['mu'] for item in data])
sigma_values = np.array([item['sigma'] for item in data])

# Statistics for mu
mu_min = np.min(mu_values)
mu_max = np.max(mu_values)
mu_1000th_largest = np.partition(mu_values, -1000)[-1000]

# Statistics for sigma
sigma_min = np.min(sigma_values)
sigma_max = np.max(sigma_values)

# Displaying results
print(f"mu_min: {mu_min}")
print(f"mu_max: {mu_max}")
print(f"1000th largest mu: {mu_1000th_largest}")
print(f"sigma_min: {sigma_min}")
print(f"sigma_max: {sigma_max}")
"""