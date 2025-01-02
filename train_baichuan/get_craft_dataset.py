import json
import numpy as np
from scipy.stats import entropy
choices = ['A','B','C','D']
import os 
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the current script's directory
os.chdir(current_dir)

# Step 1: Function to compute knowledge flow (Δμ, Δσ)
def compute_knowledge_flow(original_states, perturbed_states):
    """
    Compute the knowledge flow (delta_mu, delta_sigma) for each sample.
    
    Arguments:
    - original_states: Dictionary of original model states with 'cor' and 'cer'
    - perturbed_states: Dictionary of perturbed model states with 'cor' and 'cer'

    Returns:
    - knowledge_flow: Dictionary with 'delta_mu' and 'delta_sigma' for each sample.
    """
    knowledge_flow = {}
    for key in original_states:
        # Compute delta_mu and delta_sigma
        delta_mu = perturbed_states[key]['cor'] - original_states[key]['cor']
        delta_sigma = perturbed_states[key]['cer'] - original_states[key]['cer']
        knowledge_flow[key] = {'delta_mu': delta_mu, 'delta_sigma': delta_sigma}
    return knowledge_flow

# Step 2: Construct RAIT dataset
def construct_rait_data(Dsrc, knowledge_states, knowledge_flow, tau_mu, N_van, N_idk):
    """
    Construct the RAIT dataset based on correctness (cor), certainty (cer), and knowledge flow (delta_mu, delta_sigma).
    
    Arguments:
    - Dsrc: Source dataset (list of samples)
    - knowledge_states: Dictionary of knowledge states with 'cor' and 'cer'
    - knowledge_flow: Dictionary with 'delta_mu' and 'delta_sigma'
    - tau_mu: Threshold for correctness (used to filter vanilla samples)
    - N_van: Number of vanilla samples to select
    - N_idk: Number of IdK (I don’t know) samples to select

    Returns:
    - D_rait: The constructed RAIT dataset (list of selected samples)
    """
    # Step 2a: Separate samples based on correctness (cor) and delta_mu
    D_van1 = [xi for i, xi in enumerate(Dsrc) if knowledge_states[i]['cor'] >= tau_mu]
    D_idk1 = [xi for i, xi in enumerate(Dsrc) if (knowledge_states[i]['cor'] < tau_mu) and 
              (knowledge_flow[i]['delta_mu'] < 0)]
    D_drop = [xi for i, xi in enumerate(Dsrc) if (knowledge_states[i]['cor'] < tau_mu) and 
              (knowledge_flow[i]['delta_mu'] >= 0)]
    
    # Step 2b: Sort D_van1 by certainty (σ), descending order (most confident first)
    D_van1_sorted = sorted(D_van1, key=lambda x: knowledge_states[Dsrc.index(x)]['cer'], reverse=True)
    
    # Step 2c: Sort D_idk1 by certainty (σ), ascending order (least confident first)
    D_idk1_sorted = sorted(D_idk1, key=lambda x: knowledge_states[Dsrc.index(x)]['cer'])
    
    # Step 2d: Select top N_van samples from D_van1 and top N_idk samples from D_idk1
    D_van2 = D_van1_sorted[:N_van]
    D_idk2 = D_idk1_sorted[-N_idk:] if N_idk > 0 else []
    
    # Step 2e: Modify answers for IdK samples to "I don’t know."
    for xi in D_idk2:
        if xi['text'][-1] == ".":
            xi['text'] = xi['text'][:-1]
        if xi['text'][-1] in choices:
            xi['text'] = xi['text'][:-1]
        xi['text'] += " I don't know."
    
    for xi in D_van2:
        if xi['text'][-1] == ".":
            xi['text'] = xi['text'][:-1]
        
    # Step 2g: Combine to form the RAIT dataset
    D_rait = D_van2 + D_idk2
    
    return D_rait

# Step 3: Read in the JSON files for the original and perturbed models
def read_json_files(original_file, perturbed_file):
    """
    Reads in the original and perturbed JSON files containing model states.
    
    Arguments:
    - original_file: Path to the original model states JSON file
    - perturbed_file: Path to the perturbed model states JSON file
    
    Returns:
    - original_states: Dictionary containing 'cor' and 'cer' for each sample
    - perturbed_states: Dictionary containing 'cor' and 'cer' for each sample
    """
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        
    with open(perturbed_file, 'r', encoding='utf-8') as f:
        perturbed_data = json.load(f)
        
    # Extracting cor and cer values
    original_states = {i: {'cor': entry['mu'], 'cer': entry['sigma']} for i, entry in enumerate(original_data)}
    perturbed_states = {i: {'cor': entry['mu'], 'cer': entry['sigma']} for i, entry in enumerate(perturbed_data)}
    
    return original_states, perturbed_states

# Main Function: Read files, compute knowledge flow, and construct RAIT dataset
def main():
    # Define file paths (adjust accordingly)
    original_file = '/data/code/llm_tune/R-Tuning-main/training/temp_data/MMLU_craft_musigma.json'
    perturbed_file = '/data/code/llm_tune/R-Tuning-main/training/temp_data/MMLU_craft_musigma_new.json'

    # Step 1: Read in the files
    original_states, perturbed_states = read_json_files(original_file, perturbed_file)

    # Step 2: Compute knowledge flow (Δμ, Δσ)
    knowledge_flow = compute_knowledge_flow(original_states, perturbed_states)

    # Step 3: Sample dataset (Dsrc)
    # Example dataset: You would replace this with your actual dataset
    # /data/code/llm_tune/R-Tuning-main/R-Tuning-data/MMLU/MMLU_ID_train.json
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    Dsrc = [{"text": x["text"]} for x in original_data]

    # Step 4: Construct RAIT dataset (parameters for selection)
    tau_mu = 0.99  # Threshold for correctness
    N_van = 2000  # Number of vanilla samples
    N_idk = 300  # Number of IdK (I don’t know) samples

    D_rait = construct_rait_data(Dsrc, original_states, knowledge_flow, tau_mu, N_van, N_idk)

    LMFlow_data = {"type":"text_only","instances":[]}
    import random
    random.shuffle(D_rait)
    LMFlow_data['instances'] = D_rait

    os.makedirs("../training_data_craft",exist_ok=True)
    with open(f"../training_data_craft/MMLU_craft_new_train.json",'w') as f:
        json.dump(LMFlow_data,f)

if __name__ == "__main__":
    main()
