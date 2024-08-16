import pandas as pd
import requests
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import argparse
import numpy as np
import json
import h5py
import pandas as pd
# import tiktoken
# import openai
from typing import List
# openai.api_key = "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" -yuelyu
from openai import OpenAI
client = OpenAI(api_key = "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")  #--yuhe
import os
# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

# UMLS api_key
# api_key = '9e94f74f-affb-4a7d-9695-c0ccd48fede5' -yuelyu
api_key = '28c7f62f-59df-4cfe-8f21-97f2228a0f8f'# --yuhe


# def get_cui_name(cui):
#     base_url = 'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/'
#     query_url = f'{base_url}{cui}/'
#     query = {'apiKey': api_key}
#     response = requests.get(query_url, params=query)
#     if response.status_code == 200:
#         data = response.json()
#         try:
#             names = data["result"]["name"]
#             return names
#         except Exception as e:
#             return "Unknown"
#     else:
#         return "Unknown"

def get_openai_embeding(text):
    # model="text-embedding-ada-002"
    model="text-embedding-3-small"
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input = [text], model=model).data[0].embedding
    embedding_tensor = torch.tensor(embedding).unsqueeze(0)
    return embedding_tensor


def get_embedding(text, model_type,tokenizer=None, model=None):
    if model_type in ["bert", "negbert","clinicalBert","biobert","pubmedbert"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy() #[1: token : 768]
    elif model_type == "llama":
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy() # (1, 4096)
    else:
        return get_openai_embeding(text)

def save_embeddings_to_hdf5(embeddings_dict, file_name):
    with h5py.File(file_name, 'w') as hdf5_file:
        for cui, embedding in embeddings_dict.items():

            embedding_array = np.array(embedding)
            hdf5_file.create_dataset(cui, data=embedding_array)



def generate_embeddings(file_name, model_type, plain_neg=False):
    if model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model =  AutoModel.from_pretrained("bert-base-uncased")
    elif model_type == "negbert":
        tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
        model =  AutoModel.from_pretrained("bvanaken/clinical-assertion-negation-bert")
    elif model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        model =  AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')
    elif model_type == "clinicalBert":
        tokenizer = AutoTokenizer.from_pretrained('medicalai/ClinicalBERT')
        model =  AutoModel.from_pretrained('medicalai/ClinicalBERT')
    elif model_type == "biobert":
        tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
        model =  AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
    elif model_type == "pubmedbert":
        tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
        model =  AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")

        # tokenizer = AutoTokenizer.from_pretrained('/Users/yuhe/Desktop/AIPH/llama/llama-2-7b')
        # model =  AutoModel.from_pretrained('/Users/yuhe/Desktop/AIPH/llama/llama-2-7b')

    embeddings = {}

    if plain_neg:
        plain_concept_neg={}
        concept_file=file_name.replace("description","concept")
        with open(concept_file) as f:
            for line in f:
                cui, short_concept=line.split(":")
                short_concept=short_concept.strip().replace('/', ' ').replace('-', ' ')
                plain_concept_neg[cui]="Not " +short_concept

    with open(file_name, "r") as file:
        # if "UMLS" in file_name.split("/")[-1].split("_"):
            # cui_flag=True
        for line in file:
            concept = line.split(":")[0]
            concept_name = line.split(":")[1].strip()
            concept_name = concept_name.replace('/', ' ').replace('-', ' ')

            if "agegroup" in concept:
                print("age description: ", concept_name)
                if model_type in ["bert", "negbert",'clinicalBert','biobert','pubmedbert']:
                    embedding = get_embedding(concept_name, model_type,tokenizer, model)
                else:
                    embedding = get_embedding(concept_name, model_type)
                concept_name_ = concept + " " + concept_name
                embeddings[concept_name_] = embedding
            if "C0424781" in concept:
                print("C0424781: ", concept_name)
                if model_type in ["bert", "negbert",'clinicalBert','biobert','pubmedbert']:
                    embedding = get_embedding(concept_name, model_type,tokenizer, model)
                else:
                    embedding = get_embedding(concept_name, model_type)
                concept_name_ = concept + " " + concept_name
                embeddings[concept_name_] = embedding
            else:
                if plain_neg:
                    concept_neg = plain_concept_neg[concept]
                else:
                    concept_neg = "Not " + concept_name
                print(concept,"cui_name",concept_name)
                print(concept,"cui_neg",concept_neg)
                if model_type in ["bert", "negbert",'clinicalBert','biobert','pubmedbert']:
                    embedding = get_embedding(concept_name, model_type,tokenizer, model)
                    neg_embedding = get_embedding(concept_neg, model_type,tokenizer, model)
                else:
                    embedding = get_embedding(concept_name, model_type)
                    neg_embedding = get_embedding(concept_neg, model_type)

                concept_name_ = concept + " " + concept_name
                neg_cui_name = concept + " " + concept_neg
                embeddings[concept_name_] = embedding
                embeddings[neg_cui_name] = neg_embedding
    return embeddings

# 
# MODEL_MAP = {
#     "bert": (AutoTokenizer.from_pretrained("bert-base-uncased"), AutoModel.from_pretrained("bert-base-uncased")),
#     "negbert": (AutoTokenizer.from_pretrained('bvanaken/clinical-assertion-negation-bert'), AutoModel.from_pretrained('bvanaken/clinical-assertion-negation-bert')),
#  
# }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str,default="concept_pair.txt", help="File name containing CUIs")
    parser.add_argument("--use_plain_neg", type=bool, default=False, help="use plain neg")
    # parser.add_argument("--model_type", type=str, default="bert", help="Model type")
    # parser.add_argument("--model_type", type=str, default="negbert", help="Model type")
    # parser.add_argument("--model_type", type=str, default="llama", help="Model type")
    parser.add_argument("--model_type", type=str, default="openai", help="Model type")

    # parser.add_argument("--model_type", type=str, default="clinicalBert", help="Model type")
    # parser.add_argument("--model_type", type=str, default="biobert", help="Model type")
    # parser.add_argument("--model_type", type=str, default="negbert", help="Model type")


    args = parser.parse_args()

    cui_embeddings = generate_embeddings(args.file_name, args.model_type,args.use_plain_neg)
    filename = args.file_name.split("/")[-1].split(".")[0]
    # save embeddings
    if "concept" in filename:
        hdf5_file_name = "cui_embeddings/"+f"{filename}_{args.model_type}" + ".h5"
    else:
        hdf5_file_name = "cui_embeddings/"+f"{filename}_{args.model_type}_plainN{args.use_plain_neg}" + ".h5"  # 假设HDF5文件和CUI文件同名但扩展名为.h5
    save_embeddings_to_hdf5(cui_embeddings, hdf5_file_name)

    # print(cui_embeddings['C1978700 Not lab testing ordered (influenza w/other respiratory pathogens panel)'])
    # # print(cui_embeddings.keys())
    # print(cui_embeddings['C1978700 lab testing ordered (influenza w other respiratory pathogens panel)'])

    

if __name__ == "__main__":
    main()
