import os
import json
import copy
import time
import random

from rag.augment import *
from retrieve.retriever.retriever import bm25_retrieve

def process_rag(config):
    """
    Prepare the dataset for FedRAG, consisting of data owned by clients and test data on the server.

    Server test data:
        - Q&A pairs used for evaluation.

    Client data:
        - Each client owns a collection of documents related to the questions in the server test set.
        - Selection strategy: For each test question, retrieve several relevant documents from the DPR corpus and assign them to different clients as their local document collections.
        - Evaluation modes:
            1. In-context FedRAG: Clients perform local retrieval directly.
            2. Parametric FedRAG: Clients train offline and then upload their adapters for aggregation on the server.
    
    Output:
        - config['dir_path']/rag/client_data.json
        - config['dir_path']/rag/server_data.json
    """

    ## Load data and sample Q&A
    with open(f'{config['dir_path']}/train.jsonl', "r", encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    with open(f'{config['dir_path']}/test.jsonl', "r", encoding="utf-8") as f:
        test = [json.loads(line) for line in f]
    total_qa = train + test
    random.seed(config['seed'])
    qas = random.sample(total_qa, config['rag']['samples'])

    ## Retrieve relevant documents for each question as client corpus
    docs = []
    for qa in qas:
        docs.extend(bm25_retrieve(qa['question'], config['rag']['topk']))
        docs = [{"id": i, "document": docs[i]} for i in range(len(docs))]

    ## Save data
    save_dir = f'{config["dir_path"]}/rag'
    os.makedirs(save_dir, exist_ok=True)
    json.dump(docs, open(f'{save_dir}/client_data.json', 'w', encoding='utf-8'), indent=4)
    json.dump(qas, open(f'{save_dir}/server_data.json', 'w', encoding='utf-8'), indent=4)
    
    ## Augment if needed
    augment = False if config['type'] == 'rag' else True
    if augment:
        model_path = config['rag']['augment_model_path']
        augment_model = config['rag']['augment_model']
        model, tokenizer, _ = get_model(model_path)
        generation_config = dict(
            max_new_tokens=512,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            temperature=0.7,
            top_k=50,
        )
        aug_list = []
        for idx, doc in enumerate(docs):
            print(f'\nAugmenting document {idx} / {len(docs)}...')

            begin_time = time.time()
            aug = copy.deepcopy(doc)
            aug[f"{augment_model}_rewrite"] = get_rewrite(doc['document'], model, tokenizer, generation_config)
            qa = get_qa(doc['document'], augment_model, model, tokenizer, generation_config)
            if fix_qa(qa)[0] != False: 
                aug[f"{augment_model}_qa"] = qa
            else:
                print(f'Invalid QA for document {doc["id"]}, skipping...')
                continue   # skip if qa is invalid
            aug_list.append(aug)
            end_time = time.time()

            print(f'Augmentation time cost: {end_time - begin_time}')
        json.dump(aug_list, open(f'{save_dir}/client_data_aug.json', 'w', encoding='utf-8'), indent=4)