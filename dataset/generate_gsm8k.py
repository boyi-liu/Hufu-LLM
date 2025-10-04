import json
import math
import yaml
import os
from rag.precess_rag import process_rag

# 1. read file -> dict, output: two dict (dataset-specific)
# 2. (optional), output: two dict (utils)
# 3. split the two dict into several small dict (utils)
# 4. save the dict (utils)

def split_jsonl(input_file, cfg, train):
    client_num = cfg['client_num']
    dir_path = cfg['dir_path']
    os.makedirs(dir_path, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    total = len(data)
    chunk_size = math.ceil(total / client_num)

    for i in range(client_num):
        start = i * chunk_size
        end = min(start + chunk_size, total)
        chunk = data[start:end]
        if not chunk: break

        os.makedirs(f'{dir_path}/train', exist_ok=True)
        os.makedirs(f'{dir_path}/test', exist_ok=True)
        out_file = f"{dir_path}/train/{i}.json" if train else f"{dir_path}/test/{i}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            for item in chunk:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    
    # TODO: process the file into dict
    if config['type'] == 'ft':
        pass
    else:
        process_rag(config)
        

    # split_jsonl(f"dataset/{dataset_type}/gsm8k/train.jsonl", config, train=True)
    # split_jsonl(f"dataset/{dataset_type}/gsm8k/test.jsonl", config, train=False)