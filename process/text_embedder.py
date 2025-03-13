"""
Auto-generate text embedding with llm-encoder
"""

#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import torch
from transformers import AutoModel, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Generate text embeddings for given prompts.")
    parser.add_argument('--prompt_type', type=str, required=True, help='Prompt type (e.g., 9 or p9)')
    parser.add_argument('--dataset_number', type=str, required=True, help='Dataset number (e.g., 2 or 002)')
    parser.add_argument('--model', type=str, default='nvembed-v2', help='Embedding model name (default: nvembed-v2)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device number (default: 0)')
    args = parser.parse_args()

    prompt_type = args.prompt_type
    dataset_number = f"{int(args.dataset_number):03d}" 
    model_name = args.model
    cuda_device = args.device

    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = os.path.abspath(os.path.join(script_dir, "../data"))
    dataset_dir_name = f"Dataset{dataset_number}_"
    nnunet_raw_path = os.path.join(base_data_path, 'nnUNet_raw')
    
    # Find dataset directory
    dataset_dirs = [d for d in os.listdir(nnunet_raw_path) if d.startswith(dataset_dir_name)]
    if not dataset_dirs:
        print(f"Dataset directory starting with {dataset_dir_name} not found in {nnunet_raw_path}")
        return
    dataset_name = dataset_dirs[0]
    dataset_path = os.path.join(nnunet_raw_path, dataset_name)
    dataset_json_path = os.path.join(dataset_path, 'dataset.json')

    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)

    txt_embed_dim = 4096
    if model_name == "nvembed-v2":
        model_name_full = 'nvidia/NV-Embed-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name_full, use_fast=True)
    text_encoder = AutoModel.from_pretrained(model_name_full, trust_remote_code=True)
    text_encoder.to(device)
    max_length = 32

    for name, param in text_encoder.named_parameters():
        param.requires_grad_(False)

    def get_text_embedding(txt):
        inputs = tokenizer([txt], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = text_encoder(**inputs)
            emb_txt = outputs['sentence_embeddings']
        return emb_txt.squeeze(0)

    for split in ['imagesTr', 'imagesTs']:
        image_entries = dataset_info.get(split, [])
        for entry in image_entries:
            image_name = entry['image_name']  # Example: 'cvc001_0000'
            prompts = entry['prompts']
            
            if not prompt_type.startswith('p'):
                prompt_key = f'p{prompt_type}'
            else:
                prompt_key = prompt_type
            if prompt_key not in prompts:
                print(f"Prompt type {prompt_key} not found in prompts for {image_name}")
                continue
            prompt_value = prompts[prompt_key]
            if isinstance(prompt_value, list):
                text = prompt_value[0]
            else:
                text = prompt_value

            emb_txt = get_text_embedding(text)

            output_dir = os.path.join(base_data_path, 'nnUNet_preprocessed', dataset_name, f'{split}_p{prompt_type}_embeddings')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'{image_name}.pt')
            torch.save(emb_txt.cpu(), output_path)
            print(f"Saved embedding for {image_name} at {output_path}")

if __name__ == '__main__':
    main()
