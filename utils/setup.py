import sys
import os

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from modify_arch.mspoe_models import MsPoELlamaForCausalLM, MsPoEGemmaForCausalLM, MsPoEQwen2ForCausalLM, \
    MsPoEMistralForCausalLM


def setup_models(args, attn_implementation="flash_attention_2"):
    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)

    if args.enable_ms_poe:
        print('Using Ms-PoE Positional Embedding')
        config.apply_layers = list(int(x) for x in args.apply_layers)
        config.compress_ratio_min = args.compress_ratio_min
        config.compress_ratio_max = args.compress_ratio_max
        config.head_type = args.head_type
        print('Compress Ratio: from {} to {}'.format(config.compress_ratio_min, config.compress_ratio_max))
        if "mistral" in args.model_name.lower():
            Model = MsPoEMistralForCausalLM
        elif "gemma" in args.model_name.lower():
            Model = MsPoEGemmaForCausalLM
        elif "qwen" in args.model_name.lower():
            Model = MsPoEQwen2ForCausalLM
        else:
            Model = MsPoELlamaForCausalLM
        model = Model.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir, device_map="auto",
                                      torch_dtype="auto", attn_implementation=attn_implementation)
    else:
        print('Using the Baseline Model')
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, device_map="auto",
                                                     torch_dtype="auto", attn_implementation=attn_implementation)

    return config, tokenizer, model
