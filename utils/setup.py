


from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


from utils.modify_arch.llama import MsPoELlamaForCausalLM



def setup_models(args):

    config = AutoConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)

    if args.enable_ms_poe:
        print('Using Ms-PoE Positional Embedding')
        config.apply_layers = list(int(x) for x in args.apply_layers.split(','))
        config.compress_ratio_min = args.compress_ratio_min
        config.compress_ratio_max = args.compress_ratio_max
        config.head_type = args.head_type
        print('Compress Ratio: from {} to {}'.format(config.compress_ratio_min, config.compress_ratio_max))
        model = MsPoELlamaForCausalLM.from_pretrained(args.model_name, config=config, cache_dir=args.cache_dir)
    else:
        print('Using the Baseline Model')
        model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    return config, tokenizer, model