import yaml

def load_yaml(args):
    
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    for k, v in config.items():
        setattr(args, k, v)
    print(args)
    return args