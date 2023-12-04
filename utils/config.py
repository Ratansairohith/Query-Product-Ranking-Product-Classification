from utils.get_device import return_device

device = return_device()

config_params = {
    'device': device,
    'debug': False,
    'checkpoint': 'microsoft/infoxlm-base',
    'output_logits': 768,
    'max_len': 128,
    'num_folds': 5,
    'batch_size': 1,
    'dropout': 0.2,
    'num_workers': 2,
    'epochs': 10,
    'lr': 2e-5,
    'margin': 0.7,
    'scheduler_name': 'OneCycleLR',
    'max_lr': 5e-5,                 # OneCycleLR
    'pct_start': 0.1,               # OneCycleLR
    'anneal_strategy': 'cos',       # OneCycleLR
    'div_factor': 1e3,              # OneCycleLR
    'final_div_factor': 1e3,        # OneCycleLR
    'no_decay': True
}