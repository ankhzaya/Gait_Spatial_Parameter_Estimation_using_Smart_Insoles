import torch

import sys
sys.path.append('../')

from models.conformerregressor import ConformerRegressor

def create_model(configs):

    if configs.arch == 'conformer':
        model = ConformerRegressor(input_c=12, d_model=128, heads=4,
                                   ff_dim=256, conv_k=31, layers=4,
                                   out_dim=8, dropout=0.1)
        return model

if __name__ == '__main__':
    from configs.configs import parse_configs

    configs = parse_configs()

    model = create_model(configs).to(device=configs.device)
    print('arch: {}'.format(configs.arch))
    sample_input = torch.randn((4, 256, 12)).to(device=configs.device)

    out, _ = model(sample_input, return_attn=False)

    print('out shape: {}'.format(out.shape))