import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, d_model, kernels=(7, 31)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=k//2)
            for k in kernels
        ])
        self.proj = nn.Linear(in_channels * len(kernels), d_model)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1,2)  # [B, C, T]
        outs = [conv(x) for conv in self.convs]
        x = torch.cat(outs, dim=1)     # [B, C*len, T]
        x = x.transpose(1,2)           # [B, T, C*len]
        return self.proj(x)            # [B, T, d_model]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:,:x.size(1),:]


class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1  = nn.Conv1d(d_model, 2*d_model, 1)
        self.dw   = nn.Conv1d(d_model, d_model, kernel_size,
                              padding=kernel_size//2, groups=d_model)
        self.bn   = nn.BatchNorm1d(d_model)
        self.pw2  = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.glu      = nn.GLU(dim=1)

    def forward(self, x):
        res = x
        x = self.norm(x).transpose(1,2)
        x = self.pw1(x)
        x = self.glu(x)
        x = self.dw(x)
        x = F.silu(self.bn(x))
        x = self.pw2(x)
        x = self.dropout(x).transpose(1,2)
        return res + x


class ConformerBlock(nn.Module):
    def __init__(self, d_model, heads, ff_dim, conv_kernel, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout)
        )
        self.mha  = nn.MultiheadAttention(d_model, heads, dropout=dropout,
                                          batch_first=True)
        self.norm_mha = nn.LayerNorm(d_model)
        self.drop_mha = nn.Dropout(dropout)
        self.conv_mod = ConformerConvModule(d_model, conv_kernel, dropout)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout)
        )
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x, return_attn=False):
        # x: [B, T, d_model]
        x = x + 0.5*self.ffn1(x)
        res = x
        x_norm = self.norm_mha(x)
        attn_out, attn_w = self.mha(x_norm, x_norm, x_norm,
                                    need_weights=True,
                                    average_attn_weights=False)
        x = res + self.drop_mha(attn_out)
        x = self.conv_mod(x)
        x = x + 0.5*self.ffn2(x)
        x = self.norm_final(x)
        return (x, attn_w) if return_attn else x


class ConformerEncoder(nn.Module):
    def __init__(self, layers, d_model, heads, ff_dim, conv_kernel, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, heads, ff_dim, conv_kernel, dropout)
            for _ in range(layers)
        ])

    def forward(self, x, return_attn=False):
        attns = []
        for i, blk in enumerate(self.layers):
            if return_attn:
                x, w = blk(x, return_attn=True)
                attns.append(w)
            else:
                x = blk(x)
        return (x, attns) if return_attn else x


class ConformerRegressor(nn.Module):
    def __init__(self,
                 input_c=12, d_model=128, heads=4,
                 ff_dim=256, conv_k=31, layers=4,
                 out_dim=6, dropout=0.1):
        super().__init__()
        # self.multi_scale_conv = MultiScaleConv(input_c, d_model, kernels=(7, 31))
        self.embed = nn.Linear(input_c, d_model)
        self.pos   = PositionalEncoding(d_model)

        # reduced layers includes deep supervision point
        self.encoder = ConformerEncoder(layers, d_model, heads,
                                         ff_dim, conv_k, dropout)
        # auxiliary head for deep supervision
        self.aux_head = nn.Linear(d_model, out_dim)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.head  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )
        self.calibrate = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, x, return_attn=False):
        B, T, _ = x.shape
        x = self.embed(x)         # [B, T, d_model]
        x = self.pos(x)

        # deep supervision after layer 2
        attns = []
        x_temp = x
        aux = None
        for i, layer in enumerate(self.encoder.layers):
            x_temp = layer(x_temp)
            if i == 1:
                # pool and predict auxiliary at layer 2
                feat = self.pool(x_temp.transpose(1,2)).squeeze(-1)  # [B, d_model]
                aux = self.aux_head(feat)  # [B, out_dim]
        if return_attn:
            # full forward with attention
            x_out, attns = self.encoder(x, return_attn=True)
        else:
            x_out = x_temp

        # main head prediction
        x_out = x_out.transpose(1,2)  # [B, d_model, T]
        x_out = self.pool(x_out).squeeze(-1)  # [B, d_model]
        y = self.calibrate(self.head(x_out))  # [B, out_dim]

        if return_attn:
            return (y, aux), attns
        return y, aux


if __name__ == "__main__":
    B, T, C = 32, 256, 12
    model = ConformerRegressor(input_c=C, d_model=128, heads=4,
                               ff_dim=256, conv_k=31, layers=4,
                               out_dim=8, dropout=0.1)
    x = torch.randn(B, T, C)
    (y, aux), attn = model(x, return_attn=True)
    print(y.shape, aux.shape)
    print([w.shape for w in attn])  # attn list per layer

    # number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
