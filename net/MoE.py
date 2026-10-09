from typing import Any
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias=False):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out

class ES_EE(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def es(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def ee(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)  # .exp()
        if multiply_by_gates:
            if len(stitched.shape) == 2:
                stitched = stitched.mul(self._nonzero_gates)
                zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True,
                                    device=stitched.device)
            else:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
                zeros = torch.zeros(self._gates.size(0), *expert_out[-1].shape[1:], requires_grad=True,
                                    device=stitched.device)

        ensemble = zeros.index_add(0, self._batch_index, stitched.float())
        # ensemble[ensemble == 0] = np.finfo(float).eps
        return ensemble  # .log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class ExpertModel_LL_woCompress(nn.Module):
    def __init__(self, internal_dim, num_heads, kernel_size=3, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(internal_dim, internal_dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(internal_dim * 3, internal_dim * 3, kernel_size=kernel_size, stride=1, padding=1, groups=internal_dim * 3, bias=bias)
        self.project_out = nn.Conv2d(internal_dim, internal_dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out) + x

        return out


class ExpertModel_High_woCompress(nn.Module):
    def __init__(self, internal_dim, out_dim, relu_slope=0.2, use_HIN=True):
        super().__init__()
        self.identity = nn.Conv2d(internal_dim, out_dim, 1, 1, 0)

        self.conv_1 = nn.Conv2d(internal_dim, out_dim, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_3 = nn.Conv2d(internal_dim + internal_dim, out_dim, 3, 1, 1)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_dim // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        return x + resi


class RawFusion(nn.Module):
    def __init__(self, dim, num_head, bias=False):
        super().__init__()
        # mag
        self.fusion = Chanel_Cross_Attention(dim, num_head=num_head, bias=bias)

    def forward(self, raw_x, x):
        out = x + self.fusion(x, raw_x)

        return out


class UncertaintyMoETopK(nn.Module):
    def __init__(self, raw_channel, input_dim, output_dim, num_head=8, num_experts=4, topk=2):
        super().__init__()
        self.raw_channel = raw_channel
        self.num_experts = num_experts
        self.input_dim = input_dim

        Expert_model_LL_args = [
            {'internal_dim': input_dim, 'num_heads': num_head},
            {'internal_dim': input_dim, 'num_heads': num_head},
            {'internal_dim': input_dim, 'num_heads': num_head},
            {'internal_dim': input_dim, 'num_heads': num_head}
        ]

        Expert_model_high_args = [
            {'internal_dim': input_dim, 'out_dim': input_dim},
            {'internal_dim': input_dim, 'out_dim': input_dim},
            {'internal_dim': input_dim, 'out_dim': input_dim},
            {'internal_dim': input_dim, 'out_dim': input_dim}
        ]

        self.ll_list = nn.ModuleList([
            ExpertModel_LL_woCompress(**arg) for arg in Expert_model_LL_args
        ])

        self.el_list = nn.ModuleList([])
        self.el_list.append(self.ll_list)

        for _ in range(3):
            self.el_list.append(nn.ModuleList([
                ExpertModel_High_woCompress(**arg) for arg in Expert_model_high_args
            ]))

        self.noise_std = (1.0 / num_experts) * 1.0

        self.k = topk
        self.softmax = nn.Softmax(1)

        self.gate = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Rearrange('b c 1 1 -> b c'),
                nn.Linear(input_dim, num_experts, bias=False)
            )
            for _ in range(4)
        ])

        self.uncertainty_gate = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(raw_channel, input_dim, 3, 1, 1),  #
                nn.AdaptiveAvgPool2d(1),
                Rearrange('b c 1 1 -> b c'),
                nn.Linear(input_dim, num_experts, bias=False)
            )
            for _ in range(4)
        ])

        self.dwt = DWT()
        self.iwt = IWT()

        # raw_x cnn
        self.raw_cnn = nn.Conv2d(raw_channel, input_dim, kernel_size=3, stride=1, padding=1)

        # wave fusion
        self.fusion = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(input_dim, input_dim, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(input_dim, input_dim, 3, 1, 1)
                ),
                RawFusion(input_dim, num_head=num_head, bias=False)
            ])
            for _ in range(4)
        ])

        self.output_fusion = Chanel_Cross_Attention(input_dim, num_head=num_head, bias=False)
        self.channel_scale = nn.Parameter(torch.full((1, input_dim, 1, 1), 0.5))

    def forward(self, x, raw_x, spa_var, spe_var):
        B, C, H, W = x.shape
        x_wave = self.dwt(x)
        raw_x_wave = self.dwt(self.raw_cnn(F.interpolate(raw_x, (H, W), mode='bilinear')))
        out_wave = []

        aux_loss = 0

        for i, x_i in enumerate(x_wave):
            x_i = self.fusion[i][1](self.fusion[i][0](raw_x_wave[i]), x_i)
            logits = self.gate[i](x_i) + self.uncertainty_gate[i](spa_var + spe_var)  

            if self.training:
                loss_imp = self.importance_loss(logits.softmax(dim=-1), i)

            if self.training:
                noise = torch.randn_like(logits) * self.noise_std
                noisy_logits = logits + noise
            else:
                noisy_logits = logits

            top_k_values, top_k_indices = torch.topk(noisy_logits, self.k, dim=-1)
            top_k_gates = self.softmax(top_k_values)
            w = torch.zeros_like(noisy_logits).to(top_k_gates.dtype).scatter_(1, top_k_indices, top_k_gates)

            # Final auxiliary loss
            if self.training:
                loss_load = self.load_loss(logits, noisy_logits, self.noise_std)
                aux_loss += (0.5 * loss_imp + 0.5 * loss_load)
            else:
                aux_loss += 0

            es_ee = ES_EE(self.num_experts, w)
            expert_inputs = es_ee.es(x_i)
            # gates = es_ee.expert_to_gates()
            expert_outputs = [self.el_list[i][j](expert_inputs[j]) if expert_inputs[j].shape[0] != 0 else expert_inputs[j] for j in range(self.num_experts)]
            y = es_ee.ee(expert_outputs)

            out_wave.append(y)

        y_wave = torch.cat(out_wave, dim=1)  # b 4c h/2 w/2
        y = self.iwt(y_wave)
        y = self.output_fusion(x, y) * self.channel_scale + x * (1 - self.channel_scale)

        return y, aux_loss

    def importance_loss(self, gating_scores, wave_i):
        importance = gating_scores.sum(dim=0)
        imp_mean = importance.mean()
        imp_std = importance.std()
        loss_imp = (imp_std / (imp_mean + 1e-8)) ** 2
        return loss_imp

    def load_loss(self, logits, logits_noisy, noise_std):
        # Compute the noise threshold
        thresholds = torch.topk(logits_noisy, self.k, dim=-1).indices[:, -1]

        # Compute the load for each expert
        threshold_per_item = torch.sum(
            F.one_hot(thresholds, self.num_experts) * logits_noisy,
            dim=-1
        )

        # Calculate noise required to win
        noise_required_to_win = threshold_per_item.unsqueeze(-1) - logits
        noise_required_to_win /= noise_std

        # Probability of being above the threshold
        normal_dist = Normal(0, 1)
        p = 1. - normal_dist.cdf(noise_required_to_win)

        # Compute mean probability for each expert over examples
        p_mean = p.mean(dim=0)

        # Compute p_mean's coefficient of variation squared
        p_mean_std = p_mean.std()
        p_mean_mean = p_mean.mean()
        loss_load = (p_mean_std / (p_mean_mean + 1e-8)) ** 2

        return loss_load
