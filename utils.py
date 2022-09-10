import logging
import math
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from functools import partial
import click
from pathlib import Path
import logging
import re
from datamaestro import prepare_dataset

from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import datetime
from pathlib import Path
import time


    
class DeLightBlock(nn.Module):
    """
    Impelementation of the Delight Block
    """

    def __init__(self, dm, nb, wb, glt_shuffle):
        """
        : param dm: Input dimension of the data (embedding size)
        : param nb: Number of layers in the DeLight block
        : param wb: Expansion width in the DeLight block
        
        """
        super(DeLightBlock, self).__init__()
        
        # DeLight block layer
        self.delightTrans = DExTraUnit(dm, dm, dm // 2, dextra_depth=nb, width_multiplier=wb,  max_glt_groups=4, glt_shuffle= glt_shuffle)
        
        # Self-Attention layers
        self.value = nn.Linear(dm // 2, dm // 2)
        self.query = nn.Linear(dm // 2, dm // 2)
        self.key = nn.Linear(dm // 2, dm // 2)

        self.q = nn.Linear(dm // 2, dm)
        
        # Normalization Layer
        self.norm = nn.LayerNorm(dm, eps=1e-5, elementwise_affine=True)
        
        # DeLight FFN layers
        self.FFN1 = nn.Linear(dm, dm // 4)
        self.FFN2 = nn.Linear(dm // 4, dm)

        self.soft = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, x_orig):
        x = torch.transpose(x_orig, 0, 1)
        x = self.delightTrans(x)
        X = torch.transpose(x, 0, 1)
        
        # Compute self-attention
        V = self.value(X)
        Q = self.query(X)
        K = self.key(X)

        V = torch.transpose(V, 0, 1)
        Q = torch.transpose(Q, 0, 1)
        K = torch.transpose(K, 0, 1)
        K = torch.transpose(K, 1, 2)

        weights = torch.matmul(Q, K) / math.sqrt(Q.shape[2])
        weights = self.soft(weights)
        output = torch.transpose(torch.matmul(weights, V), 0, 1)

        output = self.q(output)
        
        # Normalization and link connection
        output_delight = self.norm(self.relu(output) + x_orig)
        
        # DeLight FFN
        output = self.FFN1(output_delight)
        output = self.relu(output)
        
        # Normalization and link connection
        output = self.norm(self.FFN2(output) + output_delight)

        return output

    
class PositionalEncoding(nn.Module):
    "Positional embedding class"
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        :param d_model (int): Dimension des embeddings à générer
        :param max_len (int, optional): Longueur maximale des textes.
                Attention, plus cette valeur est haute, moins bons seront les embeddings de position.
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Ajoute les embeddings de position"""
        x = x + self.pe[:, :x.size(1)]
        return x

class DExTraUnit(nn.Module):
    '''
        This class impelement DeLight Transformation by impelemnting the DeFINE unit and the DeXTRA unit introduced in
        DeFINE unit: https://arxiv.org/abs/1911.12385 and DeXTRA Unit: https://arxiv.org/pdf/2008.00623.pdf
        
    '''

    def __init__(self,
                 in_features: int,
                 in_proj_features: int,
                 out_features: int,
                 width_multiplier: float = 2.0,
                 dextra_depth: int = 4,
                 dextra_dropout: float = 0.1,
                 max_glt_groups: int = 8,
                 act_type: str = 'gelu',
                 norm_type: str = 'ln',
                 use_bias: bool = True,
                 glt_shuffle: bool = False,
                 is_iclr_version: bool = False,
                 *args, **kwargs):
        '''
        :param in_features: Input features
        :param in_proj_features: Projected features for the first layer
        :param out_features: Output features
        :param width_multiplier: Width multiplier. Max. dimension in DExTra or DeFINE is width_multiplier * in_features
        :param dextra_depth: Number of GLT layers
        :param dextra_dropout: Dropout value between GLT layers
        :param max_glt_groups: Max groups in GLT
        :param act_type: Activation function
        :param norm_type: Normalization function
        :param use_bias: Bias or not
        :param glt_shuffle: Feature shuffling in GLT or not
        :param is_iclr_version: Using DeFINE or Dextra (default is Dextra)
        :param args: Unused args
        :param kwargs: Unused kwargs
        '''
        super(DExTraUnit, self).__init__()
        assert dextra_depth > 1, 'We need atleast 2 layers for DeFINE'
        assert in_features % 2 == 0, '# of Input features should be divisible by 2'
        assert in_features % max_glt_groups == 0, '# of Input features ({}) should be divisible by max groups ({})'.format(
            in_features, max_glt_groups)

        self.in_features = in_features
        self.in_proj_features = in_proj_features
        self.out_features = out_features
        self.width_multiplier = width_multiplier
        self.max_features = in_features * self.width_multiplier
        self.num_glt_layers = dextra_depth
        self.max_glt_groups = max_glt_groups
        self.dextra_dropout = dextra_dropout
        self.act_type = act_type
        self.norm_type = norm_type
        self.glt_shuffle = False if is_iclr_version else glt_shuffle  # no shuffling in ICLR version (yes shuffling in DeExtra)

        self.input_layer = get_weight_layer(name='linear',
                                            in_features=self.in_features,
                                            out_features=self.in_proj_features,
                                            use_bias=True,
                                            norm_type=self.norm_type,
                                            act_type=self.act_type,
                                            dropout=self.dextra_dropout
                                            )

        # get config for Group linear transformations
        if is_iclr_version:
            layer_config = self.define_config(in_features=self.in_proj_features,
                                              out_features=self.out_features,
                                              max_features=self.max_features,
                                              n_layers=self.num_glt_layers,
                                              max_groups=self.max_glt_groups)
        else:
            layer_config = self.dextra_config(in_features=self.in_proj_features,
                                              out_features=self.out_features,
                                              max_features=self.max_features,
                                              n_layers=self.num_glt_layers,
                                              max_groups=self.max_glt_groups
                                              )

        # setup expansion and reduction
        dextra_layers = nn.ModuleList()
        groups_next_layer = layer_config['groups'][1:] + [1]

        for idx, (n_in, n_out, g_l, g_l1) in enumerate(zip(layer_config['in'],
                                                           layer_config['out'],
                                                           layer_config['groups'],
                                                           groups_next_layer)):
            wt_layer = get_weight_layer(name='glt', in_features=n_in,
                                        out_features=n_out,
                                        groups=g_l,
                                        use_bias=use_bias,
                                        norm_type=self.norm_type,
                                        act_type=self.act_type,
                                        dropout=self.dextra_dropout,
                                        shuffle=self.glt_shuffle
                                        )

            dextra_layers.append(wt_layer)

        self.output_layer = get_weight_layer(name='linear',
                                             in_features=self.out_features + self.in_proj_features,
                                             out_features=self.out_features,
                                             use_bias=True,
                                             norm_type=norm_type,
                                             act_type=act_type,
                                             dropout=dextra_dropout
                                             )

        self.dextra_layers = dextra_layers
        self.groups_per_layer = layer_config['groups']

    def __repr__(self):
        s = '{name}(in_features={in_features}, out_features={out_features}, width_multiplier={width_multiplier}, ' \
            'normalization={norm_type}, activation={act_type}, dextra_dropout={dextra_dropout})'
        s += '\n  \t |---- {}'.format(self.input_layer)
        for layer_name in self.dextra_layers:
            s += '\n  \t |---- {}'.format(layer_name)
        s += '\n  \t |---- {}'.format(self.output_layer)
        return s.format(name=self.__class__.__name__, **self.__dict__)

    @staticmethod
    def dextra_config(in_features, out_features, max_features, n_layers, max_groups):

        mid_point = int(math.ceil(n_layers / 2.0))
        # decide number of groups per layer
        groups_per_layer = [min(2 ** (i + 1), max_groups) for i in range(mid_point)]

        # divide the space linearly between input_features and max_features
        output_sizes = np.linspace(in_features, max_features, mid_point, dtype=np.int).tolist()
        # invert lists to get the reduction groups and sizes
        inv_output_sizes = output_sizes[::-1]
        inv_group_list = groups_per_layer[::-1]
        if n_layers % 2 == 0:
            # even
            groups_per_layer = groups_per_layer + inv_group_list
            output_sizes = output_sizes + inv_output_sizes
        else:
            # for odd case,
            groups_per_layer = groups_per_layer + inv_group_list[1:]
            output_sizes = output_sizes + inv_output_sizes[1:]

        assert len(output_sizes) == len(groups_per_layer), '{} != {}'.format(len(output_sizes), len(groups_per_layer))
        output_sizes = output_sizes[:-1]

        # ensure that output and input sizes are divisible by group size
        input_sizes = [1] * len(groups_per_layer)
        input_sizes[0] = in_features
        for i in range(n_layers - 1):
            # output should be divisible by ith groups as well as i+1th group
            # Enforcing it to be divisble by 8 so that we can maximize tensor usage
            g_l = max(groups_per_layer[i + 1], groups_per_layer[i], 8)
            out_dim_l = int(math.ceil(output_sizes[i] / g_l)) * g_l
            inp_dim_l1 = out_dim_l + in_features

            if out_dim_l % 8 != 0:
                print_warning_message(
                    'To maximize tensor usage, output dimension {} should be divisible by 8'.format(out_dim_l))

            if inp_dim_l1 % 8 != 0:
                print_warning_message(
                    'To maximize tensor usage, input dimension {} should be divisible by 8'.format(inp_dim_l1))

            input_sizes[i + 1] = inp_dim_l1
            output_sizes[i] = out_dim_l

        # add dimensions corresponding to reduction step too
        output_sizes = output_sizes + [out_features]

        return {'in': input_sizes,
                'out': output_sizes,
                'groups': groups_per_layer
                }

    @staticmethod
    def define_config(in_features, out_features, max_features, n_layers, max_groups):
        # decide number of groups per layer
        groups_per_layer = []
        counter = 0
        for i in range(n_layers):
            g = 2 ** counter
            if g <= max_groups:
                counter += 1
            else:
                # reset
                g = 1  # set the current groups to 1
                counter = 1  # so that next group has 2 groups
            groups_per_layer.append(g)

        groups_per_layer = groups_per_layer[::-1]

        # divide the space linearly between input_features and max_features
        output_sizes = np.linspace(in_features, max_features, n_layers)
        output_sizes = output_sizes.astype(np.int).tolist()
        output_sizes = output_sizes[1:]

        # ensure that output and input sizes are divisible by group size
        input_sizes = [1] * len(groups_per_layer)
        input_sizes[0] = in_features
        for i in range(n_layers - 1):
            # output should be divisible by ith groups as well as i+1th group
            g_l = max(groups_per_layer[i + 1], groups_per_layer[i], 8)
            out_dim_l = int(math.ceil(output_sizes[i] / g_l)) * g_l
            inp_dim_l1 = out_dim_l + in_features

            if out_dim_l % 8 != 0:
                print_warning_message(
                    'To maximize tensor usage, output dimension {} should be divisible by 8'.format(out_dim_l))

            if inp_dim_l1 % 8 != 0:
                print_warning_message(
                    'To maximize tensor usage, input dimension {} should be divisible by 8'.format(inp_dim_l1))

            input_sizes[i + 1] = inp_dim_l1
            output_sizes[i] = out_dim_l

        # add dimensions corresponding to reduction step too
        output_sizes = output_sizes + [out_features]

        return {'in': input_sizes,
                'out': output_sizes,
                'groups': groups_per_layer
                }

    def forward_dextra(self, x):
        '''
        T -- > time steps
        B --> Batch size
        N, M --> Input, output features
        :param x: Input is [TxBxN] or [BxTxN]
        :return: output is [TxBxM] or [BxTxM]
        '''
        B = x.size(0)
        T = x.size(1)

        out = x

        for i, layer_i in enumerate(self.dextra_layers):
            # Transform Layer
            out = layer_i(out)

            g_next_layer = self.groups_per_layer[i + 1] if i < self.num_glt_layers - 1 else 1
            if g_next_layer == 1:
                # Linear layer is connected to everything so shuffle and split is useless for G=1
                out = torch.cat([x, out], dim=-1)
            else: ###### function H of paper #######
                # SPLIT and MIX LAYER
                # [B x T x M] --> [B x T x  G x M/G]
                x_g = x.contiguous().view(B, T, g_next_layer, -1)

                out = out.contiguous().view(B, T, g_next_layer, -1)

                # [B x T x G x M / G] || [B x T x G x N/G] --> [B x T x G x (N+M)/G]
                out = torch.cat([x_g, out], dim=-1)  # Input Mixer

                # [B x T x G x N+ M/G] --> [B x T x N + M]
                out = out.contiguous().view(B, T, -1)

        out = self.output_layer(out)
        return out

    def forward(self, x):
        '''
        :param x: Input is [B x T x N] (B: batch size, T: input size (max sequence len), N: emb size)
        :return: Output is [B x T x M]
        '''

        # process input
        x = self.input_layer(x)
        n_dims = x.dim()

        if n_dims == 2:
            # [B x N] --> [B x 1 x N]
            x = x.unsqueeze(dim=1)  # add dummy T dimension
            # [B x 1 x N] --> [B x 1 x M]
            x = self.forward_dextra(x)
            # [B x 1 x M] --> [B x M]
            x = x.squeeze(dim=1)  # remove dummy T dimension
        elif n_dims == 3:
            x = self.forward_dextra(x)
        else:
            raise NotImplementedError
        return x

    def compute_macs_params(self):
        macs = 0
        n_params = 0

        macs_params_in = self.input_layer.compute_macs_params()
        macs += macs_params_in['macs']
        n_params += macs_params_in['params']

        macs_params_out = self.output_layer.compute_macs_params()
        macs += macs_params_out['macs']
        n_params += macs_params_out['params']

        for layer in self.dextra_layers:
            macs_params_define = layer.compute_macs_params()
            macs += macs_params_define['macs']
            n_params += macs_params_define['params']

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }

class GroupLinear(nn.Module):
    '''
        This class implements the Grouped Linear Transform
        This is based on the Pyramidal recurrent unit paper:
            https://arxiv.org/abs/1808.09029
    '''

    def __init__(self, in_features: int, out_features: int, n_groups: int = 4, use_bias: bool = False,
                 use_shuffle: bool = False,
                 norm_type: Optional[str] = None, dropout: float = 0.0, act_type: Optional[str] = None):
        '''
        :param in_features: number of input features
        :param out_features: number of output features
        :param n_groups: number of groups in GLT
        :param use_bias: use bias or not
        :param use_shuffle: shuffle features between different groups
        :param norm_type: Normalization type (e.g. LayerNorm)
        :param dropout: Dropout value (default is 0.0)
        :param act_type: Activation type (e.g., Gelu or ReLU)
        '''
        super(GroupLinear, self).__init__()

        if in_features % n_groups != 0:
            err_msg = "Input dimensions ({}) must be divisible by n_groups ({})".format(in_features, n_groups)
            print_error_message(err_msg)
        if out_features % n_groups != 0:
            err_msg = "Output dimensions ({}) must be divisible by n_groups ({})".format(out_features, n_groups)
            print_error_message(err_msg)

        # warning_message = 'Please install custom cuda installation for faster training and inference'

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        self.weights = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if use_bias:
            # add 1 in order to make it broadcastable across batch dimension
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        if norm_type is not None:
            self.normalization_fn = get_norm_layer(name=norm_type, out_features=out_groups)
            self.norm_type = norm_type
        else:
            self.normalization_fn = None
            self.norm_type = None

        self.use_dropout = False
        self.drop_p = dropout
        if dropout > 0:
            self.drop_layer = nn.Dropout(p=dropout)
            self.use_dropout = True

        if act_type is not None:
            self.act_fn = get_activation_layer(name=act_type)
            self.act_type = act_type
        else:
            self.act_fn = None
            self.act_type = None

        self.n_groups = n_groups
        self.use_bias = use_bias
        self.shuffle = use_shuffle
        self.feature_shuffle = True if use_shuffle else False

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def process_input_bmm(self, x):
        '''
        N --> Input dimension
        M --> Output dimension
        g --> groups
        G --> gates
        :param x: Input of dimension B x N
        :return: Output of dimension B x M
        '''
        bsz = x.size(0)
        # [B*T x N] --> [B*T x g  x N/g]
        x = x.contiguous().view(bsz, self.n_groups, -1)
        # [B*T x g x N/g] --> [g x B*T  x N/g]
        x = x.transpose(0, 1)  # transpose so that group is first

        # [g x B*T x N/g] x [g x N/g x M/g] --> [g x B*T x M/g]
        x = torch.bmm(x, self.weights)  # multiply with Weights

        # add bias
        if self.use_bias:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g x B*T x M/g] --> [B*T x M/g x g]
            x = x.permute(1, 2, 0)
            # [B*T x M/g x g] --> [B*T x g x M/g]
            x = x.contiguous().view(bsz, self.n_groups, -1)
        else:
            # [g x B*T x M/g] --> [B*T x g x M/g]
            x = x.transpose(0, 1)  # transpose so that batch is first

        # feature map normalization
        if self.normalization_fn is not None:
            x = self.normalization_fn(x)

        # feature map activation (or thresholding)
        if self.act_fn is not None:
            x = self.act_fn(x)

        return x

    def forward(self, x):
        '''
        :param x: Input of shape [T x B x N] (should work with [B x T x N])
        :return:
        '''
        if x.dim() == 2:
            x = self.process_input_bmm(x)
        elif x.dim() == 3:
            T, B, N = x.size()
            x = x.contiguous().view(B * T, -1)
            x = self.process_input_bmm(x)
            x = x.contiguous().view(T, B, -1)
        else:
            raise NotImplementedError

        # dropout
        if self.use_dropout:
            x = self.drop_layer(x)
        return x

    def __repr__(self):
        s = '{name}(in_features={in_features}, out_features={out_features}, num_groups={n_groups}'
        if self.use_bias:
            s += ', bias={use_bias}'
        if self.shuffle:
            s += ', shuffle={shuffle}'

        if self.norm_type is not None:
            s += ', norm_type={norm_type}'
        if self.act_type is not None:
            s += ', act_type={act_type}'
        if self.drop_p > 0.0:
            s += ', drop_p={drop_p}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def compute_macs_params(self):
        '''
            # of operations in group linear transformation (GLT) are given as:
            Let N and M be dimensions of the input and the output tensor
            Both input and output are split into G groups, so that each input and output group has dimension of N/G and M/G
            Each input group of dimension N/G is mapped to each output group of dimension M/G using a matrix with dimensions [N/G x M/G].
            This mapping involves NM/G^2 additions and NM/G^2 multiplications.
            Since, there are G such groups, we will have total of NM/G addiations and NM/G multipplications.
            Or in simple words, total multiplication-additions (MACs) would be NM/G and FLOPs would be 2NM/G.
            Relationship with # of parameters:
            We have G matrices, each of dimension [N/G x M/G]. The number of parameters in each matrix is NM/G^2.
            Therefore, the total number of parameters in GLT is NM/G.
            MACs = parameters
        '''
        n_mul_wt = self.weights.numel()
        n_add_bias = self.bias.numel() if self.use_bias else 0
        macs = n_mul_wt + n_add_bias
        n_params = n_mul_wt + n_add_bias

        if self.normalization_fn is not None:
            n_params += sum([p.numel() for p in self.normalization_fn.parameters()])
            # MACS are zero for LayerNorm because they can be fused

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }
    

class Linear(nn.Module):
    '''
    This class implements the fully connected layer
    '''

    def __init__(self, in_features, out_features, use_bias=True, num_gates=1,
                 norm_type=None, dropout=0.0, act_type=None):
        '''
        :param in_features: number of input features
        :param out_features: number of output features (first layer is dm)
        :param use_bias: use bias or not
        :param num_gates: number of gates (useful if you want to use it within gating structures, like LSTMs)
        :param norm_type: Normalization type (e.g. LayerNorm)
        :param dropout: Dropout value (default is 0.0)
        :param act_type: Activation type (e.g., Gelu or ReLU)
        '''
        super(Linear, self).__init__()

        self.weights = torch.nn.Parameter(torch.Tensor(out_features * num_gates, in_features))
        if use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features * num_gates))
        else:
            self.bias = None

        if norm_type is not None:
            self.normalization_fn = get_norm_layer(name=norm_type, out_features=out_features * num_gates)
            self.norm_type = norm_type
        else:
            self.normalization_fn = None
            self.norm_type = None

        self.use_dropout = False
        self.drop_p = dropout
        if dropout > 0:
            self.drop_layer = nn.Dropout(p=dropout)
            self.use_dropout = True

        if act_type is not None:
            self.act_fn = get_activation_layer(name=act_type)
            self.act_type = act_type
        else:
            self.act_fn = None
            self.act_type = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.gates = num_gates
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, x):
        '''
        :param x: Input
        :return: Output
        '''
        x = F.linear(x, weight=self.weights, bias=self.bias)
        # feature map normalization
        if self.normalization_fn is not None:
            x = self.normalization_fn(x)

        # feature map activation (or thresholding)
        if self.act_fn is not None:
            x = self.act_fn(x)

        # recurrent dropout
        if self.use_dropout:
            x = self.drop_layer(x)

        return x

    def __repr__(self):
        s = '{name}(in_features={in_features}, out_features={out_features}'

        if self.use_bias:
            s += ', bias={use_bias}'

        if self.gates > 1:
            s += ', gates={gates}'

        if self.norm_type is not None:
            s += ', norm_type={norm_type}'
        if self.act_type is not None:
            s += ', act_type={act_type}'
        if self.drop_p > 0.0:
            s += ', drop_p={drop_p}'
        s += ')'

        return s.format(name=self.__class__.__name__, **self.__dict__)

    def compute_macs_params(self):
        '''
        # of operations in LT are given as:
            Let N and M be dimensions of the input and the output tensor
            Input dimension N is mapped to output of dimension M using a matrix with dimensions [N x M].
            This conversion will involve NM additions and NM multiplications.
            Or in simple words, total multiplication-additions (MACs) would be NM and FLOPs would be 2NM.
            Relationship with # of parameters:
            We have a matrix of dimension [N x M]. The number of parameters is NM.
            Therefore, the total number of parameters in LT is NM.
            MACs = parameters and FLOPs = 2 * parameters
        '''
        n_mul_wt = self.weights.numel()
        n_add_bias = self.bias.numel() if self.use_bias else 0
        macs = n_mul_wt + n_add_bias
        n_params = n_mul_wt + n_add_bias

        if self.normalization_fn is not None:
            n_params += sum([p.numel() for p in self.normalization_fn.parameters()])
            # MACS are zero for LayerNorm because they can be fused

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }

def get_weight_layer(name: str, in_features: int, out_features: int, groups: int = 4, use_bias: bool = True,
                     gates: int = 1, shuffle: bool = False,
                     norm_type: Optional[str] = None, dropout: float = 0.0, act_type: Optional[str] = None):
    """
    Functon that outputs regular linear layer or glt layer
    """
    # Group linear transform with groups=1 is the same as Linear Transformation
    if name == 'glt' and groups == 1:
        name = 'linear'

    if name == 'linear':
        layer = Linear(in_features=in_features, out_features=out_features, use_bias=use_bias, num_gates=gates,
                       norm_type=norm_type, dropout=dropout, act_type=act_type)
    elif name == 'glt':
        layer = GroupLinear(in_features=in_features, out_features=out_features, n_groups=groups,
                            use_bias=use_bias, use_shuffle=shuffle, norm_type=norm_type,
                            dropout=dropout, act_type=act_type)
    else:
        raise NotImplementedError
    return layer


def get_embedding_layer(num_embeddings, embedding_dim, padding_idx=None):
    """
    Function that outputs embedding layer
    """
    emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx)
    # initialize embedding layer
    nn.init.normal_(emb.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(emb.weight[padding_idx], 0)
    return emb

class GELU(torch.nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)

class Swish(torch.nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)

activation_list = [
    'relu', 'leaky', 'selu', 'elu', 'celu', 'prelu', 'sigmoid', 'tanh', 'gelu', 'swish'
]

def get_activation_layer(name):
    if name == 'relu':
        return nn.ReLU(inplace=False)
    elif name == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    elif name == 'elu':
        return nn.ELU(inplace=True)
    elif name == 'celu':
        return nn.CELU(inplace=True)
    elif name == 'prelu':
        return nn.PReLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'gelu':
        return GELU()
    elif name =='swish':
        return Swish()
    else:
        print_error_message('Supported activation functions: {}'.format(activation_list))
        return None

class BatchNorm(nn.Module):
    """
    Classs of 1D BatchNormalization
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(BatchNorm, self).__init__()
        self.layer = nn.BatchNorm1d(num_features=num_features, eps=eps, affine=affine)

    def forward(self, x):
        if x.dim() == 3:
            bsz, seq_len, feature_size = x.size()
            out = self.layer(x.view(-1, feature_size))
            return out.contiguous().view(bsz, seq_len, -1)
        else:
            return self.layer(x)

norm_layer_list = [
    'gn', 'bn', 'ln'
]

def get_norm_layer(name, out_features, num_groups=1, eps=1e-5, affine=True):
    """
    Class that outputs a normalization layer from: Batch Normalization, Layer Normalization and Group Normalization
    """
    if name == 'gn' and num_groups == 1:
        name = 'bn'

    if name == 'bn':
        return BatchNorm(num_features=out_features, eps=eps, affine=affine)
    elif name == 'ln':
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(out_features, eps, affine)
        except:
            return nn.LayerNorm(out_features, eps=eps, elementwise_affine=affine)
    elif name == 'gn':
        return nn.GroupNorm(num_groups=num_groups, num_channels=out_features, eps=eps, affine=affine)
    else:
        print_error_message('Supported normalization functions: {}'.format(norm_layer_list))
        return None

text_colors = {
               'logs': '\033[34m', # 033 is the escape code and 34 is the color code
               'info': '\033[32m',
               'warning': '\033[33m',
               'error': '\033[31m',
               'bold': '\033[1m',
               'end_color': '\033[0m'
               }


def get_curr_time_stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_error_message(message):
    """
    Error dispkay function
    """
    time_stamp = get_curr_time_stamp()
    error_str = text_colors['error'] + text_colors['bold'] + 'ERROR  ' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, error_str, message))
    print('{} - {} - {}'.format(time_stamp, error_str, 'Exiting!!!'))
    exit(-1)

def print_warning_message(message):
    time_stamp = get_curr_time_stamp()
    warn_str = text_colors['warning'] + text_colors['bold'] + 'WARNING' + text_colors['end_color']
    print('{} - {} - {}'.format(time_stamp, warn_str, message))

