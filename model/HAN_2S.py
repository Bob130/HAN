"""
HAN_ShareInMapRes_2stream
"""

import torch.nn as nn
import torch
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
import copy


class HAN(nn.Module):
    def __init__(self, num_classes, dp_rate, input_frames, raw_input_dim, dataset, device):
        super(HAN, self).__init__()

        h_dim = 32
        h_num = 8

        self.device = device

        if dataset == 'FPHA':
            # https://github.com/guiggh/hand_pose_action
            # [0Wrist, 1TMCP, 2IMCP, 3MMCP, 4RMCP, 5PMCP, 6TPIP, 7TDIP, 8TTIP, 9IPIP, 10IDIP, 11ITIP, 12MPIP, 13MDIP,
            # 14MTIP, 15RPIP, 16RDIP, 17RTIP, 18PPIP, 19PDIP, 20PTIP]
            self.finger_list = [[0],
                                [1, 6, 7, 8],
                                [2, 9, 10, 11],
                                [3, 12, 13, 14],
                                [4, 15, 16, 17],
                                [5, 18, 19, 20]]
        else:
            # http://www-rech.telecom-lille.fr/shrec2017-hand/
            # 1.Wrist, 2.Palm, 3.thumb_base, 4.thumb_first_joint, 5.thumb_second_joint, 6.thumb_tip, 7.index_base,
            # 8.index_first_joint, 9.index_second_joint, 10.index_tip, 11.middle_base, 12.middle_first_joint,
            # 13.middle_second_joint, 14.middle_tip, 15.ring_base, 16.ring_first_joint, 17.ring_second_joint, 18.ring_tip,
            # 19.pinky_base, 20.pinky_first_joint, 21.pinky_second_joint, 22.pinky_tip.
            self.finger_list = [[0, 1],
                                [2, 3, 4, 5],
                                [6, 7, 8, 9],
                                [10, 11, 12, 13],
                                [14, 15, 16, 17],
                                [18, 19, 20, 21]]
        self.raw_input_dim = raw_input_dim

        self.BN_HPEV = nn.BatchNorm2d(3)
        self.BN_HMM = nn.BatchNorm2d(3)
        self.BN_FRPV = nn.BatchNorm1d(96)

        self.input_map_HPEV = nn.Sequential(
            nn.Linear(self.raw_input_dim, 128),
        )
        self.input_map_HMM = nn.Sequential(
            nn.Linear(self.raw_input_dim, 128),
        )

        # HPEV stream
        # self.layers = clones(self.input_map, 6)

        # input_size, h_num, h_dim, dp_rate, time_len, domain
        self.joint_att_HPEV = ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                                        domain="spatial", time_len=input_frames)
        self.finger_att_HPEV = ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                                         domain="temporal", time_len=input_frames)
        self.temporal_att_HPEV = ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                                           domain="temporal", time_len=input_frames)
        self.fusion_att_HPEV = ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                                         domain="temporal", time_len=input_frames)
        self.cls_HPEV = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(128*2, num_classes),
        )

        # FRPV
        self.fc_FPRV = nn.Sequential(
            nn.Linear(96, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        # HMM stream
        self.finger_att_HMM = ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                                        domain="temporal", time_len=input_frames)
        self.temporal_att_HMM = ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                                          domain="temporal", time_len=input_frames)
        self.fusion_att_HMM = ATT_Layer(input_size=128, output_size=128, h_num=h_num, h_dim=h_dim, dp_rate=dp_rate,
                                        domain="temporal", time_len=input_frames)
        self.cls_HMM = nn.Sequential(
            nn.Dropout(dp_rate),
            nn.Linear(128, num_classes),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        HPEV = input['HPEV']
        HMM  = input['HMM']
        FRPV = input['FRPV']

        HPEV = HPEV.to(self.device)
        HMM = HMM.to(self.device)
        FRPV = FRPV.to(self.device)
        # label = label.to(self.device)

        # input shape: [batch_size, time_len, joint_num, self.raw_input_dim]

        time_len = HPEV.shape[1]
        joint_num = HPEV.shape[2]

        # HPEV
        x = HPEV
        x = x.permute(0, 3, 1, 2)
        x = self.BN_HPEV(x)
        x = x.permute(0, 2, 3, 1)

        # reshape x
        # x = x.reshape(-1, time_len * joint_num, self.raw_input_dim)

        finger_feature_HPEV = [torch.mean(self.joint_att_HPEV(self.input_map_HPEV(x[:, :, finger, :])), -2)
                               for finger in self.finger_list]

        # for linear, finger in zip(self.layers, self.finger_list):
        #     x_t = x[:, :, finger, :]
        #     x_t = linear(x_t)
        #     x_t = self.joint_att(x_t)
        #     x_t = torch.mean(x_t, -2)

        hand_feature_HPEV = torch.stack(finger_feature_HPEV, -2)
        hand_feature_HPEV = torch.mean(self.finger_att_HPEV(hand_feature_HPEV), -2)

        finger_temporal_feature_HPEV = [torch.mean(self.temporal_att_HPEV(item), -2) for item in finger_feature_HPEV]
        hand_temporal_feature_HPEV = torch.mean(self.temporal_att_HPEV(hand_feature_HPEV), -2)

        finger_temporal_feature_HPEV.append(hand_temporal_feature_HPEV)
        temporal_feature_HPEV = finger_temporal_feature_HPEV
        temporal_feature_HPEV = torch.stack(temporal_feature_HPEV, -2)

        temporal_feature_HPEV = torch.mean(self.fusion_att_HPEV(temporal_feature_HPEV), -2)

        FRPV = self.BN_FRPV(FRPV)
        f_FRPV = self.fc_FPRV(FRPV)

        f_HPEV = torch.cat((temporal_feature_HPEV, f_FRPV), 1)

        logits_HPEV = self.cls_HPEV(f_HPEV)

        # HMM
        HMM = HMM.permute(0, 3, 1, 2)
        HMM = self.BN_HMM(HMM)
        HMM = HMM.permute(0, 2, 3, 1)

        f_HMM = self.input_map_HMM(HMM)

        hand_feature_HMM = torch.mean(self.finger_att_HMM(f_HMM), -2)

        finger_temporal_feature_HMM = [torch.mean(self.temporal_att_HMM(f_HMM[:, :, i, :]), -2) for i in range(f_HMM.shape[-2])]
        hand_temporal_feature_HMM = torch.mean(self.temporal_att_HMM(hand_feature_HMM), -2)

        finger_temporal_feature_HMM.append(hand_temporal_feature_HMM)
        temporal_feature_HMM = finger_temporal_feature_HMM
        temporal_feature_HMM = torch.stack(temporal_feature_HMM, -2)

        temporal_feature_HMM = torch.mean(self.fusion_att_HMM(temporal_feature_HMM), -2)

        logits_HMM = self.cls_HMM(temporal_feature_HMM)

        # average logits
        logits = (logits_HPEV + logits_HMM) / 2.0

        pred = self.softmax(logits)

        return pred


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).cuda()
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(-2), :]
        return x
        # return self.dropout(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, ft_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(ft_dim))
        self.b_2 = nn.Parameter(torch.zeros(ft_dim))
        self.eps = eps

    def forward(self, x):
        # [batch, time, ft_dim)
        mean = x.mean(-1, keepdim=True)

        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
    def __init__(self, h_num, h_dim, input_frames, input_dim, dp_rate, domain):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # assert d_model % h == 0
        # We assume d_v always equals d_k
        self.h_dim = h_dim  # head dimension
        self.h_num = h_num  # head num
        self.attn = None  # calculate_att weight
        # self.att_ft_dropout = nn.Dropout(p=dp_rate)
        self.domain = domain  # spatial of  tempoal
        self.input_frames = input_frames

        self.register_buffer('t_mask', self.get_domain_mask()[0])
        self.register_buffer('s_mask', self.get_domain_mask()[1])

        self.key_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.Dropout(dp_rate),
                            )

        self.query_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.Dropout(dp_rate),
                            )

        self.value_map = nn.Sequential(
                            nn.Linear(input_dim, self.h_dim * self.h_num),
                            nn.ReLU(),
                            nn.Dropout(dp_rate),)

    def get_domain_mask(self):
        # Sec 3.4
        time_len = self.input_frames
        joint_num = 22
        t_mask = torch.ones(time_len * joint_num, time_len * joint_num)
        filted_area = torch.zeros(joint_num, joint_num)

        for i in range(time_len):
            row_begin = i * joint_num
            column_begin = row_begin
            row_num = joint_num
            column_num = row_num

            t_mask[row_begin: row_begin + row_num, column_begin: column_begin + column_num] *= filted_area  # Sec 3.4

        I = torch.eye(time_len * joint_num)
        s_mask = Variable((1 - t_mask)).cuda()
        t_mask = Variable(t_mask + I).cuda()
        return t_mask, s_mask

    def attention(self, query, key, value):
        "Compute 'Scaled Dot Product Attention'"
        # [batch, time, ft_dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)

        # apply weight_mask to bolck information passage between ineer-joint

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, x):
        "Implements Figure 2"
        size0 = list(x.shape)
        size = list(x.shape[0:-1])  # [batch, t, dim]
        size.append(self.h_num)
        size.append(self.h_dim)
        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.query_map(x).view(size).transpose(-2, -3)
        key = self.key_map(x).view(size).transpose(-2, -3)
        value = self.value_map(x).view(size).transpose(-2, -3)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value)  # [batch, h_num, T, h_dim ]

        # 3) "Concat" using a view and apply a final linear.
        size0[-1] = self.h_num * self.h_dim
        x = x.transpose(-2, -3).contiguous() \
            .view(size0)  # [batch, T, h_dim * h_num ]

        return x


class ATT_Layer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, input_size, output_size, h_num, h_dim, dp_rate, time_len, domain):
        # input_size : the dim of input
        # output_size: the dim of output
        # h_num: att head num

        # h_dim: dim of each att head
        # time_len: input frame number
        # domain: do att on spatial domain or temporal domain

        super(ATT_Layer, self).__init__()

        self.pe = PositionalEncoding(d_model=input_size, dropout=0)
        # h_num, h_dim, input_dim, dp_rate,domain
        self.attn = MultiHeadedAttention(h_num, h_dim, time_len, input_size, dp_rate, domain)  # do att on input dim

        self.ft_map = nn.Sequential(
                        nn.Linear(h_num * h_dim, output_size),
                        nn.ReLU(),
                        LayerNorm(output_size),
                        nn.Dropout(dp_rate),
                        )

        self.norm = LayerNorm(output_size)
        self.dropout = nn.Dropout(dp_rate)

        self.init_parameters()

    def forward(self, x):
        # x0 = x
        x = self.pe(x)  # add PE
        x0 = x
        x = self.attn(x)  # pass attention model
        x = self.ft_map(x)
        return self.norm(x0 + self.dropout(x))
        # return self.norm(x0 + x)

    def init_parameters(self):
        model_list = [self.attn, self.ft_map]
        for model in model_list:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

