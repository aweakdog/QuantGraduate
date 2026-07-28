"""
HSN v0.2 — 重写: 全部数据 GPU 常驻, 模型做大, GPU 跑满

关键改动:
  1. 直接把 tensor 搬到 GPU, 不经过 DataLoader
  2. 模型 16× → 256K 参数, 让 GPU 有事干
  3. 训练数据一次性预处理完毕, 每 epoch 只做 CUDA kernel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HSN(nn.Module):
    def __init__(self, dim_config: dict, hidden: int = 512):
        super().__init__()
        
        # ─── 每维度 MLP encoder ───
        self.encoders = nn.ModuleDict()
        for name, dim in dim_config.items():
            self.encoders[name] = nn.Sequential(
                nn.Linear(dim, hidden // 2),
                nn.BatchNorm1d(hidden // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden // 2, hidden // 2),
                nn.GELU(),
            )
        
        n_encoders = len(dim_config)
        fusion_in = (hidden // 2) * n_encoders
        
        # ─── Cross-dimension attention ───
        self.attn_q = nn.Linear(hidden // 2, hidden // 2)
        self.attn_k = nn.Linear(hidden // 2, hidden // 2)
        self.attn_v = nn.Linear(hidden // 2, hidden // 2)
        
        # ─── 深层融合网络 ───
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_dict):
        # 各维度独立编码
        embeddings = []
        for name, enc in self.encoders.items():
            e = enc(x_dict[name])  # (B, hidden/2)
            embeddings.append(e)
        
        # Stack: (B, n_encoders, hidden/2)
        stacked = torch.stack(embeddings, dim=1)
        
        # Cross-dimension attention
        q = self.attn_q(stacked)
        k = self.attn_k(stacked)
        v = self.attn_v(stacked)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(stacked.size(-1)), dim=-1)
        attended = torch.matmul(attn, v)  # (B, n_encoders, hidden/2)
        
        # Flatten and fuse
        fused = self.fusion(attended.reshape(attended.size(0), -1))
        return fused.squeeze(-1)
