#
# 文件名: vit_model.py
# ------------------------
# 包含 ViT 模型架构的定义和预训练权重加载函数
#

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
from typing import Tuple

# --- 辅助函数 ---
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# --- 模块定义 ---

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., qkv_bias = True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias) # 使用 qkv_bias
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., qkv_bias = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, qkv_bias = qkv_bias),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., qkv_bias = True):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # 替换为 timm 兼容的 Conv2d Patch Embedding
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, qkv_bias=qkv_bias)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 自动处理不同图像大小（尽管我们这里固定为224x224）
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

# --- 权重加载辅助函数 ---
def _load_pretrained_weights(my_model, timm_model_name):
    """
    内部辅助函数，用于加载 timm 预训练权重。
    """
    print(f"开始加载 {timm_model_name} 的预训练权重...")
    timm_model = timm.create_model(timm_model_name, pretrained=True)
    timm_state_dict = timm_model.state_dict()
    my_state_dict = my_model.state_dict()
    
    new_state_dict = {}
    
    # 键映射 (My Model Key -> Timm Key)
    key_map = {
        'cls_token': 'cls_token',
        'pos_embedding': 'pos_embed',
        'transformer.norm.weight': 'norm.weight',
        'transformer.norm.bias': 'norm.bias',
        'to_patch_embedding.0.weight': 'patch_embed.proj.weight', # Conv2d weight
        'to_patch_embedding.0.bias': 'patch_embed.proj.bias',     # Conv2d bias
    }

    for my_key in my_state_dict.keys():
        # 1. 跳过分类头 (我们要在 CIFAR-10 上重新训练)
        if my_key.startswith('mlp_head'):
            continue
            
        # 2. 映射简单键
        if my_key in key_map:
            timm_key = key_map[my_key]
            if timm_key in timm_state_dict:
                if timm_state_dict[timm_key].shape == my_state_dict[my_key].shape:
                    new_state_dict[my_key] = timm_state_dict[timm_key]
                else:
                    print(f"形状不匹配! My: {my_key} {my_state_dict[my_key].shape}, Timm: {timm_key} {timm_state_dict[timm_key].shape}")
            else:
                print(f"警告: 未在 timm 中找到 {timm_key}")
            continue

        # 3. 映射 Transformer 内部的层
        if my_key.startswith('transformer.layers.'):
            parts = my_key.split('.')
            layer_num, block_type, sub_key = parts[2], parts[3], ".".join(parts[4:])
            
            timm_key = None
            if block_type == '0': # Attention
                if sub_key.startswith('norm.'): timm_key = f"blocks.{layer_num}.norm1.{sub_key.replace('norm.', '')}"
                elif sub_key.startswith('to_qkv.'): timm_key = f"blocks.{layer_num}.attn.{sub_key.replace('to_qkv.', 'qkv.')}"
                elif sub_key.startswith('to_out.0.'): timm_key = f"blocks.{layer_num}.attn.proj.{sub_key.replace('to_out.0.', '')}"
            elif block_type == '1': # FeedForward
                if sub_key == 'net.0.weight': timm_key = f"blocks.{layer_num}.norm2.weight"
                elif sub_key == 'net.0.bias': timm_key = f"blocks.{layer_num}.norm2.bias"
                elif sub_key == 'net.1.weight': timm_key = f"blocks.{layer_num}.mlp.fc1.weight"
                elif sub_key == 'net.1.bias': timm_key = f"blocks.{layer_num}.mlp.fc1.bias"
                elif sub_key == 'net.4.weight': timm_key = f"blocks.{layer_num}.mlp.fc2.weight"
                elif sub_key == 'net.4.bias': timm_key = f"blocks.{layer_num}.mlp.fc2.bias"
            
            if timm_key:
                if timm_key in timm_state_dict:
                    if timm_state_dict[timm_key].shape == my_state_dict[my_key].shape:
                        new_state_dict[my_key] = timm_state_dict[timm_key]
                else:
                     print(f"未映射 (timm 中未找到): {timm_key}")
            elif not (sub_key.startswith('net.2') or sub_key.startswith('net.3')): # 忽略GELU/Dropout
                 print(f"未映射: {my_key}")

    # 加载我们构建的新 state_dict
    msg = my_model.load_state_dict(new_state_dict, strict=False)
    print("权重加载完成。")
    print(f"缺失的键 (应主要为 mlp_head): {msg.missing_keys}")
    print(f"意外的键 (应为空): {msg.unexpected_keys}")
    
    return my_model


# --- 工厂函数 (供外部调用) ---

def create_vit_model(
    image_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 10,
    dim: int = 768,
    depth: int = 12,
    heads: int = 12,
    mlp_dim: int = 3072,
    dim_head: int = 64,
    pretrained: bool = True,
    timm_model_name: str = 'vit_base_patch16_224'
) -> nn.Module:
    """
    创建、配置并（可选地）加载 ViT 模型的预训练权重。
    """
    
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head
    )

    if pretrained:
        model = _load_pretrained_weights(model, timm_model_name)
    
    return model