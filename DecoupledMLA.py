## coi các giải thích trong colab 
import os

import numpy as np
import torch 
import torch.nn as nn 
from torch.nn import functional as F

from datasets import load_dataset
import tiktoken
from tqdm import tqdm

import math

from dataclasses import dataclass

batch_size = 32     # số lượng dữ liệu sử dụng cùng lúc 
block_size = 512     # độ dài ngữ cảnh 
max_iters = 10000    # số lần lặp huấn luyện 
eval_interval = 500    # khoảng cách các lần kiểm tra để xem kết quả
learning_rate = 2e-4    # tốc độ học
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200    # số lần kiểm tra
n_embd = 384 # số chiều vector
n_head = 8
n_layer = 8
dropout = 0.1
kv_lora_rank = 64
warmup_iters = 500   # 500 bước đầu để model làm quen
lr_decay_iters = 10000 # Giảm dần về cuối
min_lr = 2e-5        # LR thấp nhất khi kết thúc (1/10 LR gốc)

# nn.Linear(độ dài vector, kích thước đầu ra, bias=True)

@dataclass
class ModelConfig:
    block_size: int = 512
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 384
    dropout: float = 0.1
    kv_lora_rank: int = 64
    q_lora_rank: int = 64
    rope_dim: int = 32

config = ModelConfig()

# tạo seed --> đồng nhất 
torch.manual_seed(1337)

# mở file
dataset = load_dataset("openwebtext", split='train', streaming=True)
enc = tiktoken.get_encoding("gpt2")

vocab_size = enc.n_vocab

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

def process_and_save(num_samples = 50000):
    print(f"Đang trích xuất {num_samples} mẫu dữ liệu...")
    all_tokens = []
    
    count = 0
    for example in tqdm(dataset):
        tokens = enc.encode_ordinary(example['text'])
        tokens.append(enc.eot_token) # Thêm token kết thúc văn bản
        all_tokens.extend(tokens)
        count += 1
        if count >= num_samples:
            break
    all_tokens = np.array(all_tokens, dtype=np.uint16)
  
    n = len(all_tokens)
    train_data = all_tokens[:int(n*0.9)]
    val_data = all_tokens[int(n*0.9):]
    
    train_data.tofile('train.bin')
    val_data.tofile('val.bin')
    print(f"Hoàn thành! File train.bin và val.bin đã sẵn sàng.")
  
  
process_and_save(num_samples=20000)  


# lấy data, tạo bài kiểm tra và tài liệu 
def get_batch(split):
    filename = 'train.bin' if split == 'train' else 'val.bin'
    # memmap giúp đọc dữ liệu từ ổ đĩa mà không tốn RAM
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # số ngẫu nhiên sẽ từ 0 --> len(data) - block_size -1 --> vẫn sẽ có phần target 
    # hàm sẽ gồm 4 số ngẫu nhiên
    ix = torch.randint(len(data) - block_size, (batch_size,))   # lấy ngẫu nhiên 1 vị trí để bắt đầu --> tránh học vẹt
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# tính độ chính xác của mô hình
@torch.no_grad()    #Tạm dừng chế độ học tập của mô hình.
def estimate_loss():
    out = {}    # tạo 1 danh sách
    model.eval()    #bắt đầu chế độ kiểm tra để tránh cập nhập
    for split in ['train', 'val']:  # 1 lần là kiến thức để train lần còn lại là kiến thức mới(10% đã chia ở trên)
        losses = torch.zeros(eval_iters)    # tạo 1 bẳng để sau này tính tb+
        for k in range(eval_iters): # chạy nhiều lần để tránh nhiễu
            X, Y = get_batch(split)
            #   X (Inputs): Là đoạn văn bản đầu vào.
            #   Y (Targets): Là đoạn văn bản "đáp án" mà mô hình phải dự đoán.
            logits, loss = model(X, Y)
            losses[k] = loss.item() # lưu vào biến 
        out[split] = losses.mean()  # tính trung bình cộng 
    model.train()
    return out

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # dim ở đây chính là qk_rope_head_dim (ví dụ: 32)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float() # [end, dim/2]
    
    # Quan trọng: Lặp lại mỗi cột 2 lần để từ dim/2 thành dim (16 -> 32)
    # Kết quả: [cos1, cos1, cos2, cos2, ...]
    freqs_cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    freqs_sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    
    return freqs_cos, freqs_sin

def apply_rope(x, freqs_cos, freqs_sin):
    # Lấy shape: [B, T, n_head, rope_dim]
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    
    rotated_x = torch.cat((-x2, x1), dim=-1)
    return x * freqs_cos + rotated_x * freqs_sin

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # Chuyển đổi x sang số phức
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Đảm bảo freqs_cis có cùng kích thước để broadcast
    shape = [1] * len(x.shape[:-1])
    shape[1] = x.shape[1] # Thường là chiều T (sequence length)
    freqs_cis = freqs_cis.view(*shape)
    # Phép nhân số phức chính là phép xoay vector
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)

class MLALayer(nn.Module):
    tril: torch.Tensor
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.rope_dim = config.rope_dim
        
        self.kv_down_proj = nn.Linear(config.n_embd, config.kv_lora_rank, bias=False)
        self.kv_up_proj = nn.Linear(config.kv_lora_rank, self.n_head * self.head_size, bias=False)
        # Phần của Key: Mỗi đầu chú ý có kích thước là head_size. Với n_head đầu, tổng số chiều cần thiết cho Key là n_head * head_size
        # tương tự vs Value nên 2 lần --> công thức
        self.k_rope_proj = nn.Linear(config.kv_lora_rank, self.n_head * self.rope_dim, bias=False)
        
        self.q_rope_dim = 64
        self.q_down_proj = nn.Linear(config.n_embd, config.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(config.q_lora_rank, self.n_head * self.head_size, bias=False)
        self.q_rope_proj = nn.Linear(config.q_lora_rank, self.n_head * self.rope_dim, bias=False)
        
        self.out_proj = nn.Linear(self.n_head * self.head_size, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
    def forward(self, x, freqs_cos, freqs_sin):
        B,T,C = x.shape
        
        latent_kv = self.kv_down_proj(x)
        
        v = self.kv_up_proj(latent_kv).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k_content = self.kv_up_proj(latent_kv).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k_rope = self.k_rope_proj(latent_kv).view(B, T, self.n_head, self.rope_dim).transpose(1, 2)
        
        latent_q = self.q_down_proj(x)
        q_content = self.q_up_proj(latent_q).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q_rope = self.q_rope_proj(latent_q).view(B, T, self.n_head, self.rope_dim).transpose(1, 2)
        
        q_rope = apply_rope(q_rope.transpose(1,2), freqs_cos, freqs_sin).transpose(1,2)
        k_rope = apply_rope(k_rope.transpose(1,2), freqs_cos, freqs_sin).transpose(1,2)
        
        q = torch.cat([q_content, q_rope], dim=-1) # [B, n_head, T, head_size + rope_dim]
        k = torch.cat([k_content, k_rope], dim=-1)
        
        att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # THÊM DÒNG NÀY
        att = F.softmax(att, dim=-1)
        att = self.dropout(att) 
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size) # chuyển lại về dạng vector chuẩn để tính toán sau 
        
        return self.out_proj(y)
    
class FeedForwardSwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(2/3 * 4 * n_embd)
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False) # Cổng (Gate)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False) # Giá trị (Value)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False) # Lớp chiếu ngược lại
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        #Công thức: SwiGLU(x) = (SiLU(xW1) * xW2)W3
        gate = self.w1(x)
        value = self.w2(x)
        
        return self.dropout(self.w3(F.silu(gate) * value))
 
class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MLALayer(config)
        self.ffwd = FeedForwardSwiGLU(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
    def forward(self, x, freqs_cos, freqs_sin):
        x = x + self.sa(self.ln1(x), freqs_cos, freqs_sin)
        x = x + self.ffwd(self.ln2(x))
        return x
         

### B - Batch size: số văn bản được đưa vào = batch_size
### T - Time / Sequence Length: số lượng các token (ký tự hoặc từ) trong một mẫu dữ liệu = block_size
### C - Channel / Embedding Dimension: độ dài của vector đại diện cho mỗi token = n_embd


class GPTLanguageModel(nn.Module):
    # hàm khởi tạo
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)   # tạo 1 bảng 2 chiều (ma trận) vs số hàng và cột để khi gọi sẽ tả cứu
        
        # block_size là vị trí tokens, và vect vtrí khớp vs vect từ 
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.token_embedding_table.weight
        self.apply(self._init_weights)
        
        freqs_cos, freqs_sin = precompute_freqs_cis(config.rope_dim, config.block_size)
        self.register_buffer("freqs_cos", freqs_cos.unsqueeze(0).unsqueeze(2)) # [1, T, 1, rope_dim/2]
        self.register_buffer("freqs_sin", freqs_sin.unsqueeze(0).unsqueeze(2))
        
        cos, sin = precompute_freqs_cis(config.rope_dim, config.block_size)
        self.register_buffer("freqs_cos", cos.view(1, config.block_size, 1, -1)) 
        self.register_buffer("freqs_sin", sin.view(1, config.block_size, 1, -1))
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)     
    
    def forward(self, idx, targets=None):
        # Lấy thông tin kích thước và device trực tiếp từ tensor đầu vào
        B, T = idx.shape

        # 1. Embedding (Bỏ pos_emb vì đã có RoPE bên trong MLA)
        x = self.token_embedding_table(idx) # (B, T, n_embd)

        assert isinstance(self.freqs_cos, torch.Tensor)
        assert isinstance(self.freqs_sin, torch.Tensor)
        
        f_cos = self.freqs_cos[:, :T, :, :] 
        f_sin = self.freqs_sin[:, :T, :, :]
        
        for block in self.blocks:
            x = block(x, f_cos, f_sin)

        # 3. Truyền qua các Blocks
        # LƯU Ý: Nếu self.blocks là nn.Sequential, bạn KHÔNG THỂ gọi self.blocks(x, freqs_cis)
        # Bạn phải dùng vòng lặp for để truyền tham số bổ sung
        for block in self.blocks:
            x = block(x, f_cos, f_sin)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.5, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            
            logits = logits / temperature

            if top_k is not None:
                # torch.topk: iá trị (values) và chỉ số (indices) của $K$ phần tử lớn nhất
                # logits.size(-1) là tổng số từ trong từ điển. Dùng min phòng trường hợp top_k > tổng số từ 
                # nếu K=3, v có thể là [10.5, 8.2, 5.0].
                #   --> v[:, [-1]] chọn từ đầu --> 5.0 --> là ngưỡng cửa
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx

model = GPTLanguageModel(config=config)
m = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=learning_rate, 
    weight_decay=0.1,    # Giá trị 0.1 thường là "điểm ngọt" cho Transformer
    betas=(0.9, 0.95),   # Betas 0.95 thường ổn định hơn cho mô hình sâu
    eps=1e-8
)

def get_lr(it):
    # 1. Giai đoạn Warmup: tăng tuyến tính
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # 2. Nếu vượt quá max_iters, trả về LR tối thiểu
    if it > max_iters:
        return min_lr
    
    # 3. Giai đoạn Cosine Decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Giảm từ 1 về 0
    return min_lr + coeff * (learning_rate - min_lr)

scaler = torch.cuda.amp.GradScaler()

# Khởi tạo trước vòng lặp
model.train() 

accumulation_steps = 4
optimizer.zero_grad()

accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)

for iter in range(max_iters):
    # 1. Cập nhật Learning Rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # 2. Lấy dữ liệu
    xb, yb = get_batch('train')
    
    # 3. Forward Pass & Backward Pass (Mixed Precision)
    # Không dùng zero_grad ở đây vì ta cần tích lũy grad cho đến khi đủ accumulation_steps
    with torch.cuda.amp.autocast():
        logits, loss = model(xb, yb)
        # Chia loss cho accumulation_steps để trung bình hóa gradient
        dist_loss = loss / accumulation_steps
        
    # Scale loss và tính backward (tích lũy gradient vào .grad)
    scaler.scale(dist_loss).backward()
    
    # 4. Cập nhật trọng số sau khi tích lũy đủ bước
    if (iter + 1) % accumulation_steps == 0:
        # Unscale để thực hiện gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step optimizer và cập nhật scaler
        scaler.step(optimizer)
        scaler.update()
        
        # QUAN TRỌNG: Xóa gradient sau khi đã cập nhật xong
        optimizer.zero_grad(set_to_none=True)
    
    # 5. Đánh giá (Evaluation)
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss() 
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        model.train()

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))