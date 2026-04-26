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

batch_size = 32     # số lượng dữ liệu sử dụng cùng lúc 
block_size = 256     # độ dài ngữ cảnh 
max_iters = 10000    # số lần lặp huấn luyện 
eval_interval = 500    # khoảng cách các lần kiểm tra để xem kết quả
learning_rate = 2e-4    # tốc độ học
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200    # số lần kiểm tra
n_embd = 384 # số chiều vector
n_head = 8
n_layer = 8
dropout = 0.2
kv_lora_rank = 64
warmup_iters = 500   # 500 bước đầu để model làm quen
lr_decay_iters = 10000 # Giảm dần về cuối
min_lr = 2e-5        # LR thấp nhất khi kết thúc (1/10 LR gốc)

# nn.Linear(độ dài vector, kích thước đầu ra, bias=True)

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

class MLALayer(nn.Module):
    tril: torch.Tensor
    def __init__(self, n_embd, n_head, head_size, kv_lora_rank):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.kv_lora_rank = kv_lora_rank
        
        self.kv_down_proj = nn.Linear(n_embd, kv_lora_rank, bias=False)
        self.kv_up_proj = nn.Linear(kv_lora_rank, n_head*(head_size+head_size), bias=False)
        # Phần của Key: Mỗi đầu chú ý có kích thước là head_size. Với n_head đầu, tổng số chiều cần thiết cho Key là n_head * head_size
        # tương tự vs Value nên 2 lần --> công thức
        
        self.q_proj = nn.Linear(n_embd, n_head * head_size, bias=False)
        
        self.out_proj = nn.Linear(n_head * head_size, n_embd, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        
    def forward(self, x):
        B,T,C = x.shape
        latent_kv = self.kv_down_proj(x)
        kv = self.kv_up_proj(latent_kv)
        k, v = torch.split(kv, self.n_head*self.head_size, dim=-1)  # chia và tạo ra key và value 2 phần = nhau
        
        k = k.view(B,T, self.n_head, self.head_size).transpose(1,2)
        v = v.view(B,T, self.n_head, self.head_size).transpose(1,2)
        q = self.q_proj(x).view(B,T, self.n_head, self.head_size).transpose(1,2)
        
        att = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # THÊM DÒNG NÀY
        att = F.softmax(att, dim=-1)
        att = self.dropout(att) 
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size) # chuyển lại về dạng vector chuẩn để tính toán sau 
        
        return self.out_proj(y)
 
class FeedForwardSwiGLU(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(2/3 * 4 * n_embd)
        
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False) # Cổng (Gate)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False) # Giá trị (Value)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False) # Lớp chiếu ngược lại
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #Công thức: SwiGLU(x) = (SiLU(xW1) * xW2)W3
        gate = self.w1(x)
        value = self.w2(x)
        x = F.silu(gate) * value
        return self.dropout(self.w3(x))
 
class Block(nn.Module):
    def __init__(self, n_embd, n_head, kv_lora_rank):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MLALayer(n_embd, n_head, head_size, kv_lora_rank)
        self.ffwd = FeedForwardSwiGLU(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 
         

### B - Batch size: số văn bản được đưa vào = batch_size
### T - Time / Sequence Length: số lượng các token (ký tự hoặc từ) trong một mẫu dữ liệu = block_size
### C - Channel / Embedding Dimension: độ dài của vector đại diện cho mỗi token = n_embd


class BigramLanguageModel(nn.Module):
    # hàm khởi tạo
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)   # tạo 1 bảng 2 chiều (ma trận) vs số hàng và cột để khi gọi sẽ tả cứu
        # block_size là vị trí tokens, và vect vtrí khớp vs vect từ 
        self.pos_embedding_table = nn.Embedding(block_size, n_embd) #block_size: Số lượng vị trí tối đa mà mô hình có thể xử lý
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head,kv_lora_rank=kv_lora_rank) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok_emd = self.token_embedding_table(idx)   # tra cứu
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) #Tạo ra một dãy số nguyên từ 0 đến T-1   "chỉ số vị trí" của các từ hiện tại trong chuỗi.
        #   Kết quả pos_emb có kích thước là (T, C)
        x = tok_emd + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # chiêu lên lớp tuyến tính 

        if targets is None:
            loss = None
        else:
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

model = BigramLanguageModel()
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

for iter in range(max_iters):
    # 1. Cập nhật Learning Rate
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # 2. Lấy dữ liệu (Nên lấy TRƯỚC khi tính toán)
    xb, yb = get_batch('train')
    
    # 3. Forward Pass với Mixed Precision
    with torch.cuda.amp.autocast():
        logits, loss = model(xb, yb)
        
    # 4. Backward Pass & Optimizer Step
    optimizer.zero_grad(set_to_none=True)
    
    scaler.scale(loss).backward()
    
    # Unscale để thực hiện Clip Gradient (quan trọng cho độ ổn định)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
    
    # 5. Đánh giá (Evaluation)
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad(): # Thêm cái này để tiết kiệm RAM khi eval
            losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        model.train()

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))