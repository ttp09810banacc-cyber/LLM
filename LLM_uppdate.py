## coi các giải thích trong colab 

import torch 
import torch.nn as nn 
from torch.nn import functional as F

batch_size = 64     # số lượng dữ liệu sử dụng cùng lúc 
block_size = 256     # độ dài ngữ cảnh 
max_iters = 5000    # số lần lặp huấn luyện 
eval_interval = 500    # khoảng cách các lần kiểm tra để xem kết quả
learning_rate = 3e-4    # tốc độ học
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200    # số lần kiểm tra
n_embd = 384 # số chiều vector
n_head = 6
n_layer = 6
dropout = 0.2

# tạo seed --> đồng nhất 
torch.manual_seed(1337)

# mở file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  
# lấy kí tự    
chars = sorted(list(set(text)))
vocab_size = len(chars)

# ánh xạ
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# phân chia dữ liệu --> train và test
# tạo ra dãy số 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# lấy data, tạo bài kiểm tra và tài liệu 
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # số ngẫu nhiên sẽ từ 0 --> len(data) - block_size -1 --> vẫn sẽ có phần target 
    # hàm sẽ gồm 4 số ngẫu nhiên
    ix = torch.randint(len(data) - block_size, (batch_size,))   # lấy ngẫu nhiên 1 vị trí để bắt đầu --> tránh học vẹt
    x = torch.stack([data[i:i+block_size] for i in ix]) # lấy data 
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # kiểm tra
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

class Head(nn.Module):
    tril: torch.Tensor  # Thêm dòng này để báo cho IDE biết tril là Tensor
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)     # về cơ bản 3 dòng này tạo ra các trọng số khác nhau nên có các chức năng khác nhau
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) # tam giác có dưới là 1 --> nhìn thấy đc
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x) 
        out = wei @ v 
        return out
     
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)     
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Phình to ra gấp 4 lần
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # Co lại về n_embd
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
 
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
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
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
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

    def generate(self, idx, max_new_tokens, temperature=1.2, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 
            
            logits = logits / temperature
            
            if top_k is not None:
                # torch.topk: giá trị (values) và chỉ số (indices) của $K$ phần tử lớn nhất
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500, top_k=20)[0].tolist()))

torch.save(model.state_dict(), 'model_weights2.pth')
print("Đã lưu trọng số thành công!")