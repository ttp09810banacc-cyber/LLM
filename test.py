import torch
import tiktoken
from DecoupledMLA import BigramLanguageModel # Giả sử bạn để class model trong file model.py

# --- 1. Cấu hình (Phải khớp với lúc train) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = 'path/gpt_webtext_step8000.pth' # Đường dẫn file của bạn

# --- 2. Khởi tạo Tokenizer và Model ---
enc = tiktoken.get_encoding("gpt2")

# Bạn cần khởi tạo lại class model với thông số y hệt lúc train
model = BigramLanguageModel(
    vocab_size = enc.n_vocab,
    n_layer = 8, 
    n_head = 8, 
    n_embd = 384,
    block_size = 512,
    dropout = 0.1,
    kv_lora_rank = 64,
    q_lora_rank=64,
    rope_dim=32
)

# --- 3. Load trọng số từ file .pth ---
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval() # CỰC KỲ QUAN TRỌNG: Tắt Dropout và Batch Normalization

print("Đã load model thành công! Bắt đầu chat (gõ 'exit' để thoát).")

# --- 4. Hàm generate văn bản có Top-k và Temperature ---
def generate_response(prompt, max_new_tokens=100, temperature=0.7, top_k=40):
    # Encode input
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # Chỉ lấy block_size cuối cùng nếu prompt quá dài
        idx_cond = idx[:, -256:] 
        
        # Forward pass
        with torch.no_grad():
            logits, _ = model(idx_cond)
            # Lấy logit ở bước cuối cùng và chia cho temperature
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Chuyển thành xác suất
            probs = torch.softmax(logits, dim=-1)
            
            # Chọn từ tiếp theo
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Ghép vào chuỗi hiện tại
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Nếu gặp token kết thúc (nếu bạn có quy định) thì dừng
            if idx_next.item() == enc.eot_token:
                break
                
    return enc.decode(idx[0].tolist())

# --- 5. Vòng lặp Chat ---
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit': break
    
    response = generate_response(user_input)
    print(f"AI: {response}\n")