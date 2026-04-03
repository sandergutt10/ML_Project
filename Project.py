import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import math
from collections import Counter

############################################
# 1. Стриминговый Dataset
############################################

class StreamingASTDataset(Dataset):
    def __init__(self, path, vocab, seq_len=32):
        self.path = path
        self.vocab = vocab
        self.seq_len = seq_len
        with open(path, "r", encoding="utf8") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        ast_nodes = json.loads(self.lines[idx])
        tokens = []

        for node in ast_nodes:
            tokens.append("TYPE_" + node["type"])
            if "value" in node and node["value"] is not None:
                tokens.append("VALUE_" + str(node["value"]))

        encoded = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]

        # pad
        if len(encoded) < self.seq_len + 1:
            encoded += [self.vocab["<PAD>"]] * (self.seq_len + 1 - len(encoded))

        x = encoded[:self.seq_len]
        y = encoded[1:self.seq_len+1]

        return torch.tensor(x), torch.tensor(y)


############################################
# 2. Построение словаря (потоковый)
############################################

def build_vocab_streaming(path, min_freq=5):
    counter = Counter()
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            ast_nodes = json.loads(line)
            for node in ast_nodes:
                counter["TYPE_" + node["type"]] += 1
                if "value" in node and node["value"] is not None:
                    counter["VALUE_" + str(node["value"])] += 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, freq in counter.items():
        if freq >= min_freq:
            vocab[tok] = len(vocab)

    ivocab = {i: tok for tok, i in vocab.items()}
    return vocab, ivocab


############################################
# 3. Positional Encoding
############################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


############################################
# 4. Модель
############################################

class CodeModel(nn.Module):
    def __init__(self, vocab_size, emb=64, heads=2, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb)
        self.pos = PositionalEncoding(emb)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=heads,
            dim_feedforward=128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.fc = nn.Linear(emb, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos(x)
        x = self.transformer(x)
        return self.fc(x)


############################################
# 5. Генерация
############################################

def generate(model, start_tokens, max_len, ivocab, device, temperature=1.0, top_k=30):
    model.eval()
    tokens = start_tokens.copy()

    for _ in range(max_len):
        x = torch.tensor(tokens[-32:], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)[0, -1] / temperature
        probs = torch.softmax(logits, dim=0)

        if top_k > 0:
            top_probs, top_idx = torch.topk(probs, top_k)
            probs = top_probs / top_probs.sum()
            next_token = top_idx[torch.multinomial(probs, 1)]
        else:
            next_token = torch.multinomial(probs, 1)

        tokens.append(next_token.item())

    return [ivocab.get(t, "<UNK>") for t in tokens]


############################################
# 6. Оценка
############################################

def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred.view(-1, pred.size(-1)), y.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


############################################
# 7. Обучение
############################################

def train(dataset_path, epochs=10, batch_size=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Building vocab...")
    vocab, ivocab = build_vocab_streaming(dataset_path)

    dataset = StreamingASTDataset(dataset_path, vocab)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CodeModel(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("Training...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred.view(-1, pred.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_loss, ppl = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}: train_loss={total_loss:.2f}, val_loss={val_loss:.4f}, ppl={ppl:.2f}")

    torch.save(model.state_dict(), "model.pt")
    return model, vocab, ivocab


############################################
# 8. Тест модели
############################################

def test_model(model, vocab, ivocab, device):
    print("\n=== MODEL TEST ===")
    start = ["TYPE_Module"]
    start_ids = [vocab.get(t, vocab["<UNK>"]) for t in start]
    result = generate(model, start_ids, max_len=50, ivocab=ivocab, device=device, temperature=0.8, top_k=30)
    print("\nGenerated sequence:\n")
    print(" ".join(result))


############################################
# 9. Запуск
############################################

if __name__ == "__main__":
    dataset_path = "/content/drive/MyDrive/python100k_train.json"
    model, vocab, ivocab = train(dataset_path, epochs=5, batch_size=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model(model, vocab, ivocab, device)