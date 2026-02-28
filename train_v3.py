import pandas as pd
import html
import time
import torch as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class BasicDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index):
        self.text = self.df.iloc[index]["fused_txt"]
        self.label = self.df.iloc[index]["label"]
        encoding = self.tokenizer(self.text,
                                padding="max_length",
                                truncation=True,
                                max_length=512,
                                return_tensors="pt")
        
        return {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0),
        'labels': nn.tensor(self.label, dtype=nn.long)
        }



tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Cleaning function
def clean_text(text):
    txt = str(text)
    txt = html.unescape(txt)
    txt = " ".join(text.split())
    return txt

# Open and clean dataset
df = pd.read_csv("./datasets/WELFake_Dataset.csv")
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["fused_txt"] = "[TITLE] " + df["title"] + " [TEXT] " + df["text"]
df["fused_txt"] = df["fused_txt"].apply(clean_text)

# Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize data
train_dataset = BasicDataset(df_train, tokenizer)
test_dataset = BasicDataset(df_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = nn.device("cuda" if nn.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on {device}")

optimizer = nn.optim.AdamW(model.parameters(), lr=2e-5)

epochs = 1
model.train()

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1}/{epochs}")

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

model.eval()
correct = 0
total = 0

print("\n--- Evaluating ---")
with nn.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = nn.argmax(outputs.logits, dim=-1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy: {100 * correct / total:.2f}%")

model.save_pretrained("./output/my_fake_news_model")
tokenizer.save_pretrained("./output/my_fake_news_model")
print("Model saved!")