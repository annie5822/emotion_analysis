import pandas as pd
from transformers import BertTokenizer, BertPreTrainedModel, BertModel, AdamW, BertConfig
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import KFold

# 定義基於 BERT 的模型，用於回歸任務（Valence、Arousal 和 Sociality）
class BertForValenceArousalSociality(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),                     # 隨機丟棄 30% 神經元
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),                  # 加入批歸一化
            nn.Linear(64, 3),
            nn.Sigmoid()
        )
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        # 輸出範圍擴展至 [0, 9]
        logits = self.regressor(pooled_output) * 9  

        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels)
            return loss, logits
        return logits

# 讀取數據
data_path = "C:/Users/user/Desktop/train_set.csv"
df = pd.read_csv(data_path)  # 從 CSV 文件加載數據

# 確保數據包含必要的列
assert 'text' in df.columns and 'Valence' in df.columns and 'Arousal' in df.columns and 'Sociality' in df.columns, "數據缺少必要的列"
df = df[['text', 'Valence', 'Arousal', 'Sociality']].dropna()

# 定義分詞器與預處理函數
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)
    tokens['labels'] = [examples['Valence'], examples['Arousal'], examples['Sociality']]
    return tokens

# 將數據轉換為 Hugging Face 的 Dataset 格式
dataset = Dataset.from_pandas(df)

def collate_fn(batch):
    input_ids = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b['attention_mask'] for b in batch], dtype=torch.long)
    labels = torch.tensor([b['labels'] for b in batch], dtype=torch.float32)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# 設定 k-fold
k_folds = 9
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

num_epochs = 10
fold_results = []

for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
    print(f"========== Fold {fold+1}/{k_folds} ==========")

    # 分割資料集為該 fold 的訓練集與驗證集
    train_dataset_fold = dataset.select(train_index)
    val_dataset_fold = dataset.select(val_index)

    # 對該 fold 的資料進行預處理
    train_dataset_fold = train_dataset_fold.map(preprocess_function)
    val_dataset_fold = val_dataset_fold.map(preprocess_function)

    train_loader = DataLoader(train_dataset_fold, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset_fold, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 重新初始化模型與優化器（確保每個 fold 從相同初始狀態開始）
    config = BertConfig.from_pretrained(model_name)
    model = BertForValenceArousalSociality.from_pretrained(model_name, config=config).to("cpu")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training Fold {fold+1}, Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_avg = train_loss / len(train_loader)
        
        # 驗證
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Fold {fold+1}, Epoch {epoch+1}"):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                loss, _ = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)

        print(f"Fold {fold+1}, Epoch {epoch+1}: Train Loss = {train_loss_avg}, Val Loss = {val_loss_avg}")

    # 將該 fold 的最後結果記錄下來（如最後一個 epoch 的驗證 Loss）
    fold_results.append(val_loss_avg)

# 印出各 fold 的結果與平均
for i, res in enumerate(fold_results):
    print(f"Fold {i+1} final Val Loss: {res}")
print("Average Val Loss across folds:", sum(fold_results) / len(fold_results))

# （可選擇性）將最後一個 fold 訓練完的模型保存
model.save_pretrained("./va_model")
tokenizer.save_pretrained("./va_model")
print("模型已保存！")
