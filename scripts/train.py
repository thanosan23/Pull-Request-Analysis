import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

DEBUG = True

data = pd.read_csv("../data/pullreq_with_code.csv")

data = data[data["added_code"].astype(str) != "None"]

rejected_data = data.loc[data['merged_or_not'] == 0]
rejected_data = rejected_data.loc[rejected_data['contrib_gender'].notna()]

le = LabelEncoder()
le.fit(rejected_data['contrib_gender'])
rejected_data['contrib_gender'] = le.transform(rejected_data['contrib_gender'])

rejected_data = rejected_data.drop(['ownername', 'reponame', 'id', 'project_id', 'github_id', 'creator_id'], axis=1)

X = rejected_data['added_code']
Y = rejected_data['contrib_gender']

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")

X_tokens = tokenizer.batch_encode_plus(
    [code for code in X], 
    add_special_tokens=True, 
    padding='longest',
    truncation=True, 
    return_tensors="pt" 
)["input_ids"]

X_attention_masks = np.array(tokenizer.batch_encode_plus(
    [code for code in X], 
    add_special_tokens=True, 
    padding='longest',
    truncation=True, 
    return_tensors="pt" 
)["attention_mask"])



class Net(nn.Module):
    def __init__(self, bert_model, hidden_size, num_classes):
        super(Net, self).__init__()
        self.bert = bert_model
        self.fc1 = nn.Linear(bert_model.config.hidden_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, hidden_size) 
        self.fc4 = nn.Linear(hidden_size, num_classes)  
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x, attention_masks=None):
        if attention_masks is None:
            x = self.bert(x)[0][:, 0, :]
        else:
            x = self.bert(x, attention_mask=attention_masks)[0][:, 0, :]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out

model = Net(codebert_model, hidden_size=1000, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

X = np.array(X_tokens)
y = np.array(Y)

X = X[:5000]
y = y[:5000]

X_attention_masks = X_attention_masks[:5000]

smote = SMOTE()

if not DEBUG:
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X, y)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        X_train, X_test = X[train_ids], X[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]
        X_attention_masks_b = torch.tensor(X_attention_masks[train_ids])

        X_train, y_train = smote.fit_resample(X_train, y_train)

        for epoch in range(20):
            for i, (codes, labels) in enumerate(zip(X_train, y_train)):
                codes = torch.tensor(codes).unsqueeze(0)
                labels = torch.tensor([labels]).long()
                attention_mask = torch.tensor(X_attention_masks_b[i]).unsqueeze(0)

                outputs = model(codes, attention_mask)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(loss.item())

            correct = 0
            total = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for i, (codes, labels) in tqdm(enumerate(zip(X_test, y_test))):

                    codes = torch.tensor(codes).unsqueeze(0)
                    labels = torch.tensor([labels]).long()
                    attention_mask = torch.tensor(X_attention_masks[test_ids][i]).unsqueeze(0)

                    outputs = model(codes, attention_mask)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    y_true.append(labels.item())
                    y_pred.append(predicted.item())

            accuracy = 100 * correct / total
            print ('Epoch [{}/{}], Train Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(epoch+1, 20, loss.item(), accuracy))
            print(classification_report(y_true, y_pred, zero_division=0))
else:

    X = X[:150]
    y = y[:150]
    X_attention_masks = X_attention_masks[:150]

    X_train, X_test, y_train, y_test, X_attention_masks_b, X_attention_masks_test = train_test_split(
        X, y, X_attention_masks, test_size=0.2, random_state=42
    )

    X_train, y_train = smote.fit_resample(X_train, y_train)

    for epoch in range(20):
        for i, (codes, labels) in enumerate(zip(X_train, y_train)):
            codes = torch.tensor(codes).unsqueeze(0)
            labels = torch.tensor([labels]).long()
            attention_mask = torch.tensor(X_attention_masks_b[i]).unsqueeze(0)

            outputs = model(codes, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss.item())

        correct = 0
        total = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, (codes, labels) in tqdm(enumerate(zip(X_test, y_test))):

                codes = torch.tensor(codes).unsqueeze(0)
                labels = torch.tensor([labels]).long()
                attention_mask = torch.tensor(X_attention_masks_test[i]).unsqueeze(0)

                outputs = model(codes, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.append(labels.item())
                y_pred.append(predicted.item())

        accuracy = 100 * correct / total
        print ('Epoch [{}/{}], Train Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(epoch+1, 20, loss.item(), accuracy))
        print(classification_report(y_true, y_pred, zero_division=0))