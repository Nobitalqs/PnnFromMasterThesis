import torch
import os
import yaml
import uproot
import awkward as ak
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

import configs.classSets as classSets

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F


print(f"current pytorch version is {torch.__version__}")

def weighted_cross_entropy(pred, target, weight):
    loss = F.cross_entropy(pred, target, reduction='none')
    return torch.sum(loss * weight)/torch.sum(weight)

class DatasetProcessor:
    def __init__(self, config):
        self.config = config
        self.label_dict = dict()

    def _add_labels(self, df: pd.DataFrame, cl: str) -> pd.DataFrame:
        if cl not in self.label_dict:
            self.label_dict[cl] = len(self.label_dict)
        else:
            pass
        df["label"] = self.label_dict[cl]
        return df

    def dataframe_split(self, class_dict, split: str) -> pd.DataFrame:
        dfs = []
        for cl in class_dict:
            tmp_file_dict = {
                os.path.join(
                    "/ceph/qli/nmssm_ml/05_03_2025_leftupper/preselection/2017/mt/",
                    split,
                    file + ".root",
                ): "ntuple"
                for file in class_dict[cl]
            }

            events = uproot.concatenate(tmp_file_dict)
            df = ak.to_dataframe(events)
            df = self._add_labels(df, cl)
            print(f"Number of events for {cl}: {df.shape[0]}")
            df = df.reset_index(drop=True)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)
    


def balance_samples(df_train,df_val,classes,label_dict):
    sum_weights_all = sum(df_train["weight"].values) + sum(df_val["weight"].values)
    for cl in classes:
        mask_train = df_train["label"].isin([label_dict[cl]])
        mask_val = df_val["label"].isin([label_dict[cl]])
        sum_weights_class = sum(df_train.loc[mask_train, 'weight'].values) + sum(df_val.loc[mask_val, 'weight'].values)
        df_train.loc[mask_train, "weight"] = df_train.loc[mask_train, "weight"] * (sum_weights_all / (len(classes) * sum_weights_class))
        df_val.loc[mask_val, "weight"] = df_val.loc[mask_val, "weight"] * (sum_weights_all / (len(classes) * sum_weights_class))
        
        sum_weights_class_new = sum(df_train.loc[mask_train, 'weight'].values) + sum(df_val.loc[mask_val, 'weight'].values)


# Load configuration and classes
with open("configs/neural_net.yaml", "r") as file:
    config = yaml.load(file, yaml.FullLoader)
class_dict = classSets.classes[config["classes"]]

# Create processor instance
processor = DatasetProcessor(config=config)

# Training and test data
df_train = processor.dataframe_split(class_dict, "even")
# Train-validation split
df_for_train, val_for_df = train_test_split(df_train, test_size=0.2, random_state=0)
balance_samples(df_for_train,val_for_df,class_dict,processor.label_dict)

df_test = processor.dataframe_split(class_dict, "odd")


mask_train= ~((df_for_train["label"].isin([0, 1]))) | ((df_for_train["massX"] == 280) & (df_for_train["massY"] == 125))
mask_val= ~((val_for_df["label"].isin([0, 1]))) | ((val_for_df["massX"] == 280) & (val_for_df["massY"] == 125))
mask_test= ~((df_test["label"].isin([0, 1]))) | ((df_test["massX"] == 280) & (df_test["massY"] == 125))
df_for_train = df_for_train[mask_train]
val_for_df = val_for_df[mask_val]
df_test = df_test[mask_test]

for i in df_for_train.columns:
    if df_for_train[i].isna().any():
        df_for_train.drop(columns=i, inplace=True)
        val_for_df.drop(columns=i, inplace=True)
        df_test.drop(columns=i, inplace=True)

# Transfer dataframe as Tensor
class MyDataset():
    def __init__(self, dataframe):
        self.X = dataframe.drop(columns=["massX", "massY", "label", "weight"]).values.astype("float32")
        self.y = dataframe["label"].values.astype("int64")
        self.w = dataframe["weight"].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx]), torch.tensor(self.w[idx])


# Create data loader
training_dataloader = DataLoader(MyDataset(df_for_train),batch_size=5000,shuffle=True)
val_dataloader = DataLoader(MyDataset(val_for_df),batch_size=5000,shuffle=True)
test_dataloader = DataLoader(MyDataset(df_test),batch_size=5000,shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(37,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,300),
            nn.ReLU(),
            nn.Linear(300,5)
        )

    def forward(self,x):
        return self.linear_relu_stack(x)

model = SimpleNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

def train(dataloader,model,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X,y,w) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        pred = model(X)
        loss = weighted_cross_entropy(pred, y, w)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader,model,show_confusion=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y, w in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += weighted_cross_entropy(pred, y, w).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    if show_confusion:
        cm = confusion_matrix(all_labels, all_preds, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.show()

epochs = 1500
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_dataloader, model, optimizer)
    if t == (epochs -1):
        test(test_dataloader, model, show_confusion=True)
    else:
        test(test_dataloader, model)
print("Done!")


torch.save(model.state_dict(), "SimpleNN.pth")
print("Saved PyTorch Model State to SimpleNN.pth")

model = SimpleNN().to(device)
model.load_state_dict(torch.load("SimpleNN.pth", weights_only=True))


