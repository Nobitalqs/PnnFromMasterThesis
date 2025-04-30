import torch
import os
import yaml
import uproot
import awkward as ak
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

import configs.classSets as classSets

print(f"current pytorch version is {torch.__version__}")

# First load the training and test data
with open("configs/neural_net.yaml", "r") as file:
    config = yaml.load(file, yaml.FullLoader)
class_dict=classSets.classes[config["classes"]]
def dataframe_split(split):
    for cl in class_dict:
        tmp_file_dict = dict()
        for file in class_dict[cl]:
            file_path = os.path.join(
                "/ceph/qli/nmssm_ml/05_03_2025_leftupper/preselection/2017/mt/" + split,
                file + ".root",
            )
            tmp_file_dict[file_path] = "ntuple"

        events = uproot.concatenate(tmp_file_dict)
        # transform the loaded awkward array to a pandas DataFrame
        df = ak.to_dataframe(events)
        df = self._randomize_masses(df, cl)
        df = self._add_labels(df, cl)
        log.info(f"number of events for {cl}: {df.shape[0]}")
        df = df.reset_index(drop=True)
    return df

training_data = dataframe_split("even")
test_data = dataframe_split("odd")

df_for_train,val_for_df = train_test_split(df_train,test_size=0.2,random_state=0)


# Transfer dataframe as Tensor
class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.X = dataframe.drop(columns=["label"]).values.astype("float32")
        self.y = dataframe["label"].values.astype("int64")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# Create data loader
training_dataloader = Dataloader(MyDataset(df_for_train),batch_size=5000,shuffle=True)
val_dataloader = Dotaloader(MyDataset(val_for_df),batch_size=5000,shuffle=True)
test_dataloader = Dataloader(MyDataset(test_data),batch_size=5000,shuffle=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__
        self.linear_relu_stack = nn.Sequential(
            nn.linear(),
            nn.ReLU(),
            nn.Linear(),
            nn.ReLU(),
            nn.Linear()
        )

    def forward(self,x):
        return self.linear_relu_stack(x)

model = SimpleNN().to(device)

lss_fn = config["L2"]
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.val()
    test_loss,correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "SimpleNN.pth")
print("Saved PyTorch Model State to SimpleNN.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("SimpleNN.pth", weights_only=True))


test_X,test_y = test_dataloader
with torch.no_grad():
    test_X = test_X.to(device)
    pred = model(test_X)
    print(pred)

