import torch
import os
import yaml
import uproot
import awkward as ak
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import configs.classSets as classSets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import mode

#libraries for tca analysis
from tayloranalysis.model_extension import extend_model
import itertools

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-x",
    "--massx",
    type=int
)

parser.add_argument(
    "-y",
    "--massy",
    type=int
)

parser.add_argument(
    "-e",
    "--epochs",
    type=int
)

parser.add_argument(
    "-s",
    "--essemble",
    type=int
)

args = parser.parse_args()

output_dir = "simpletrain_results"
os.makedirs(output_dir, exist_ok=True)


# List of input features that would be used for training
SAMPLES = OrderedDict([
    (
        "ybb_htt",
        [
            "XToYHTo2B2Tau.root",
        ],
    ),
    (
        "ytt_hbb",
        [
            "XToYHTo2Tau2B.root",
        ],
    ),
    (
        "tt_st",
        [

            "ttbar.root",
            "ST.root",
        ],
    ),
    (
        "dy_h_tautau",
        [
            "HToTauTau.root",
            "DYjets.root",
        ],
    ),
    (
        "misc",
        [
            "EWK.root",
            "WJets.root",
            "diboson.root",
        ],
    ),
])

TRAIN_SAMPLES = OrderedDict()
TEST_SAMPLES = OrderedDict()
for key, item in SAMPLES.items():
        TRAIN_SAMPLES[key] = [os.path.join("/ceph/qli/nmssm_ml/05_03_2025_leftupper/preselection/2017/mt/even", i) for i in item]
        TEST_SAMPLES[key] = [os.path.join("/ceph/qli/nmssm_ml/05_03_2025_leftupper/preselection/2017/mt/odd", i) for i in item]

INPUT_FEATURES = [
    "njets",
    "nbtag",
    #"nfatjets",
    "pt_1",
    "pt_2",
    "eta_1",
    #"phi_1",
    "eta_2",
    #"phi_2",
    "deltaR_ditaupair",
    "m_vis",
    "m_fastmtt",
    "pt_fastmtt",
    "eta_fastmtt",
    "phi_fastmtt",
    "bpair_pt_1",
    "bpair_eta_1",
    "bpair_phi_1",
    "bpair_btag_value_1",
    "bpair_pt_2",
    "bpair_eta_2",
    "bpair_phi_2",
    "bpair_btag_value_2",
    "bpair_m_inv",
    "bpair_deltaR",
    "bpair_pt_dijet",
    "fj_Xbb_pt",
    "fj_Xbb_eta",
    "fj_Xbb_phi",
    "fj_Xbb_msoftdrop",
    "fj_Xbb_nsubjettiness_2over1",
    #"fj_Xbb_nsubjettiness_3over2", # has NaN values, due to wrong eta cut 
    "met",
    "metphi",
    #"mass_tautaubb",
    #"pt_tautaubb",
    "kinfit_mX",
    "kinfit_mY",
    "kinfit_chi2",
    "mt_1",
    #"deltaPhi_met_tau1",
    #"deltaPhi_met_tau2",
    #"jpt_1",
    #"jpt_2",
    #"mjj",
]

CLASSES = list(SAMPLES.keys())

def load_events(samples, classes, mass_x, mass_y):
    data_frames = OrderedDict()
    for key, value in samples.items():
        events = uproot.concatenate({f: "ntuple" for f in value}, library="pd")
        if key in ["ybb_htt", "ytt_hbb"]:
            events = events.loc[(events["massX"] == mass_x) & (events["massY"] == mass_y)]
        #events = events.loc[events["bpair_deltaR"] > 0.4]
        events["label"] = classes.index(key)
        data_frames[key] = events
    return data_frames

mass_x = args.massx
mass_y = args.massy

training_dataframes = load_events(TRAIN_SAMPLES, CLASSES, mass_x, mass_y)
test_dataframes = load_events(TEST_SAMPLES, CLASSES, mass_x, mass_y)

def weighted_cross_entropy(pred, target, weight):
    loss = F.cross_entropy(pred, target, reduction='none')
    return torch.sum(loss * weight)/torch.sum(weight)

def balance_samples(dfs):
    sum_weights_all = sum([sum(df["weight"].values) for df in dfs.values()])
    for output, df in dfs.items():
        sum_weights_class = sum(df['weight'].values)
        if output in ["ybb_htt","ytt_hbb"]:
            df["weight"] = 2*df["weight"] * (sum_weights_all / (len(dfs) * sum_weights_class))
        else:
            df["weight"] = df["weight"] * (sum_weights_all / (len(dfs) * sum_weights_class))


def create_dataset(data_frames, input_features, calculate_weights=True, mode="train"):
    x, y, w, index = [], [], [], []
    if calculate_weights:
        balance_samples(data_frames)
    for sample, data_frame in data_frames.items():
        inputs = torch.from_numpy(data_frame[input_features].values).float()
        inputs = (inputs - torch.mean(inputs, axis=0)) / torch.std(inputs, axis=0)
        targets = torch.from_numpy(data_frame["label"].values).long()
        weights = torch.from_numpy(data_frame["weight"].values).float()
        df_index = torch.from_numpy(data_frame.index.values).long()
        x.append(inputs)
        y.append(targets)
        w.append(weights)
        index.append(df_index)

    x = torch.cat(x, axis=0)
    y = torch.cat(y, axis=0)
    w = torch.cat(w, axis=0)
    index = torch.cat(index, axis=0)

    if mode == "train":
        
        x_train, x_val, y_train, y_val, w_train, w_val, i_train, i_val = train_test_split(x, y, w, index, train_size=0.8, shuffle=True)
        
        return (
        TensorDataset(x_train, y_train, w_train, i_train),
        TensorDataset(x_val, y_val, w_val, i_val)
        )

    elif mode == "test":
        
        return (
            TensorDataset(x, y, w, index)
        )

dataset_train, dataset_val = create_dataset(training_dataframes, INPUT_FEATURES, calculate_weights=True, mode="train")
dataset_test = create_dataset(test_dataframes, INPUT_FEATURES, calculate_weights=True, mode="test")


# Create data loader
training_dataloader = DataLoader(dataset_train,batch_size=1024,shuffle=True, num_workers=4)
val_dataloader = DataLoader(dataset_val,batch_size=1024,shuffle=False, num_workers=4)
test_dataloader = DataLoader(dataset_test,batch_size=1024,shuffle=False, num_workers=4)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(len(INPUT_FEATURES),200),
            nn.Tanh(),
            nn.BatchNorm1d(200),
            nn.Linear(200,200),
            nn.Tanh(),
            nn.BatchNorm1d(200),
            nn.Linear(200,200),
            nn.Tanh(),
            nn.BatchNorm1d(200),
            nn.Linear(200,5)
        )

    def forward(self,x):
        return self.linear_relu_stack(x)

model = SimpleNN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

def train(dataloader,model,optimizer):
    size = len(dataloader.dataset)
    model = model.train()
    for batch,(X,y,w,index) in enumerate(dataloader):
        X,y,w = X.to(device),y.to(device),w.to(device)
        pred = model(X)
        loss = weighted_cross_entropy(pred, y, w)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader,model,show_confusion=False,do_essemble=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    all_preds = []
    all_labels = []
    if not do_essemble:
        with torch.no_grad():
                for X, y, w, index in dataloader:
                    X,y,w = X.to(device),y.to(device),w.to(device)
                    pred = model(X)
                    test_loss += weighted_cross_entropy(pred, y, w).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                    all_preds.extend(torch.argmax(torch.softmax(pred, dim=1), dim=1).cpu().numpy())
                    all_labels.extend(y.cpu().numpy())

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    else:
        with torch.no_grad():
            essemble_list = []
            for i in range(do_essemble):
                preds_one_round = []
                for X, y, w, index in dataloader:
                    X,y,w = X.to(device),y.to(device),w.to(device)
                    pred = model(X)
                    preds_one_round.extend(torch.argmax(torch.softmax(pred, dim=1), dim=1).cpu().numpy())
                    if i == 0:
                        all_labels.extend(y.cpu().numpy())
            
                    #preds_one_round = np.concatenate(preds_one_round)
                essemble_list.append(preds_one_round)

            essemble_list = np.array(essemble_list).T
            voted_preds, _ = mode(essemble_list, axis=1)
            voted_preds = voted_preds.flatten()
            all_preds.extend(voted_preds)

    if show_confusion:
        cm = confusion_matrix(all_labels, all_preds, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.savefig(f"{output_dir}/confusion_matrix_x{mass_x}_y{mass_y}.png", dpi=300, bbox_inches='tight')

epochs = args.epochs
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(training_dataloader, model, optimizer)
    if t == (epochs -1):
        test(test_dataloader, model, show_confusion=True, do_essemble=args.essemble)
    else:
        test(test_dataloader, model)


# calculate TCA
model_tca = extend_model(model)
model_tca = model_tca.to(device)
model_tca = model_tca.eval()

def get_feature_combis(feature_list: list, combi_list: list):
    feature_combinations = []
    for combination in combi_list:
        feature_combi = tuple(feature_list[val] for val in combination)
        feature_combinations.append(feature_combi)
    return feature_combinations

# get the label combinations of input features
combinations = []

for i in range(len(INPUT_FEATURES)):
    combinations.append((i,))

combinations += [
    i for i in itertools.combinations(list(range(len(INPUT_FEATURES))),2)
]

# get input features names via combinations labels 
labels = get_feature_combis(INPUT_FEATURES, combinations)
labels = [",".join(label) for label in labels]


#define a dataframe for tca, columns name corresponds to each tca name
def DataframeConstruct(combs,input,node,labels):
    tc_dict = model_tca.get_tc(
        "x",
        forward_kwargs={"x": input.float().to(device)},
        selected_output_node=node,
        eval_max_output_node_only=False,
        tc_idx_list=combs,
        reduce_func=lambda x : x)  
    data = pd.DataFrame({f"{labels[i]}":tc_dict[combs[i]].cpu().detach().numpy() for i in range(len(combs))})
    # sorting dataframe with tca abs mean value
    sorted_columns = np.abs(data).mean(axis=0).sort_values(ascending=False).index
    sorted_df=data[sorted_columns].copy()
    return sorted_df

# construct tca dataframes
tca_dfs = OrderedDict()
node_list = [0,1,2]
for i in node_list:
    tca_dfs[CLASSES[i]] = DataframeConstruct(combinations, dataset_test.tensors[0], i, labels)


for class_name, df in tca_dfs.items():
    #df.to_csv(f"{output_dir}/{class_name}_tca.csv", index=False)
    
    mean_series = df.mean(axis=0)

    mean_series.to_frame(name="mean").to_csv(f"{output_dir}/{class_name}_tca_mean_x{mass_x}_y{mass_y}.csv")

print("Done!")


# torch.save(model.state_dict(), "SimpleNN.pth")
# print("Saved PyTorch Model State to SimpleNN.pth")

# model = SimpleNN().to(device)
# model.load_state_dict(torch.load("SimpleNN.pth", weights_only=True))


