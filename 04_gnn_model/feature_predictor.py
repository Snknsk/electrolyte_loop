#!/usr/bin/env python3
"""
gnn_train.py  ──────────────────────────────────────────────────────
Crystal-graph neural network for Materials-Project rows saved by
`data_collector.py`.  Predicts formation-energy-per-atom.

• reads ./raw/materials_project.csv
• builds graphs from pymatgen Structure objects
• simple 3-layer CGCNN-style model
• k-fold cross-validation + MAE metric
--------------------------------------------------------------------
"""

import os, json, math, random, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

RAW = Path(__file__).resolve().parent / "raw"
CSV = RAW / "materials_project.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────── 1.  featurisation helpers ──────────────────
import pymatgen as mg
from pymatgen.core.periodic_table import Element

def atom_features(atom: mg.core.Site) -> list[float]:
    elem = Element(atom.specie.symbol)
    return [
        elem.Z / 100,               # atomic number (scaled)
        elem.X if elem.X else 0,    # electronegativity
        elem.atomic_mass / 300,     # mass (scaled)
        elem.row / 10,              # periodic‐table row
        elem.group / 20,            # periodic‐table group
    ]

CUTOFF = 5.0   # Å neighbour cutoff for edges

def structure_to_graph(struct: mg.Structure) -> Data:
    cart = struct.cart_coords
    num_atoms = len(struct)
    senders, receivers, dists = [], [], []

    lattice = struct.lattice
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue
            # minimum-image distance
            dist = lattice.get_distance_and_image(cart[i], cart[j])[0]
            if dist <= CUTOFF:
                senders.append(i); receivers.append(j); dists.append(dist)

    x = torch.tensor([atom_features(s) for s in struct], dtype=torch.float)
    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    edge_attr  = torch.tensor(dists, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ──────────────────── 2.  PyG in-memory dataset ──────────────────
class MPDataset(Dataset):
    def __init__(self, csv_path: Path):
        super().__init__()
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["structure", "formation_energy_per_atom"])
        self.y   = torch.tensor(df["formation_energy_per_atom"].values,
                                dtype=torch.float).view(-1, 1)
        # unpickle structures
        self.structs: list[mg.Structure] = [
            mg.core.structure.Structure.from_dict(json.loads(s))
            for s in df["structure"]
        ]

    def len(self): return len(self.structs)

    def get(self, idx):
        g = structure_to_graph(self.structs[idx])
        g.y = self.y[idx]
        return g

# ──────────────────── 3.  GNN model  ──────────────────────────────
class CGCNN(nn.Module):
    def __init__(self, in_dim=5, hidden=64, edge_dim=1, out_dim=1):
        super().__init__()
        self.conv1 = CGConv(in_dim, edge_dim=edge_dim, aggr="mean")
        self.conv2 = CGConv(in_dim, edge_dim=edge_dim, aggr="mean")
        self.conv3 = CGConv(in_dim, edge_dim=edge_dim, aggr="mean")
        self.fc    = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, ei, ea)
        x = self.conv2(x, ei, ea)
        x = self.conv3(x, ei, ea)
        x = global_mean_pool(x, batch)
        return self.fc(x)

# ──────────────────── 4.  train / eval loop ──────────────────────
def run_fold(train_idx, test_idx, dataset, epochs=30, batch_size=32):
    train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset[test_idx],  batch_size=batch_size)

    model = CGCNN().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.L1Loss()

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = lossf(pred, batch.y)
            loss.backward(); opt.step()

        if ep % 10 == 0 or ep == epochs:
            mae = evaluate(model, test_loader)
            print(f"epoch {ep:>3} | MAE  {mae:>6.3f} eV/atom")

    return evaluate(model, test_loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, ys = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        preds.append(model(batch).cpu())
        ys.append(batch.y.cpu())
    return mean_absolute_error(torch.cat(ys), torch.cat(preds))

# ──────────────────── 5.  k-fold experiment ──────────────────────
def main():
    ds = MPDataset(CSV)
    print(f"📊 Dataset  {len(ds)} structures")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(ds))), 1):
        print(f"\n🌀 Fold {fold} …")
        mae = run_fold(train_idx, test_idx, ds)
        maes.append(mae)

    print("\n────────────────────────────────────────")
    print("CV MAE (eV/atom):", [f"{m:.3f}" for m in maes])
    print(f"AVG ± STD  =  {np.mean(maes):.3f} ± {np.std(maes):.3f}")

if __name__ == "__main__":
    main()