// All content here is derived from the repository's own source of truth:
// README.md (context/intent) and NASA_pytorch.ipynb (actual implementation).
// Per the hierarchy of truth, code wins on any discrepancy — none were found
// between the README and the notebook for this repository as of the commit
// this site was generated from.

export const REPO = {
  owner: "shravanvinayhegde",
  name: "Predictive-Maintenance-PyTorch",
  url: "https://github.com/shravanvinayhegde/Predictive-Maintenance-PyTorch",
  notebookUrl:
    "https://github.com/shravanvinayhegde/Predictive-Maintenance-PyTorch/blob/main/NASA_pytorch.ipynb",
  colabUrl:
    "https://colab.research.google.com/github/shravanvinayhegde/Predictive-Maintenance-PyTorch/blob/main/NASA_pytorch.ipynb",
  tagline:
    "Remaining Useful Life (RUL) prediction for aircraft turbofan engines, built entirely on PyTorch, benchmarked on NASA's C-MAPSS engine degradation dataset.",
};

export const HYPERPARAMS = [
  { name: "WINDOW_SIZE", value: "30 cycles", where: "Sequence length fed to every model" },
  { name: "RUL_CAP", value: "125 cycles", where: "Piecewise-linear RUL target cap" },
  { name: "BATCH_SIZE", value: "64", where: "DataLoader batching" },
  { name: "EPOCHS", value: "10", where: "Training loop" },
  { name: "LR", value: "1e-3", where: "Adam optimizer learning rate" },
  { name: "Train/val split", value: "80% / 20%, grouped by engine", where: "GroupShuffleSplit(test_size=0.2)" },
  { name: "Random seed", value: "42", where: "torch.manual_seed, np.random.seed, random_state" },
];

export const SUBDATASETS = [
  { id: "FD001", train: 100, test: 100, conditions: 1, faults: "1 (HPC degradation)", used: true },
  { id: "FD002", train: 260, test: 259, conditions: 6, faults: "1 (HPC degradation)", used: false },
  { id: "FD003", train: 100, test: 100, conditions: 1, faults: "2 (HPC + Fan degradation)", used: false },
  { id: "FD004", train: 248, test: 249, conditions: 6, faults: "2 (HPC + Fan degradation)", used: false },
];

export const SENSORS = [
  { id: 1, symbol: "T2", desc: "Total temperature at fan inlet", unit: "°R" },
  { id: 2, symbol: "T24", desc: "Total temperature at LPC outlet", unit: "°R" },
  { id: 3, symbol: "T30", desc: "Total temperature at HPC outlet", unit: "°R" },
  { id: 4, symbol: "T50", desc: "Total temperature at LPT outlet", unit: "°R" },
  { id: 5, symbol: "P2", desc: "Pressure at fan inlet", unit: "psia" },
  { id: 6, symbol: "P15", desc: "Total pressure in bypass-duct", unit: "psia" },
  { id: 7, symbol: "P30", desc: "Total pressure at HPC outlet", unit: "psia" },
  { id: 8, symbol: "Nf", desc: "Physical fan speed", unit: "rpm" },
  { id: 9, symbol: "Nc", desc: "Physical core speed", unit: "rpm" },
  { id: 10, symbol: "epr", desc: "Engine pressure ratio (P50/P2)", unit: "—" },
  { id: 11, symbol: "Ps30", desc: "Static pressure at HPC outlet", unit: "psia" },
  { id: 12, symbol: "phi", desc: "Ratio of fuel flow to Ps30", unit: "pps/psi" },
  { id: 13, symbol: "NRf", desc: "Corrected fan speed", unit: "rpm" },
  { id: 14, symbol: "NRc", desc: "Corrected core speed", unit: "rpm" },
  { id: 15, symbol: "BPR", desc: "Bypass ratio", unit: "—" },
  { id: 16, symbol: "farB", desc: "Burner fuel-air ratio", unit: "—" },
  { id: 17, symbol: "htBleed", desc: "Bleed enthalpy", unit: "—" },
  { id: 18, symbol: "Nf_dmd", desc: "Demanded fan speed", unit: "rpm" },
  { id: 19, symbol: "PCNfR_dmd", desc: "Demanded corrected fan speed", unit: "rpm" },
  { id: 20, symbol: "W31", desc: "HPT coolant bleed", unit: "lbm/s" },
  { id: 21, symbol: "W32", desc: "LPT coolant bleed", unit: "lbm/s" },
];

export type PipelineStage = {
  id: string;
  n: string;
  title: string;
  summary: string;
  detail: string;
  cell: number;
  code: string;
  ioIn: string;
  ioOut: string;
};

export const PIPELINE: PipelineStage[] = [
  {
    id: "download",
    n: "01",
    title: "Download & extract",
    summary: "Fetch the C-MAPSS archive from NASA's PHM S3 mirror; unzip twice (zip nested in zip).",
    detail:
      "The dataset ships as a zip nested inside a zip — the outer download wraps a folder that itself contains CMAPSSData.zip — so extraction runs in two passes before the raw .txt files are reachable.",
    cell: 2,
    code: `url = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
response = requests.get(url)
response.raise_for_status()
with open("/content/CMAPSSData.zip", "wb") as f:
    f.write(response.content)`,
    ioIn: "HTTP GET → bytes",
    ioOut: "CMaps/*.zip on disk",
  },
  {
    id: "load",
    n: "02",
    title: "Load into DataFrames",
    summary: "Read whitespace-separated, headerless text files with explicit column names.",
    detail:
      "Each row is one engine at one operating cycle across 26 columns: unit_number, time_in_cycles, 3 operational settings, and 21 sensor_measurement channels.",
    cell: 8,
    code: `columns = (
    ["unit_number", "time_in_cycles", "operational_setting_1",
     "operational_setting_2", "operational_setting_3"]
    + [f"sensor_measurement_{i}" for i in range(1, 22)]
)
train_df = pd.read_csv(train_path, sep=r"\\s+", header=None, names=columns)`,
    ioIn: "train_FD001.txt / test_FD001.txt",
    ioOut: "train_df, test_df — DataFrame[rows × 26]",
  },
  {
    id: "eda",
    n: "03",
    title: "EDA: missing values & constant sensors",
    summary: "Check for NaNs; programmatically drop sensor columns that never change.",
    detail:
      "In FD001 a handful of sensor channels never vary across the whole dataset. The notebook detects them with nunique() <= 1 rather than hard-coding which ones — so the same cell stays correct if pointed at a different sub-dataset.",
    cell: 10,
    code: `constant_sensors = train_df[sensor_cols].columns[
    train_df[sensor_cols].nunique() <= 1
]
feature_cols = [c for c in sensor_cols + op_cols if c not in constant_sensors]`,
    ioIn: "train_df[26 cols]",
    ioOut: "feature_cols (constant sensors dropped)",
  },
  {
    id: "label",
    n: "04",
    title: "Engineer the RUL label",
    summary: "RUL(t) = final cycle − t, capped with a piecewise-linear function at 125.",
    detail:
      "Training engines run to failure, so the label comes directly from each engine's own max cycle. Raw RUL would grow unboundedly for long-lived engines, and early-life degradation is effectively negligible, so the target is capped at RUL_CAP = 125 — a standard construction in RUL-prediction literature.",
    cell: 15,
    code: `RUL_CAP = 125
max_cycle = train_df.groupby("unit_number")["time_in_cycles"].max()
train_df = train_df.merge(max_cycle.rename("max_cycle"), on="unit_number")
train_df["RUL"] = (train_df["max_cycle"] - train_df["time_in_cycles"]).clip(upper=RUL_CAP)`,
    ioIn: "train_df + time_in_cycles",
    ioOut: "train_df + RUL column (capped)",
  },
  {
    id: "split",
    n: "05",
    title: "Engine-grouped train/val split",
    summary: "GroupShuffleSplit by unit_number — no engine's cycles cross the split boundary.",
    detail:
      "Rows are not split randomly: consecutive cycles from the same engine are highly correlated, so a per-row split would leak adjacent cycles across train/validation. GroupShuffleSplit keeps every cycle from a given engine entirely on one side, 80/20.",
    cell: 17,
    code: `splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(train_df, groups=train_df["unit_number"]))
train_split = train_df.iloc[train_idx].copy()
val_split = train_df.iloc[val_idx].copy()`,
    ioIn: "train_df (100 engines)",
    ioOut: "train_split / val_split (grouped 80/20)",
  },
  {
    id: "scale",
    n: "06",
    title: "Standardize features",
    summary: "Mean/std computed on the training split only, then applied to val and test.",
    detail:
      "Fitting scaling statistics on data the model will later be evaluated on would leak information about that data's distribution into training — so mean and std come strictly from train_split.",
    cell: 19,
    code: `train_mean = train_split[feature_cols].mean()
train_std = train_split[feature_cols].std()
train_split[feature_cols] = (train_split[feature_cols] - train_mean) / train_std
val_split[feature_cols] = (val_split[feature_cols] - train_mean) / train_std`,
    ioIn: "train_split, val_split (raw scale)",
    ioOut: "train_split, val_split (z-scored)",
  },
  {
    id: "window",
    n: "07",
    title: "Slide a window per engine",
    summary: "WINDOW_SIZE=30 consecutive cycles → one (X, y) sequence, per engine, chronologically.",
    detail:
      "A neural net needs a short history to spot a trend, not one isolated reading. Every engine is sorted by time_in_cycles and swept with a size-30 sliding window; the label for each window is the RUL at its final cycle.",
    cell: 21,
    code: `def create_sequences(df, feature_cols, window_size):
    X, y = [], []
    for _, engine in df.groupby("unit_number"):
        engine = engine.sort_values("time_in_cycles")
        features, rul = engine[feature_cols].values, engine["RUL"].values
        for i in range(len(engine) - window_size + 1):
            X.append(features[i : i + window_size])
            y.append(rul[i + window_size - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)`,
    ioIn: "train_split / val_split [rows × n_features]",
    ioOut: "X [N, 30, n_features], y [N]",
  },
  {
    id: "dataset",
    n: "08",
    title: "Dataset & DataLoader",
    summary: "Wrap X/y NumPy arrays as tensors; batch with BATCH_SIZE=64.",
    detail:
      "A minimal torch.utils.data.Dataset (just __len__ / __getitem__) is enough — DataLoader handles batching and shuffling on top of it.",
    cell: 23,
    code: `class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(RULDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(RULDataset(X_val, y_val), batch_size=64, shuffle=False)`,
    ioIn: "X [N, 30, n_features], y [N]",
    ioOut: "batches of (X, y), shape [64, 30, n_features]",
  },
  {
    id: "testprep",
    n: "09",
    title: "Prepare the official test set",
    summary: "Only the most recent 30-cycle window per test engine; left-pad if an engine has fewer.",
    detail:
      "NASA's test engines are truncated before failure — the task is to predict RUL at that truncation point, so only each engine's final window is used. Engines with fewer than 30 recorded cycles are left-padded by repeating their first reading.",
    cell: 25,
    code: `def create_test_sequences(df, feature_cols, window_size):
    X = []
    for _, engine in df.groupby("unit_number"):
        engine = engine.sort_values("time_in_cycles")
        features = engine[feature_cols].values
        if len(features) < window_size:
            pad = np.repeat(features[0:1], window_size - len(features), axis=0)
            features = np.vstack([pad, features])
        X.append(features[-window_size:])
    return np.array(X, dtype=np.float32)`,
    ioIn: "test_df (truncated engines)",
    ioOut: "X_test [n_test_engines, 30, n_features]",
  },
];

export type ModelSpec = {
  id: string;
  name: string;
  kind: string;
  approach: string;
  keyLayers: string;
  cell: number;
  code: string;
  layers: { label: string; shape: string }[];
};

export const MODELS: ModelSpec[] = [
  {
    id: "mlp",
    name: "MLP",
    kind: "PyTorch",
    approach: "Flattens the whole window into one vector, ignores time order entirely — the simplest baseline.",
    keyLayers: "Flatten → Linear → ReLU → Linear → ReLU → Linear",
    cell: 29,
    code: `class MLPModel(nn.Module):
    def __init__(self, window_size, n_features, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)`,
    layers: [
      { label: "Input window", shape: "[B, 30, F]" },
      { label: "Flatten", shape: "[B, 30·F]" },
      { label: "Linear → ReLU", shape: "[B, 128]" },
      { label: "Linear → ReLU", shape: "[B, 64]" },
      { label: "Linear", shape: "[B, 1]" },
      { label: "squeeze(1)", shape: "[B]  →  RUL" },
    ],
  },
  {
    id: "cnn1d",
    name: "1D-CNN",
    kind: "PyTorch",
    approach: "Slides learnable filters along the time axis to pick up short local degradation trends.",
    keyLayers: "Conv1d → ReLU → Conv1d → ReLU → AdaptiveAvgPool1d → Linear",
    cell: 29,
    code: `class CNN1DModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, window, features) -> (B, features, window)
        x = self.conv(x).squeeze(-1)
        return self.head(x).squeeze(1)`,
    layers: [
      { label: "Input window", shape: "[B, 30, F]" },
      { label: "permute", shape: "[B, F, 30]" },
      { label: "Conv1d(k=3) → ReLU", shape: "[B, 64, 30]" },
      { label: "Conv1d(k=3) → ReLU", shape: "[B, 64, 30]" },
      { label: "AdaptiveAvgPool1d(1)", shape: "[B, 64, 1]" },
      { label: "Linear (head)", shape: "[B, 1]  →  RUL" },
    ],
  },
  {
    id: "lstm",
    name: "LSTM",
    kind: "PyTorch",
    approach: "Reads the window one cycle at a time, keeping a memory cell that can retain long-range information.",
    keyLayers: "LSTM(batch_first=True) → Linear on the final hidden state",
    cell: 29,
    code: `class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)   # h_n: (1, B, hidden)
        return self.head(h_n[-1]).squeeze(1)`,
    layers: [
      { label: "Input window", shape: "[B, 30, F]" },
      { label: "LSTM (batch_first)", shape: "h_n: [1, B, 64]" },
      { label: "h_n[-1]", shape: "[B, 64]" },
      { label: "Linear (head)", shape: "[B, 1]  →  RUL" },
    ],
  },
  {
    id: "gru",
    name: "GRU",
    kind: "PyTorch",
    approach: "Same recurrent idea as LSTM with simpler gating — fewer parameters, typically faster to train.",
    keyLayers: "GRU(batch_first=True) → Linear on the final hidden state",
    cell: 29,
    code: `class GRUModel(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1]).squeeze(1)`,
    layers: [
      { label: "Input window", shape: "[B, 30, F]" },
      { label: "GRU (batch_first)", shape: "h_n: [1, B, 64]" },
      { label: "h_n[-1]", shape: "[B, 64]" },
      { label: "Linear (head)", shape: "[B, 1]  →  RUL" },
    ],
  },
  {
    id: "rf",
    name: "Random Forest",
    kind: "scikit-learn (benchmark)",
    approach:
      "Ensemble of decision trees on the flattened window — a sanity check for whether the extra complexity of a neural net is actually earning its keep. Not part of the modeling approach; used strictly as a non-deep-learning benchmark.",
    keyLayers: "RandomForestRegressor(n_estimators=100, max_depth=10)",
    cell: 27,
    code: `X_train_flat = X_train.reshape(len(X_train), -1)
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_flat, y_train)`,
    layers: [
      { label: "Input window", shape: "[B, 30, F]" },
      { label: "reshape (flatten)", shape: "[B, 30·F]" },
      { label: "RandomForestRegressor", shape: "100 trees, depth ≤ 10" },
      { label: "predict", shape: "[B]  →  RUL" },
    ],
  },
];

export const TRAINING_LOOP_CODE = `EPOCHS = 10
LR = 1e-3

def train_model(model, name, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss, val_preds, val_true = 0, [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += loss_fn(preds, y_batch).item()
                val_preds.extend(preds.cpu().tolist())
                val_true.extend(y_batch.cpu().tolist())

        val_mae = mean_absolute_error(val_true, val_preds)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_mae"].append(val_mae)

    return history, time.time() - start_time`;

export const EVAL_CODE = `def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    return {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        "R2": r2_score(y_test, preds),
    }`;

export const RESULTS_ROWS = [
  { model: "MLP", mae: "Not reported in repository", rmse: "Not reported in repository", r2: "Not reported in repository" },
  { model: "1D-CNN", mae: "Not reported in repository", rmse: "Not reported in repository", r2: "Not reported in repository" },
  { model: "LSTM", mae: "Not reported in repository", rmse: "Not reported in repository", r2: "Not reported in repository" },
  { model: "GRU", mae: "Not reported in repository", rmse: "Not reported in repository", r2: "Not reported in repository" },
  { model: "Random Forest (benchmark)", mae: "Not reported in repository", rmse: "Not reported in repository", r2: "Not reported in repository" },
];

export type FileNode = {
  name: string;
  type: "file" | "dir";
  note?: string;
  children?: FileNode[];
};

export const FILE_TREE: FileNode[] = [
  { name: "NASA_pytorch.ipynb", type: "file", note: "Core project: NASA C-MAPSS RUL prediction pipeline" },
  {
    name: "Experiments/",
    type: "dir",
    note: "Standalone PyTorch notebooks on other domains/datasets",
    children: [
      { name: "Animal_face_classification (1).ipynb", type: "file", note: "Image classification, 5-model comparison" },
      { name: "Audio_Classification.ipynb", type: "file", note: "Audio classification (Quran reciters dataset)" },
      { name: "GTSRB.ipynb", type: "file", note: "Traffic sign recognition, 8-model comparison" },
      { name: "Rice_Classification.ipynb", type: "file", note: "Tabular classification (rice grain type)" },
      { name: "Text_Classification_Transformers.ipynb", type: "file", note: "Text classification with Transformers" },
    ],
  },
  {
    name: "Quick-Reaps/",
    type: "dir",
    note: "PyTorch fundamentals reference notebooks",
    children: [
      { name: "pytorch_01.ipynb", type: "file", note: "Tensors and core PyTorch operations" },
      { name: "pytorch_02.ipynb", type: "file", note: "Training workflow via linear regression" },
      { name: "pytorch_03.ipynb", type: "file", note: "Neural network classification (binary & multi-class)" },
      { name: "pytorch_04.ipynb", type: "file", note: "Computer vision on FashionMNIST" },
      { name: "pytorch_05.ipynb", type: "file", note: "Custom image datasets (Pizza/Steak/Sushi)" },
    ],
  },
  { name: "README.md", type: "file" },
];

export const LIMITATIONS = [
  "Single sub-dataset: only FD001 (1 operating condition, 1 fault mode) is used. FD002–FD004 add multiple operating conditions and a second fault mode, which typically requires per-condition normalization instead of a single global standardization.",
  "Fixed training budget: 10 epochs, no learning-rate scheduling, no early stopping, and no hyperparameter search — the comparison is meant to be a fair like-for-like baseline across architectures, not a tuned leaderboard entry.",
  "Single train/validation split: one GroupShuffleSplit, not k-fold cross-validation, so validation metrics carry some split-to-split variance.",
  "No attention-based or Transformer sequence model is included alongside the MLP/CNN/LSTM/GRU comparison.",
  "No official PHM scoring function: the standard C-MAPSS literature also reports an asymmetric \"PHM score\" that penalizes late predictions more than early ones; this project reports MAE/RMSE/R² only.",
];
