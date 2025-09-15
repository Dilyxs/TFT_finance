	
# --- must be first, before torch/numpy/pandas ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- standard libs ---
import random
import json
import ast
from datetime import datetime, timedelta,timezone

# --- third-party libs ---
import pandas as pd
import numpy as np
import joblib

# Torch (import once, not twice)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# limit torch threads after import
torch.set_num_threads(1)

# sklearn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)

# ML libs
import xgboost as xgb
import lightgbm as lgb
import optuna

# your project-specific modules
from elements import (
    df_for_news, ApplyScaling, get_news,
    df_for_interference, get_market_sentiment
)
from features_dict import dict_for_features
from PostGresConn import PostgresSQL
from MetaApiConn import MetaV2


def save_map_to_file(data_map: dict, output_filepath: None):
    output_filepath= output_filepath if output_filepath else f"{datetime.now(timezone.utc}_TFT_trading_file.txt"}
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, "w") as f:
        for key, value in data_map.items():
            f.write(f"{key}={value}\n")
    
    return output_filepath
class DetectedTrade:
	    """
	    Represents a detected trade opportunity and its metadata.
	
	    Attributes:
	        pair (str): The currency pair being traded, e.g., "EURUSD".
	        currentPrice (float): The current market price at the time of trade detection.
	        direction (int): Either 1 or 0.
	        risk (float): The percentage risk associated with the trade.
	        currentTime (datetime): The UTC timestamp when the trade was initialized.
	        expectedTime (datetime): The UTC timestamp 5 business days later, when the trade is expected to be closed.
	    """
	    
	    def __init__(self, pair, currentPrice, direction, risk):
	        self.pair = pair
	        self.currentPrice = currentPrice
	        self.direction = direction
	        self.risk = risk
	        self.currentTime, self.expectedTime = self.ReturnTime()
	        self.LimitClosure = pd.to_datetime(datetime.utcnow() + timedelta(hours=72))
	
	    def ReturnTime(self):
	        currentTime = datetime.utcnow()
	        currentTime = pd.to_datetime(currentTime)
	        futureTime = get_5th_business_day_later(currentTime)
	        return currentTime, futureTime

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Should be a Tensor of shape [num_classes]
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TFTInferenceDataset(Dataset):
    def __init__(self, features, encoder_len=30, decoder_len=1, pred_step=5):
        self.features = features
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.pred_step = pred_step
        self.max_idx = len(features) - pred_step - decoder_len

    def __len__(self):
        return self.max_idx - self.encoder_len + 1

    def __getitem__(self, idx):
        encoder_input = self.features[idx:idx + self.encoder_len]
        decoder_input = np.repeat(encoder_input[-1:], self.decoder_len, axis=0)

        return (
            torch.tensor(encoder_input, dtype=torch.float32),
            torch.tensor(decoder_input, dtype=torch.float32)
        )




# --- 1. TFTClassificationDataset ---
class TFTClassificationDataset(Dataset):
    def __init__(self, features, labels, encoder_len=30, decoder_len=1, pred_step=5):
        self.features = features
        self.labels = labels
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.pred_step = pred_step
        self.max_idx = len(features) - pred_step - decoder_len

    def __len__(self):
        return self.max_idx - self.encoder_len + 1

    def __getitem__(self, idx):
        encoder_input = self.features[idx:idx + self.encoder_len]
        decoder_input = np.repeat(encoder_input[-1:], self.decoder_len, axis=0)
        target_idx = idx + self.encoder_len + self.decoder_len + self.pred_step - 1
        y = self.labels[target_idx]

        return (
            torch.tensor(encoder_input, dtype=torch.float32),
            torch.tensor(decoder_input, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

# --- 2. TFTClassifier ---
class TFTClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, 
                 num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(TFTClassifier, self).__init__()

        # Project input_dim → hidden_dim (embed dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  # ✅ FIXED
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,  # ✅ FIXED
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, encoder_input, decoder_input):
        # Project inputs to hidden_dim
        encoder_input = self.input_proj(encoder_input)
        decoder_input = self.input_proj(decoder_input)

        enc_out = self.encoder(encoder_input)
        dec_out = self.decoder(decoder_input, enc_out)

        combined = torch.cat((enc_out[:, -1, :], dec_out[:, -1, :]), dim=1)
        gated = self.gate(combined)
        output = self.classifier(gated)
        return output

# --- 3. PreProcessing ---
def PreProcessing(seq_len, pred_step, split_ratio, currency, batch_size_train, batch_size_test, seed=42):
    set_global_seed(seed)

    df = df_for_news(currency, "D")
    df = ApplyScaling(df, currency)
    df.set_index('time', inplace=True)
    df.dropna(inplace=True)

    df_exo = df[dict_for_features[currency]].apply(pd.to_numeric, errors='coerce')
    df_y = df[['future_close_encoded']]

    split_index = round(len(df) * split_ratio)

    X_train = df_exo.iloc[:split_index].to_numpy()
    y_train = df_y.iloc[:split_index]['future_close_encoded'].to_numpy()

    X_test = df_exo.iloc[split_index:].to_numpy()
    y_test = df_y.iloc[split_index:]['future_close_encoded'].to_numpy()

    train_dataset = TFTClassificationDataset(X_train, y_train, encoder_len=seq_len)
    test_dataset = TFTClassificationDataset(X_test, y_test, encoder_len=seq_len)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, generator=g)

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset, train_loader, test_loader, df, split_index


# --- 4. ModelReturner ---
def ModelReturner(X_train, y_train, hidden_dim, output_dim, dropout, lr, n_heads, 
                  num_encoder_layers, num_decoder_layers,alpha_scale = 1, gamma=2, seed=42):
    set_global_seed(seed)  # ← ensure deterministic weight init

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]

    model = TFTClassifier(input_dim, hidden_dim, output_dim, n_heads=n_heads, 
                          num_encoder_layers=num_encoder_layers, 
                          num_decoder_layers=num_decoder_layers, 
                          dropout=dropout).to(device)

    # Adjust or load weights
    y_train_np = np.array(y_train)
    class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train_np),
                                     y=y_train_np)

    alpha = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(alpha=alpha * alpha_scale, gamma=gamma)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer, device

# --- 5. train ---
def train(model, train_loader, criterion, optimizer, device, n_epochs, seed = 42):
    set_global_seed(seed)
    for epoch in range(n_epochs):
        model.train()
        
        running_loss = 0.0

        for encoder_in, decoder_in, y_batch in train_loader:
            encoder_in, decoder_in, y_batch = encoder_in.to(device), decoder_in.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(encoder_in, decoder_in)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")

    return model

# --- 6. evaluate ---
def evaluate(model, test_loader, device, test_dataset, df, split_index):
    model.eval()
    all_preds, all_true, all_prob = [], [], []

    with torch.no_grad():
        for encoder_in, decoder_in, y_batch in test_loader:
            encoder_in, decoder_in = encoder_in.to(device), decoder_in.to(device)
            logits = model(encoder_in, decoder_in)

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            prob_array = probs.cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())
            all_prob.extend(prob_array)

            for i, prob in enumerate(prob_array):
                print(f"Sample {i+1} prediction percentages: {(prob * 100).round(2)}")
                print(f"Predicted class: {preds[i]} with probability {prob[preds[i]]*100:.2f}%")

    accuracy = accuracy_score(all_true, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    results_df = pd.DataFrame({
        'true': all_true,
        'pred': all_preds,
        'prob_0': [p[0] for p in all_prob],
        'prob_1': [p[1] for p in all_prob],
        'prob_2': [p[2] for p in all_prob],
    })

    # Align timestamps (key fix)
    enc_len = test_dataset.encoder_len
    dec_len = test_dataset.decoder_len
    pred_step = test_dataset.pred_step

    relative_target_indices = [
        i + enc_len + dec_len + pred_step - 1
        for i in range(len(test_dataset))
    ]
    global_indices = [split_index + i for i in relative_target_indices]
    timestamps = df.index[global_indices]
    results_df["timestamp"] = timestamps.values

    return accuracy, all_preds, all_true, all_prob, results_df

def get_best_params(currency, df = None):
    if not df:
        df = pd.read_csv("TFT_PARAMS_V101.csv")
    row = df[df['currency'] == currency].iloc[0]
    return {
        'seq_len': int(row['seq_len']),
        'batch_size_train': int(row['batch_size_train']),
        'hidden_dim': int(row['hidden_dim']),
        'n_epochs': int(row['n_epochs']),
        'lr': float(row['lr']),
        'num_encoder_layers': int(row['num_encoder_layers']),
        'num_decoder_layers': int(row['num_decoder_layers']),
        'n_heads': int(row['n_heads']),
        'dropout': float(row['dropout']),
        'gamma': float(row['gamma']),
        'alpha_scale': float(row['alpha_scale']),
    }



def save_tft_model(model, config, currency):
    # Save model weights
    torch.save(model.state_dict(), f"./SavedModels/TFT_{currency}.pth")
    
    # Save architecture config
    with open(f"./SavedModels/TFT_{currency}_config.json", "w") as f:
        json.dump(config, f)

def load_tft_model(currency, model_class=TFTClassifier, device="cpu"):
    # Load config
    with open(f"./SavedModels/TFT_{currency}_config.json", "r") as f:
        config = json.load(f)
    
    # Reconstruct model with same config
    model = model_class(**config)
    
    # Load weights
    model.load_state_dict(torch.load(f"./SavedModels/TFT_{currency}.pth", map_location=device))
    model.to(device)
    model.eval()

    return model

def PipelineReturner(currency):
    # === Step 1: Run TFT model ===
    results_df, X_train, y_train, X_test, y_test, split_index, df = TFT_returner(currency, save_data=True)

    # === Step 2: Run XGB ===
    res_xgb, modelXGB = XGBReturner(X_train, y_train, X_test, y_test, currency, df, split_index)
    joblib.dump(modelXGB, f"./SavedModels/XGB_{currency}.pkl")

    # === Step 3: Run LGB ===
    res_lgb, modelLGB = LGBReturner(X_train, y_train, X_test, y_test, currency, df, split_index)
    joblib.dump(modelLGB, f"./SavedModels/LGB_{currency}.pkl")

    # === Step 4: Merge all results ===
    return results_df, res_xgb, res_lgb


def apply_decision_logic(df, tft_thresh, xgb_thresh, lgb_thresh):
    return np.where(
        (df['pred'] == 2) & (df['Prediction_xgb'] == 2) & (df['Prediction_lgb'] == 2) &
        (df['prob'] > tft_thresh) & (df['Confidence_xgb'] > xgb_thresh) & (df['Confidence_lgb'] > lgb_thresh),
        2,
        np.where(
            (df['pred'] == 0) & (df['Prediction_xgb'] == 0) & (df['Prediction_lgb'] == 0) &
            (df['prob'] > tft_thresh) & (df['Confidence_xgb'] > xgb_thresh) & (df['Confidence_lgb'] > lgb_thresh),
            0,
            1
        )
    )

def Analyzer(currency, results_df, res_xgb, res_lgb, tft_thresh, xgb_thresh, lgb_thresh,how_many_days = 10):
    merged_df = pd.merge(
        results_df[['pred', 'prob', 'prob_0', 'prob_1', 'prob_2']],
        res_xgb[['Prediction_xgb', 'Confidence_xgb', 'Prob_xgb_0', 'Prob_xgb_1', 'Prob_xgb_2', 'close', 'future_close']],
        how='inner', left_index=True, right_index=True
    )

    m2 = pd.merge(
        merged_df,
        res_lgb[['Prediction_lgb', 'Confidence_lgb', 'Prob_lgb_0', 'Prob_lgb_1', 'Prob_lgb_2']],
        how='inner', left_index=True, right_index=True
    )
    length = round(0.8 * len(m2))
    m2 = m2[length:]

    # === Step 5: Decision logic ===
    m2['decision'] = apply_decision_logic(m2, tft_thresh, xgb_thresh, lgb_thresh)

    # === Step 6: Calculate profits ===
    m2['future_close_10'] = m2['close'].shift(-how_many_days)

    m2['profit_1'] = np.where(
        m2['decision'] == 2,
        m2['future_close'] - m2['close'],
        np.where(m2['decision'] == 0, m2['close'] - m2['future_close'], 0)
    )

    m2['profit_2'] = np.where(
        m2['decision'] == 2,
        m2['future_close_10'] - m2['close'],
        np.where(m2['decision'] == 0, m2['close'] - m2['future_close_10'], 0)
    )
    JPY_like = ['USD_JPY', 'USD_MXN', 'USD_ZAR']
    pip_size = 0.01 if currency in JPY_like else 0.0001

    # Pip gains
    m2['pip_gain_1'] = m2['profit_1'] / pip_size
    m2['pip_gain_2'] = m2['profit_2'] / pip_size

    # === Step 7: Summary output ===
    summary = {
        'currency': currency,
        'total_profit_1': m2['profit_1'].sum(),
        'total_profit_2': m2['profit_2'].sum(),
        'avg_pips_per_trade_1': m2[m2['decision'] != 1]['pip_gain_1'].mean(),
        'avg_pips_per_trade_2': m2[m2['decision'] != 1]['pip_gain_2'].mean(),
        'num_trades': (m2['decision'] != 1).sum(),
        'win_rate_profit_1': (m2['profit_1'] > 0).sum() / (m2['decision'] != 1).sum(),
        'win_rate_profit_2': (m2['profit_2'] > 0).sum() / (m2['decision'] != 1).sum(),
    }

    return summary, m2 # m2 contains all merged details, useful for further analysis

def TFT_returner(currency, save_data=False, filepath = None):
    if not filepath:
        df1 = pd.read_csv("TFT_PARAMS_V101.csv")
    else:
        df1 = pd.read_csv(filepath)

    row = df1[df1['currency'] == currency].iloc[0]

    # Extract values
    seq_len = int(row['seq_len'])
    batch_size_train = int(row['batch_size_train'])
    hidden_dim = int(row['hidden_dim'])
    n_epochs = int(row['n_epochs'])
    lr = float(row['lr'])
    num_encoder_layers = int(row['num_encoder_layers'])
    num_decoder_layers = int(row['num_decoder_layers'])
    n_heads = int(row['n_heads'])
    dropout = float(row['dropout'])
    alpha = float(row['alpha_scale'])
    gamma = float(row['gamma'])
    SEED = 1
    output_dim = 3
    split_ratio = 0.8
    batch_size_test = 1
    pred_step = 5

    set_global_seed(SEED)


    # Load data
    X_train, y_train, X_test, y_test, train_dataset, test_dataset, train_loader, test_loader,df, split_index= PreProcessing(
        seq_len=seq_len,
        pred_step=pred_step,
        split_ratio=split_ratio,
        currency=currency,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        seed=SEED
    )

    # Initialize model
    model, criterion, optimizer, device = ModelReturner(
        X_train=X_train,
        y_train=y_train,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_heads=n_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout,
        lr=lr,
        seed=SEED,
        alpha_scale = alpha,
        gamma = gamma
    )

    # Train and evaluate
    model = train(model, train_loader, criterion, optimizer, device, n_epochs=n_epochs)
    accuracy, _,_,_,results_df = evaluate(model, test_loader, device, test_dataset, df, split_index)
    results_df.set_index(results_df.timestamp, inplace=True)
    results_df['prob'] = results_df[['prob_0', 'prob_1', 'prob_2']].max(axis=1)

    if save_data:
        config = {
            "input_dim": X_train.shape[1],  # num_features
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "n_heads": n_heads,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dropout": dropout
        }
        save_tft_model(model, config, currency)

    return results_df, X_train, y_train, X_test, y_test, split_index, df

def XGBReturner(X_train, y_train, X_test, y_test, pair,df, split_index):
    model = TrainXGB(X_train, y_train, pair)
    res = EvaluateXGB(model, X_test, y_test, df[split_index:])
    return res, model

def LGBReturner(X_train, y_train, X_test, y_test, pair,df, split_index):
    model = TrainLGB(X_train, y_train, pair)
    res = EvaluateLGB(model, X_test, y_test, df[split_index:])
    return res, model

def predict(model, features, batch_size, encoder_len, decoder_len=1, pred_step=5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    inference_dataset = TFTInferenceDataset(
        features=features,
        encoder_len=encoder_len,
        decoder_len=decoder_len,
        pred_step=pred_step
    )
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for encoder_in, decoder_in in inference_loader:
            encoder_in = encoder_in.to(device)
            decoder_in = decoder_in.to(device)

            logits = model(encoder_in, decoder_in)  # Shape: (batch, num_classes)
            probs = F.softmax(logits, dim=1)        # Convert to probabilities
            preds = torch.argmax(probs, dim=1)      # Choose most likely class

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    final_preds = torch.cat(all_preds).numpy()
    final_probs = torch.cat(all_probs).numpy()

    return final_preds, final_probs

def ReturnTFTPred(pair):
    params = get_best_params(pair)
    model = load_tft_model(pair)
    df_pred = df_for_news(pair, "D")
    df_pred = ApplyScaling(df_pred, pair)
    df_pred.set_index('time', inplace=True)
    X_test = df_pred[dict_for_features[pair]].apply(pd.to_numeric, errors='coerce').to_numpy()#this seleects the features
    
    
    preds, probs = predict(
        model=model,
        features=X_test,
        batch_size=64, #for now
        encoder_len=params['seq_len'],
        decoder_len=1, #never change it
        pred_step=5#never change it
    )
    
    # Optional: Wrap into a DataFrame
    pred_df = pd.DataFrame({
        "pred": preds,
        "prob":np.max(probs, axis=1),
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "prob_2": probs[:, 2],
    })
    
    enc_len = params['seq_len']
    dec_len = 1
    pred_step = 5
    
    relative_target_indices = [
        i + enc_len + dec_len + pred_step - 1
        for i in range(len(pred_df))
    ]
    timestamps = df_pred.index[relative_target_indices]
    
    pred_df["timestamp"] = timestamps.values
    pred_df.set_index("timestamp", inplace=True)
    return pred_df, X_test, df_pred


def TrainXGB(X_train, y_train, pair):
    df_merged = pd.read_csv("optuma_given_param.csv")
    params = df_merged[(df_merged.pair == pair) & (df_merged.model == "XBGboost")]['param'].values[0]
    params = ast.literal_eval(params)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
    model.fit(X_train, y_train)

    return model

def EvaluateXGB(model, X_test, y_test, df_test):
    y_pred_prob_test = model.predict_proba(X_test)
    y_pred_test = np.argmax(y_pred_prob_test, axis=1)

    df_test = df_test.copy()
    df_test['Prediction_xgb'] = y_pred_test
    df_test['Confidence_xgb'] = np.max(y_pred_prob_test, axis=1)
    df_test['Prob_xgb_0'] = y_pred_prob_test[:, 0]
    df_test['Prob_xgb_1'] = y_pred_prob_test[:, 1]
    df_test['Prob_xgb_2'] = y_pred_prob_test[:, 2]

    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"XGBoost Accuracy: {accuracy * 100:.2f}%")

    return df_test

def Predict_XGB(X_test, df_test, pair):
    model = joblib.load(f"./SavedModels/XGB_{pair}.pkl")
    y_pred_prob_test = model.predict_proba(X_test)
    y_pred_test = np.argmax(y_pred_prob_test, axis=1)

    df_test = df_test.copy()
    df_test['Prediction_xgb'] = y_pred_test
    df_test['Confidence_xgb'] = np.max(y_pred_prob_test, axis=1)
    df_test['Prob_xgb_0'] = y_pred_prob_test[:, 0]
    df_test['Prob_xgb_1'] = y_pred_prob_test[:, 1]
    df_test['Prob_xgb_2'] = y_pred_prob_test[:, 2]

    return df_test

def Predict_LGB(X_test, df_test, pair):
    model = joblib.load(f"./SavedModels/LGB_{pair}.pkl")
    y_pred_prob_test = model.predict_proba(X_test)
    y_pred_test = np.argmax(y_pred_prob_test, axis=1)

    df_test = df_test.copy()
    df_test['Prediction_lgb'] = y_pred_test
    df_test['Confidence_lgb'] = np.max(y_pred_prob_test, axis=1)
    df_test['Prob_lgb_0'] = y_pred_prob_test[:, 0]
    df_test['Prob_lgb_1'] = y_pred_prob_test[:, 1]
    df_test['Prob_lgb_2'] = y_pred_prob_test[:, 2]
    return df_test

    



def TrainLGB(X_train, y_train, pair):
    df_merged = pd.read_csv("optuma_given_param.csv")
    params = df_merged[(df_merged.pair == pair) & (df_merged.model == "LBGboost")]['param'].values[0]
    params = ast.literal_eval(params)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    return model

def EvaluateLGB(model, X_test, y_test, df_test, model_dir="models"):
    y_pred_prob_test = model.predict_proba(X_test)
    y_pred_test = np.argmax(y_pred_prob_test, axis=1)

    df_test = df_test.copy()
    df_test['Prediction_lgb'] = y_pred_test
    df_test['Confidence_lgb'] = np.max(y_pred_prob_test, axis=1)
    df_test['Prob_lgb_0'] = y_pred_prob_test[:, 0]
    df_test['Prob_lgb_1'] = y_pred_prob_test[:, 1]
    df_test['Prob_lgb_2'] = y_pred_prob_test[:, 2]

    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"LightGBM Accuracy: {accuracy * 100:.2f}%")
    return df_test


def ReturnLastRow(pair, last_row = True):
    results_df, X_test, df_pred = ReturnTFTPred(pair)
    res_lgb = Predict_LGB(X_test, df_pred, pair)
    res_xgb = Predict_XGB(X_test, df_pred, pair)
    merged_df = pd.merge(
            results_df[['pred', 'prob', 'prob_0', 'prob_1', 'prob_2']],
            res_xgb[['Prediction_xgb', 'Confidence_xgb', 'Prob_xgb_0', 'Prob_xgb_1', 'Prob_xgb_2', 'close', 'future_close']],
            how='inner', left_index=True, right_index=True
        )
    
    m2 = pd.merge(
        merged_df,
        res_lgb[['Prediction_lgb', 'Confidence_lgb', 'Prob_lgb_0', 'Prob_lgb_1', 'Prob_lgb_2']],
        how='inner', left_index=True, right_index=True
    )
    if last_row:
        return m2[-1:]
    else:
        return m2

def CombineWithSentiment(pair, moreData = False):
    if not moreData:
        last_row = ReturnLastRow(pair)
    if moreData:
        last_row = ReturnLastRow(pair, last_row=False)
    
    data = get_market_sentiment(pair)
    data = data[['market_direction', 'ema_encoded_cot']]

    # Ensure datetime index and proper formatting
    last_row = last_row.copy()
    data = data.copy()

    # Convert index to datetime explicitly and flatten to column
    last_row.index = pd.to_datetime(last_row.index)
    data.index = pd.to_datetime(data.index)

    last_row = last_row.reset_index().rename(columns={'index': 'datetime'})
    data = data.reset_index().rename(columns={'index': 'datetime'})

    # Sort both by datetime for merge_asof
    last_row = last_row.sort_values('datetime')
    data = data.sort_values('datetime')

    # Perform the asof merge (backward = use earlier or equal sentiment)
    df = pd.merge_asof(
        last_row,
        data,
        on='datetime',
        direction='backward'  # could be 'nearest' if that's better
    )

    df.set_index('datetime', inplace=True)
    return df

def ReturnWithTradeActivation(pair):
    res = CombineWithSentiment(pair, moreData=False)
    df = pd.read_csv("tft_xgb_params_thresh_optimized.csv")
    
    tft_thresh = df[df.currency == pair]['tft_thresh'].values[0]
    xgb_thresh = df[df.currency == pair]['xgb_thresh'].values[0]
    lgb_thresh = df[df.currency == pair]['lgb_thresh'].values[0]

    # Trade signal logic
    res['is_trade'] = np.where(
        (res['pred'] == res['Prediction_xgb']) &
        (res['pred'] == res['Prediction_lgb']) &
        (res['prob'] > tft_thresh) &
        (res['Confidence_xgb'] > xgb_thresh) &
        (res['Confidence_lgb'] > lgb_thresh) &
        (res['pred'] == 2),
        2,
        np.where(
            (res['pred'] == res['Prediction_xgb']) &
            (res['pred'] == res['Prediction_lgb']) &
            (res['prob'] > tft_thresh) &
            (res['Confidence_xgb'] > xgb_thresh) &
            (res['Confidence_lgb'] > lgb_thresh) &
            (res['pred'] == 0),
            0,
            1
        )
    )

    # Super trade logic
    res['super_trade'] = np.where(
        (res['market_direction'] == 1) &
        (res['ema_encoded_cot'] == 1) &
        (res['is_trade'] == 2),
        2,
        np.where(
            (res['market_direction'] == 0) &
            (res['ema_encoded_cot'] == 0) &
            (res['is_trade'] == 0),
            0,
            1
        )
    )

    return res

def get_5th_business_day_later(trade_time,Exact = False):
    """
    Given a datetime `trade_time`, return the datetime at the same time on the 5th *business* day after.
    Business days exclude weekends (Saturday & Sunday).
    
    Example: If trade is Monday 17:02, result itals next Monday 17:02.
    """
    # Create a pandas DateOffset object that adds 5 business days
    fifth_business_day = pd.date_range(start=trade_time, periods=6, freq='B')[-1]

    # Keep the original trade time (hour, minute, second)
    new_datetime = fifth_business_day.replace(
        hour=trade_time.hour,
        minute=trade_time.minute,
        second=trade_time.second,
        microsecond=trade_time.microsecond
    ) if Exact else fifth_business_day.replace(
        hour=20,
        minute=50,
        second=1,
        microsecond=trade_time.microsecond
    )
    

    return new_datetime


import logging
import pandas as pd

def calculate_risk(pair, res, df_params=None, base_risk=0.4):
    """
    Calculate the adjusted risk for a given trading pair.

    Args:
        pair (str): The trading pair.
        res (DataFrame-like): Result from ReturnWithTradeActivation(pair).
        df_params (DataFrame, optional): DataFrame containing model scores. If None, it will be loaded from CSV.
        base_risk (float): Initial base risk.

    Returns:
        float: Adjusted risk value.
        int: Binary trade signal (0 or 1).
    """
    if df_params is None:
        df_params = pd.read_csv("tft_xgb_params_thresh_optimized.csv")

    reg_risk = base_risk
    if pair=="USD_MXN" or pair=="USD_ZAR":
        reg_risk*=0.80
    # Adjust risk based on model score
    try:
        model_score = df_params[df_params.currency == pair].score.values[0]
        if model_score != "DNE":
            reg_risk *= 1.1
    except IndexError:
        logging.warning(f"No model score found for {pair}, skipping.")
        return None, None

    trade_signal = res.is_trade.values[0]

    if trade_signal == 1:
        print(f"no trade detected for {pair}")
        return 0, None  # early exit

    if trade_signal == 2:
        if res.super_trade.values[0] == trade_signal:
            reg_risk *= 1.25
    elif trade_signal == 0:
        if res.super_trade.values[0] == trade_signal:
            reg_risk *= 1.25

    trade_signal = 2 if trade_signal == 1 else 0

    return reg_risk, trade_signal


def insert_detected_trade(db, pair, current_price, trade_signal, risk):
    """
    Create a DetectedTrade object, convert it to a dictionary, and insert into the database.

    Args:
        db: Database connection object with InsertData method.
        pair (str): Currency pair.
        current_price (float): Current market price.
        trade_signal (int): Trade direction (0 or 1).
        risk (float): Risk percentage.
    """
    # Create the DetectedTrade object
    trade = DetectedTrade(pair, current_price, trade_signal, risk)
    
    # Convert to dictionary
    trade_details = {
        "pair": trade.pair,
        "currentprice": trade.currentPrice,
        "direction": trade.direction,
        "risk": trade.risk,
        "time": trade.currentTime,
        "expectedclosetime": trade.expectedTime
    }
    
    # Insert into database
    db.InsertData("detectedtrades", trade_details)
    
    return trade_details, trade



class LimitOrder:
    """
    Represents a limit order record for database insertion.
    """
    def __init__(self, res, account_id, tradeexpiration, tradeSignal):
        self.id = int(res['magic'])
        self.account_id = str(account_id)
        self.pair = res["symbol"]
        self.entryprice = res["openPrice"]
        self.stoploss = res.get('stopLoss', None)
        self.units = res["volume"]
        self.direction = tradeSignal
        self.expirationtimelimit = res['expirationTime']
        self.limitordersentday = res["time"]
        self.comment = None
        self.data = json.dumps(res, default=str)
        self.id_broker = float(res["id"])
        self.tradeexpiration = tradeexpiration

    def to_dict(self):
        """
        Convert the LimitOrder object into a dictionary for database insertion.
        """
        return {
            "id": self.id,
            "account_id": self.account_id,
            "pair": self.pair,
            "entryprice": self.entryprice,
            "stoploss": self.stoploss,
            "units": self.units,
            "expirationtimelimit": self.expirationtimelimit,
            "limitordersentday": self.limitordersentday,
            "comment": self.comment,
            "data": self.data,
            "id_broker": self.id_broker,
            "tradeexpiration": self.tradeexpiration,
			"direction" : self.direction
        }



def insert_limit_order(db, res, account_id, tradeexpiration, tradeSignal):
    """
    Create a LimitOrder object from res and insert into 'limitorderstable'.
	res here represents the order_details after sending the data
    """
    limit_order = LimitOrder(res, account_id, tradeexpiration ,tradeSignal)
    trade_code = db.InsertData("limitorderstable", limit_order.to_dict())
    return trade_code

def MakeAnId():
    return random.randint(1, 2**53 - 1)  # fits in double without rounding




class ActiveTrade:
    """
    Represents a limit order record for database insertion.
    """
    def __init__(self, res, account_id, expirationTime):
        self.id = int(res['magic']) 
        self.pair = res["symbol"]
        self.entryprice = res["openPrice"]
        self.stoploss = res.get('stopLoss', None)
        self.units = res["volume"]
        self.expirationtimelimit = expirationTime
        self.limitordersentday = res["time"]
        self.comment = None
        self.data = json.dumps(res, default=str)
        self.id_broker = float(res["id"])
        self.account_id = str(account_id) 

    def to_dict(self):
        """
        Convert the LimitOrder object into a dictionary for database insertion.
        """
        return {
            "id": self.id,
            "pair": self.pair,
            "entryprice": self.entryprice,
            "stoploss": self.stoploss,
            "units": self.units,
            "expirationtimelimit": self.expirationtimelimit,
            "limitordersentday": self.limitordersentday,
            "comment": self.comment,
            "data": self.data,
            "id_broker": self.id_broker,
            "account_id": self.account_id
        }


def insert_active_trade(db, res, account_id, expirationTime):
    """
    Create a LimitOrder object from res and insert into 'limitorderstable'.
    `res` represents the order_details after sending the data.
    """
    order = ActiveTrade(res, account_id, expirationTime)
    trade_code = db.InsertData("limitorderstable", order.to_dict())
    return trade_code


def map_limit_order_to_active_trade(order: dict) -> dict:

    trade_dict = {
        "id": order.get("id"),  # auto-generated by sequence in activetrades
        "pair": order.get("pair"),
        "entryprice": order.get("entryprice"),
        "stoploss": order.get("stoploss"),
        "units": order.get("units"),
        "tradeexpiration": order.get("tradeexpiration"),
        "limitordersentday": order.get("limitordersentday"),
        "comment": order.get("comment"),
        "data": json.dumps(order.get("data"), default=str),
        "id_broker": order.get("id_broker"),
        "account_id": order.get("account_id"),
		"direction":order.get("direction")
    }
    return trade_dict


def ReturnWithTradeActivation(pair):
    res = CombineWithSentiment(pair, moreData=False)
    df = pd.read_csv("tft_xgb_params_thresh_optimized.csv")
    
    tft_thresh = df[df.currency == pair]['tft_thresh'].values[0]
    xgb_thresh = df[df.currency == pair]['xgb_thresh'].values[0]
    lgb_thresh = df[df.currency == pair]['lgb_thresh'].values[0]

    # Trade signal logic
    res['is_trade'] = np.where(
        (res['pred'] == res['Prediction_xgb']) &
        (res['pred'] == res['Prediction_lgb']) &
        (res['prob'] > tft_thresh) &
        (res['Confidence_xgb'] > xgb_thresh) &
        (res['Confidence_lgb'] > lgb_thresh) &
        (res['pred'] == 2),
        2,
        np.where(
            (res['pred'] == res['Prediction_xgb']) &
            (res['pred'] == res['Prediction_lgb']) &
            (res['prob'] > tft_thresh) &
            (res['Confidence_xgb'] > xgb_thresh) &
            (res['Confidence_lgb'] > lgb_thresh) &
            (res['pred'] == 0),
            0,
            1
        )
    )

    # Super trade logic
    res['super_trade'] = np.where(
        (res['market_direction'] == 1) &
        (res['ema_encoded_cot'] == 1) &
        (res['is_trade'] == 2),
        2,
        np.where(
            (res['market_direction'] == 0) &
            (res['ema_encoded_cot'] == 0) &
            (res['is_trade'] == 0),
            0,
            1
        )
    )

    return res

async def ReturnMapWithPositions(meta):
    result = await meta.ReturnAllPositions()
    Map ={}
    for data in result:
        Map[int(data.get('id'))] = data
    await meta.disconnect_all_conn()
    
    return Map

async def PositionDoubleMapping():
    '''First Map takes (account_id,access_token) as the Map key which points to a map of all
    positions where position_id is the key for that map'''
    db = PostgresSQL()
    Map = {}
    
    data = db.FetchAllData('accountdata')
    for result in data:
        account_id = result.get('account_id')
        access_token = result.get('access_token')
        meta = MetaV2(access_token, account_id)
        Mapping_Where_Index_Equal_Positions = await ReturnMapWithPositions(meta)
        Map[(account_id, access_token)] = Mapping_Where_Index_Equal_Positions
        await meta.disconnect_all_conn()
    return Map

        
        

def DetermineIfOrderIsFilled(PositionId, DoubleMapping, acc_id, access_token):
    PositionId = int(PositionId)
    AllPosition = DoubleMapping[(acc_id, access_token)]

    return True if PositionId in AllPosition else False
    
 
async def ReturnMappingPositions():
    tries=0 
    while tries<3 :
        try:
            Map = await PositionDoubleMapping()
            return Map
        
        except Exception as e:
            continue
            tries+=1
def ReturnWithTradeActivation(pair):
    res = CombineWithSentiment(pair, moreData=False)
    df = pd.read_csv("tft_xgb_params_thresh_optimized.csv")
    
    tft_thresh = df[df.currency == pair]['tft_thresh'].values[0]
    xgb_thresh = df[df.currency == pair]['xgb_thresh'].values[0]
    lgb_thresh = df[df.currency == pair]['lgb_thresh'].values[0]

    # Trade signal logic
    res['is_trade'] = np.where(
        (res['pred'] == res['Prediction_xgb']) &
        (res['pred'] == res['Prediction_lgb']) &
        (res['prob'] > tft_thresh) &
        (res['Confidence_xgb'] > xgb_thresh) &
        (res['Confidence_lgb'] > lgb_thresh) &
        (res['pred'] == 2),
        2,
        np.where(
            (res['pred'] == res['Prediction_xgb']) &
            (res['pred'] == res['Prediction_lgb']) &
            (res['prob'] > tft_thresh) &
            (res['Confidence_xgb'] > xgb_thresh) &
            (res['Confidence_lgb'] > lgb_thresh) &
            (res['pred'] == 0),
            0,
            1
        )
    )

    # Super trade logic
    res['super_trade'] = np.where(
        (res['market_direction'] == 1) &
        (res['ema_encoded_cot'] == 1) &
        (res['is_trade'] == 2),
        2,
        np.where(
            (res['market_direction'] == 0) &
            (res['ema_encoded_cot'] == 0) &
            (res['is_trade'] == 0),
            0,
            1
        )
    )

    return res
