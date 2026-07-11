# ==============================================================================
# 1. Library imports
# ==============================================================================
import os
import numpy as np
import pandas as pd
import random
import warnings
import re
import joblib
from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    precision_score, recall_score, accuracy_score
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from transformers import T5Tokenizer, T5EncoderModel, AutoConfig

# ==============================================================================
# 2. Global constants and configuration
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MY_SEED = 42

# ======================== Data paths ========================
PATH_4MER = '/root/autodl-tmp/NPZ/word2vec_4mer.npz'         
PATH_Y_TRAIN = '/root/autodl-tmp/y_train.csv'
PATH_Y_TEST = '/root/autodl-tmp/y_test.csv'
PATH_TRAIN_SEQ = "/root/autodl-tmp/train_sequences.csv"
PATH_TEST_SEQ = "/root/autodl-tmp/test_sequences.csv"

ANKH_PATH = '/root/autodl-tmp/Big_Model/Ankh3-XL'
ANKH3_EMBED_PREFIX = '[NLU]'

# ======================== Training and optimization hyperparameters ========================
BATCH_SIZE = 128
EPOCHS = 80
MIN_BATCH_SIZE = 2
NUM_CLASSES = 2

# [Optimizer parameters]
LR = 0.0001               
WEIGHT_DECAY = 1e-5        

# [Learning rate scheduler parameters]
ETA_MIN = 1e-6             

# [Loss function and anti-overfitting parameters]
DROPOUT = 0.1              
LABEL_SMOOTHING = 0.01     
CLIP_GRAD_NORM = 1.0       

# ======================== Branch 1: word2vec 4-mer ========================
INPUT_SHAPE_4MER = (97, 150)

# ======================== Branch 2: Ankh PLM ========================
MAX_SEQ_LENGTH = 100
PCA_TARGET_DIM = 512              
BRANCH2_INPUT_DIM = PCA_TARGET_DIM

# ======================== Shared CNN architecture for both branches ========================
ADAPTIVE_POOL_SIZE_PER_LAYER = [50, 25, 10, 5]

# ======================== Asymmetric branch architecture configuration ========================

FILTERS_4MER = [256, 512, 256, 256]
SK_KERNELS_4MER = [[1, 3,5], [1, 3, 5], [1, 3], [1, 3]]    # word2vec branch SKAttention kernel configuration

# Branch 2 (Ankh 512-dim): keep larger capacity to fully extract PLM semantics
FILTERS_ANKH = [512, 1024, 512, 512]
SK_KERNELS_ANKH = [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]] # Ankh branch SKAttention kernel configuration

MODEL_SELECT_METRIC = 'acc'       

# ==============================================================================
# 3. Utility functions
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def filter_warnings():
    warnings.filterwarnings("ignore", message="Unable to import Triton")
    warnings.filterwarnings("ignore", message="Increasing alibi size from")
    warnings.filterwarnings("ignore", message="You are using the default legacy behaviour")
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def clean_ankh_sequence(seq):
    return re.sub(r"[UZOB]", "X", str(seq).strip().upper())


def assert_ankh_token_alignment(clean_seq, tokenizer, prefix_token_ids, input_ids_row, attention_mask_row, sequence_index):
    seq_ids = tokenizer.encode(clean_seq, add_special_tokens=False)
    combined_ids = tokenizer.encode(f"{ANKH3_EMBED_PREFIX}{clean_seq}", add_special_tokens=False)
    prefix_token_count = len(prefix_token_ids)
    seq_end = prefix_token_count + len(seq_ids)
    valid_len = int(attention_mask_row.sum().item())
    sliced_ids = input_ids_row[prefix_token_count:seq_end].tolist()
    failed_checks = []
    if len(seq_ids) != len(clean_seq):
        failed_checks.append(f"residue_aligned token_count={len(seq_ids)} residue_count={len(clean_seq)}")
    if combined_ids != prefix_token_ids + seq_ids:
        failed_checks.append("prefix_plus_sequence_is_not_additive")
    if seq_end > valid_len:
        failed_checks.append(f"slice_exceeds_valid_len seq_end={seq_end} valid_len={valid_len}")
    if sliced_ids != seq_ids:
        failed_checks.append("current_slice_does_not_match_sequence_tokens")
    if failed_checks:
        raise ValueError(f"Ankh3 token alignment failed for seq {sequence_index}: {', '.join(failed_checks)}")
    return len(seq_ids)


def get_valid_token_lengths(sequences, max_seq_length):
    return np.array(
        [min(len(clean_ankh_sequence(seq)), max_seq_length) for seq in sequences],
        dtype=np.int32
    )


def collect_valid_token_rows(data_3d, lengths):
    valid_rows = []
    max_available_len = data_3d.shape[1]
    for sample_idx, valid_len in enumerate(lengths):
        valid_len = int(max(0, min(valid_len, max_available_len)))
        if valid_len > 0:
            valid_rows.append(data_3d[sample_idx, :valid_len, :])
    if not valid_rows:
        return np.empty((0, data_3d.shape[-1]), dtype=np.float32)
    return np.concatenate(valid_rows, axis=0).astype(np.float32)


def transform_valid_token_rows(data_3d, lengths, max_seq_length, scaler, pca):
    num_samples = data_3d.shape[0]
    max_available_len = data_3d.shape[1]
    output_dim = int(getattr(pca, 'n_components_', getattr(pca, 'n_components')))
    out = np.zeros((num_samples, max_seq_length, output_dim), dtype=np.float32)
    for sample_idx, valid_len in enumerate(lengths):
        valid_len = int(max(0, min(valid_len, max_seq_length, max_available_len)))
        if valid_len == 0:
            continue
        sample_valid_rows = data_3d[sample_idx, :valid_len, :]
        sample_valid_rows = scaler.transform(sample_valid_rows)
        sample_valid_rows = pca.transform(sample_valid_rows).astype(np.float32)
        out[sample_idx, :valid_len, :] = sample_valid_rows
    return out


def calculate_metrics_original(y_true, y_pred, is_prob=False):
    assert len(y_true) == len(y_pred)
    if not is_prob:
        y_pred_prob = torch.softmax(torch.from_numpy(y_pred), dim=-1).numpy()
    else:
        y_pred_prob = y_pred
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_prob, axis=1)
    y_pred_pos_prob = y_pred_prob[:, 1]
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    if cm.shape == (1, 1):
        tn, fp, fn, tp = (cm[0, 0], 0, 0, 0) if y_true_labels[0] == 0 else (0, 0, 0, cm[0, 0])
    else:
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    mcc = matthews_corrcoef(y_true_labels, y_pred_labels)
    auc_score = roc_auc_score(y_true_labels, y_pred_pos_prob) if len(np.unique(y_true_labels)) > 1 else 0.5
    precision = precision_score(y_true_labels, y_pred_labels, zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    return accuracy, mcc, auc_score, precision, recall, specificity, sensitivity


def apply_scaler_pca(train_3d, val_3d, max_seq_length, target_dim, seed,
                     train_lengths=None, val_lengths=None, scaler=None, pca=None):
    out_train, out_val = None, None
    if train_3d is not None:
        if train_lengths is None:
            raise ValueError("train_lengths must be provided")
        _, _, fused_dim = train_3d.shape
        train_2d = collect_valid_token_rows(train_3d, train_lengths)
        if train_2d.shape[0] == 0:
            raise ValueError("No valid tokens in the training set")
        if scaler is None:
            scaler = StandardScaler()
            train_2d_scaled = scaler.fit_transform(train_2d)
        else:
            train_2d_scaled = scaler.transform(train_2d)
        if pca is None:
            actual_target_dim = min(target_dim, fused_dim)
            pca = PCA(n_components=actual_target_dim, random_state=seed)
            pca.fit_transform(train_2d_scaled)
        else:
            pca.transform(train_2d_scaled)
        out_train = transform_valid_token_rows(train_3d, train_lengths, max_seq_length, scaler, pca)
    if val_3d is not None:
        if scaler is None or pca is None:
            raise ValueError("Fitted scaler and pca must be provided")
        if val_lengths is None:
            raise ValueError("val_lengths must be provided")
        out_val = transform_valid_token_rows(val_3d, val_lengths, max_seq_length, scaler, pca)
    return out_train, out_val, scaler, pca


def extract_ankh_features(sequences, max_seq_length, tokenizer, ankh_model, device):
    hidden_states_list = []
    ankh_model.eval()
    prefix_token_ids = tokenizer.encode(ANKH3_EMBED_PREFIX, add_special_tokens=False)
    prefix_token_count = len(prefix_token_ids)
    if prefix_token_count <= 0:
        raise ValueError("Ankh3 prefix tokenization failed.")
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i:i + BATCH_SIZE].tolist()
            clean_batch_seq = [clean_ankh_sequence(seq) for seq in batch_seq]
            sequence_examples = [f"{ANKH3_EMBED_PREFIX}{seq}" for seq in clean_batch_seq]
            ids = tokenizer.batch_encode_plus(
                sequence_examples, add_special_tokens=True, is_split_into_words=False,
                padding="longest", return_tensors="pt"
            )
            input_ids = ids['input_ids'].to(device)
            attention_mask = ids['attention_mask'].to(device)
            outputs = ankh_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            batch_hidden = []
            for idx, clean_seq in enumerate(clean_batch_seq):
                residue_token_count = assert_ankh_token_alignment(
                    clean_seq, tokenizer, prefix_token_ids,
                    ids['input_ids'][idx], ids['attention_mask'][idx], i + idx
                )
                seq_start = prefix_token_count
                seq_end = seq_start + residue_token_count
                valid_len = int(attention_mask[idx].sum().item())
                if seq_end > valid_len:
                    raise ValueError(f"Ankh3 slicing exceeded valid len for seq {i + idx}")
                seq_emb = last_hidden_state[idx, seq_start:seq_end]
                if seq_emb.shape[0] != len(clean_seq):
                    raise ValueError(f"Ankh3 emb length mismatch for seq {i + idx}")
                if seq_emb.shape[0] > max_seq_length:
                    seq_emb = seq_emb[:max_seq_length, :]
                elif seq_emb.shape[0] < max_seq_length:
                    pad_len = max_seq_length - seq_emb.shape[0]
                    seq_emb = F.pad(seq_emb, (0, 0, 0, pad_len))
                batch_hidden.append(seq_emb.unsqueeze(0))
            batch_hidden = torch.cat(batch_hidden, dim=0)
            hidden_states_list.append(batch_hidden.cpu())
    all_hidden = torch.cat(hidden_states_list, dim=0)
    return all_hidden.numpy()


def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    if 'sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain the 'Sequence' and 'label' columns")
    df = df[['sequence', 'label']].copy()
    df['sequence'] = df['sequence'].str.strip().str.upper()
    df['label'] = df['label'].astype(int)
    df = df[df['sequence'].str.len() > 0]
    df = df[df['label'].isin([0, 1])]
    df = df.reset_index(drop=True)
    return df


# ==============================================================================
# 4. Model definition (asymmetric dual-branch architecture)
# ==============================================================================
class ECA1dAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y).permute(0, 2, 1)
        return x * y


class Conv1dLayerNormBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dropout=0.):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x


class SKAttention1D(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            if k % 2 == 1:
                pad_layer = nn.Identity()
                conv_padding = (k - 1) // 2
            else:
                pad_left = k // 2
                pad_right = k // 2 - 1
                pad_layer = nn.ConstantPad1d((pad_left, pad_right), 0)
                conv_padding = 0
            self.convs.append(nn.Sequential(OrderedDict([
                ('pad', pad_layer),
                ('conv', nn.Conv1d(channel, channel, kernel_size=k, padding=conv_padding, groups=group)),
                ('block', Conv1dLayerNormBlock(channel, channel, kernel_size=1, padding=0)),
                ('relu', nn.ReLU(inplace=True))
            ])))
        self.fc = nn.Sequential(
            nn.Linear(channel, self.d, bias=False),
            nn.BatchNorm1d(self.d),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, seq_len = x.size()
        conv_outs = [conv(x) for conv in self.convs]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = U.mean(-1)
        Z = self.fc(S)
        weights = [fc(Z).view(bs, c, 1) for fc in self.fcs]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V


class SKResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, eca_kernel_size=3,
                 reduction=16, group=1, L=32, adaptive_pool_size=25, dropout=0.2):
        super().__init__()
        self.sk_attention = SKAttention1D(channel=in_channels, kernels=kernels,
                                          reduction=reduction, group=group, L=L)
        self.eca_attention = ECA1dAttention(kernel_size=eca_kernel_size)
        self.channel_map = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(output_size=adaptive_pool_size)
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual.add_module('res_conv', nn.Conv1d(in_channels, out_channels, 1))
        self.residual.add_module('res_pool', nn.AdaptiveMaxPool1d(output_size=adaptive_pool_size))

    def forward(self, x):
        residual = x
        out = self.sk_attention(x)
        out = self.eca_attention(out)
        out = self.channel_map(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.pool(out)
        residual = self.residual(residual)
        out = out + residual
        out = self.relu(out)
        return out


class ModelConfig:
    
    def __init__(self, device, max_time_steps, input_size, num_classes, filters=None, sk_kernels=None):
        self.device = device
        self.max_time_steps = max_time_steps
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = DROPOUT
        self.num_cnn_layers = 4
        
        self.sk_kernels_per_layer = sk_kernels if sk_kernels is not None else [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.sk_reduction = 16
        self.sk_group = 1
        self.sk_L = 32
        self.filters = filters if filters is not None else [1024, 2048, 1024, 1024]
        self.adaptive_pool_sizes = ADAPTIVE_POOL_SIZE_PER_LAYER
        self.eca_kernel_size = 3


class ProtBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        in_channels = config.input_size
        for i in range(config.num_cnn_layers):
            self.conv_blocks.append(
                SKResidualBlock1D(
                    in_channels=in_channels,
                    out_channels=config.filters[i],
                    kernels=config.sk_kernels_per_layer[i],
                    eca_kernel_size=config.eca_kernel_size,
                    reduction=config.sk_reduction,
                    group=config.sk_group,
                    L=config.sk_L,
                    adaptive_pool_size=config.adaptive_pool_sizes[i],
                    dropout=config.dropout
                )
            )
            in_channels = config.filters[i]
        self.final_dim = config.filters[-1] * config.adaptive_pool_sizes[-1]
        self.fc_block = nn.Sequential(
            nn.Linear(self.final_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for block in self.conv_blocks:
            x = block(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_block(x)
        return x


class TwoBranchCombinedModel(nn.Module):
    def __init__(self, model_4mer, model_prot):
        super().__init__()
        self.model_4mer = model_4mer
        self.model_prot = model_prot
        self.branch_out_dim = 32
        self.combined_layers = nn.Sequential(
            nn.Linear(self.branch_out_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),       
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x4, xprot):
        out4 = self.model_4mer(x4)
        outprot = self.model_prot(xprot)
        combined = torch.cat([out4, outprot], dim=-1)
        out = self.combined_layers(combined)
        return out


# ==============================================================================
# 5. Data loading and preprocessing
# ==============================================================================
def load_all_data():
    # --- Branch 1: word2vec 4-mer ---
    data_4mer = np.load(PATH_4MER)
    x_train_4mer = data_4mer['x_train'].astype(np.float32)
    x_test_4mer = data_4mer['x_test'].astype(np.float32)
    print(f"Branch 1 word2vec: train={x_train_4mer.shape}, test={x_test_4mer.shape}")

    # --- Labels ---
    y_train_onehot = pd.read_csv(PATH_Y_TRAIN).to_numpy()
    y_train_onehot = np.eye(2)[y_train_onehot.astype(np.int32).reshape(-1)]
    y_train_label = np.argmax(y_train_onehot, axis=1)
    y_test_onehot = pd.read_csv(PATH_Y_TEST).to_numpy()
    y_test_onehot = np.eye(2)[y_test_onehot.astype(np.int32).reshape(-1)]
    y_test_label = np.argmax(y_test_onehot, axis=1)

    # --- Ankh PLM ---
    train_seq_df = read_csv_data(PATH_TRAIN_SEQ)
    test_seq_df = read_csv_data(PATH_TEST_SEQ)
    train_seqs = train_seq_df['sequence'].values
    test_seqs = test_seq_df['sequence'].values
    train_token_lengths = get_valid_token_lengths(train_seqs, MAX_SEQ_LENGTH)
    test_token_lengths = get_valid_token_lengths(test_seqs, MAX_SEQ_LENGTH)

    print(f"\nLoading large language model: {ANKH_PATH}")
    tokenizer = T5Tokenizer.from_pretrained(ANKH_PATH)
    ankh_config = AutoConfig.from_pretrained(f"{ANKH_PATH}/config.json")
    ANKH_HIDDEN_DIM = getattr(ankh_config, 'd_model', getattr(ankh_config, 'hidden_size', 768))
    print(f"Ankh hidden layer dimension: {ANKH_HIDDEN_DIM}")
    ankh_model = T5EncoderModel.from_pretrained(ANKH_PATH, config=ankh_config).to(DEVICE)

    print("Extracting Ankh features...")
    X_train_plm = extract_ankh_features(train_seqs, MAX_SEQ_LENGTH, tokenizer, ankh_model, DEVICE)
    X_test_plm = extract_ankh_features(test_seqs, MAX_SEQ_LENGTH, tokenizer, ankh_model, DEVICE)

    # Validation checks
    assert len(x_train_4mer) == len(X_train_plm) == len(y_train_label)
    assert len(x_test_4mer) == len(X_test_plm) == len(y_test_label)

    print(f"\nFusion strategy: Ankh PCA({PCA_TARGET_DIM}-dim)")

    return (
        x_train_4mer, X_train_plm,
        train_token_lengths, y_train_onehot, y_train_label,
        x_test_4mer, X_test_plm,
        test_token_lengths, y_test_onehot, y_test_label
    )


# ==============================================================================
# 6. Training and evaluation
# ==============================================================================
def main():
    set_seed(MY_SEED)
    filter_warnings()

    (
        x_train_4mer, X_train_plm,
        train_token_lengths, y_train_onehot, y_train_label,
        x_test_4mer, X_test_plm,
        test_token_lengths, y_test_onehot, y_test_label
    ) = load_all_data()

    
    config_4mer = ModelConfig(DEVICE, INPUT_SHAPE_4MER[0], INPUT_SHAPE_4MER[1], NUM_CLASSES,
                              filters=FILTERS_4MER, sk_kernels=SK_KERNELS_4MER)
  
    config_prot = ModelConfig(DEVICE, MAX_SEQ_LENGTH, BRANCH2_INPUT_DIM, NUM_CLASSES,
                              filters=FILTERS_ANKH, sk_kernels=SK_KERNELS_ANKH)

    print(f"\nBranch 1 (word2vec) filters: {FILTERS_4MER}")
    print(f"Branch 1 (word2vec) SK kernels: {SK_KERNELS_4MER}")
    print(f"Branch 2 (Ankh)     filters: {FILTERS_ANKH}")
    print(f"Branch 2 (Ankh)     SK kernels: {SK_KERNELS_ANKH}")

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=MY_SEED)
    metrics_list = ['accuracies', 'mccs', 'aucs', 'precisions', 'recalls', 'specificities', 'sensitivities']
    metrics_dict = {k: [] for k in metrics_list}

    fold_no = 1
    for train_idx, val_idx in kfold.split(x_train_4mer, y_train_label):
        print(f"\n===================== Fold {fold_no} =====================")

        if len(train_idx) < MIN_BATCH_SIZE * 2 or len(val_idx) < MIN_BATCH_SIZE:
            fold_no += 1
            continue

        # --- Split ---
        x_tr_4mer, x_val_4mer = x_train_4mer[train_idx], x_train_4mer[val_idx]
        x_tr_plm, x_val_plm = X_train_plm[train_idx], X_train_plm[val_idx]
        x_tr_lengths, x_val_lengths = train_token_lengths[train_idx], train_token_lengths[val_idx]
        y_tr_onehot, y_val_onehot = y_train_onehot[train_idx], y_train_onehot[val_idx]
        y_tr_label, y_val_label = y_train_label[train_idx], y_train_label[val_idx]

        # --- Ankh PCA dimensionality reduction ---
        print(f"Fold {fold_no}: Ankh PCA dimensionality reduction ({X_train_plm.shape[-1]}→{PCA_TARGET_DIM})...")
        x_tr_pca, x_val_pca, fold_scaler, fold_pca = apply_scaler_pca(
            train_3d=x_tr_plm, val_3d=x_val_plm,
            max_seq_length=MAX_SEQ_LENGTH, target_dim=PCA_TARGET_DIM, seed=MY_SEED,
            train_lengths=x_tr_lengths, val_lengths=x_val_lengths
        )

       
        x_tr_prot = x_tr_pca
        x_val_prot = x_val_pca

       
        tr_dataset = TensorDataset(
            torch.from_numpy(x_tr_4mer),
            torch.from_numpy(x_tr_prot),
            torch.from_numpy(y_tr_label).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(x_val_4mer),
            torch.from_numpy(x_val_prot),
            torch.from_numpy(y_val_label).long()
        )
        tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


        model_4mer = ProtBertModel(config_4mer).to(DEVICE)
        model_prot = ProtBertModel(config_prot).to(DEVICE)
        model = TwoBranchCombinedModel(model_4mer, model_prot).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=ETA_MIN
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

        best_fold_val_metric = -1.0
        best_fold_metrics = None
        fold_model_path = f"best_model_fold_{fold_no}.pth"
        fold_scaler_path = f"best_scaler_fold_{fold_no}.pkl"
        fold_pca_path = f"best_pca_fold_{fold_no}.pkl"

        for epoch in range(EPOCHS):
            model.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch in tr_loader:
                x4, xprot, y = batch
                if x4.size(0) < MIN_BATCH_SIZE:
                    continue
                x4, xprot, y = x4.to(DEVICE), xprot.to(DEVICE), y.to(DEVICE)
                outputs = model(x4, xprot)
                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
                optimizer.step()

                total_loss += loss.item() * x4.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            if total == 0:
                continue

            tr_loss = total_loss / total
            tr_acc = correct / total
            scheduler.step()

            # --- Validation ---
            model.eval()
            y_val_pred = []
            with torch.no_grad():
                for batch in val_loader:
                    x4, xprot, _ = batch
                    x4, xprot = x4.to(DEVICE), xprot.to(DEVICE)
                    outputs = model(x4, xprot)
                    y_val_pred.append(outputs.cpu().numpy())
            if not y_val_pred:
                continue

            y_val_pred = np.concatenate(y_val_pred, axis=0)
            metrics = calculate_metrics_original(y_val_onehot, y_val_pred)
            accuracy, mcc, auc, precision, recall, specificity, sensitivity = metrics

            print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
                  f"Val AUC: {auc:.4f}, Val Acc: {accuracy:.4f}")

            current_metric = auc if MODEL_SELECT_METRIC == 'auc' else accuracy
            if current_metric > best_fold_val_metric:
                best_fold_val_metric = current_metric
                best_fold_metrics = metrics
                torch.save(model.state_dict(), fold_model_path)
                joblib.dump(fold_scaler, fold_scaler_path)
                joblib.dump(fold_pca, fold_pca_path)
                print(f"  --> [Fold {fold_no} best] Val {MODEL_SELECT_METRIC.upper()} improved, saved.")

        if best_fold_metrics is not None:
            accuracy, mcc, auc, precision, recall, specificity, sensitivity = best_fold_metrics
            metrics_dict['accuracies'].append(accuracy)
            metrics_dict['mccs'].append(mcc)
            metrics_dict['aucs'].append(auc)
            metrics_dict['precisions'].append(precision)
            metrics_dict['recalls'].append(recall)
            metrics_dict['specificities'].append(specificity)
            metrics_dict['sensitivities'].append(sensitivity)
            print(f"\nFold {fold_no} best (AUC: {auc:.4f}, Acc: {accuracy:.4f})")
        fold_no += 1

    print("\n===================== Cross Validation Average =====================")
    for name, vals in metrics_dict.items():
        if vals:
            print(f"Average {name.capitalize()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # ==============================================================================
    # Test set evaluation
    # ==============================================================================
    print("\n===================== Test Set Evaluation =====================")

    all_fold_probs = []
    for fold_idx in range(1, 6):
        f_model_path = f"best_model_fold_{fold_idx}.pth"
        f_scaler_path = f"best_scaler_fold_{fold_idx}.pkl"
        f_pca_path = f"best_pca_fold_{fold_idx}.pkl"

        if not (os.path.exists(f_model_path) and os.path.exists(f_scaler_path)):
            print(f"Warning: Fold {fold_idx} weights are missing, skipped.")
            continue

        print(f"Loading Fold {fold_idx} ...")

        f_scaler = joblib.load(f_scaler_path)
        f_pca = joblib.load(f_pca_path)
        _, X_test_pca_fold, _, _ = apply_scaler_pca(
            train_3d=None, val_3d=X_test_plm,
            max_seq_length=MAX_SEQ_LENGTH, target_dim=PCA_TARGET_DIM, seed=MY_SEED,
            val_lengths=test_token_lengths, scaler=f_scaler, pca=f_pca
        )

        X_test_prot_fold = X_test_pca_fold

        test_dataset = TensorDataset(
            torch.from_numpy(x_test_4mer),
            torch.from_numpy(X_test_prot_fold),
            torch.from_numpy(y_test_label).long()
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        model_4mer = ProtBertModel(config_4mer).to(DEVICE)
        model_prot = ProtBertModel(config_prot).to(DEVICE)
        model_fold = TwoBranchCombinedModel(model_4mer, model_prot).to(DEVICE)
        model_fold.load_state_dict(torch.load(f_model_path, map_location=DEVICE))
        model_fold.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                x4, xprot, _ = batch
                x4, xprot = x4.to(DEVICE), xprot.to(DEVICE)
                outputs = model_fold(x4, xprot)
                fold_preds.append(outputs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        fold_probs = torch.softmax(torch.from_numpy(fold_preds), dim=-1).numpy()
        all_fold_probs.append(fold_probs)

    if not all_fold_probs:
        print("Ensemble failed")
        return

    ensemble_probs = np.mean(all_fold_probs, axis=0)
    nll_loss = nn.NLLLoss()
    ensemble_log_probs = torch.log(torch.from_numpy(ensemble_probs) + 1e-8)
    test_loss = nll_loss(ensemble_log_probs, torch.from_numpy(y_test_label).long()).item()

    test_metrics = calculate_metrics_original(y_test_onehot, ensemble_probs, is_prob=True)
    test_acc, test_mcc, test_auc, test_prec, test_rec, test_spec, test_sens = test_metrics

    print(f"\n[Final ensemble evaluation results]")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}")
    print(f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}")
    print(f"Test Specificity: {test_spec:.4f}, Test Sensitivity: {test_sens:.4f}")


if __name__ == "__main__":
    main()