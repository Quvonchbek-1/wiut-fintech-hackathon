import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

print("1. Ma'lumotlar yuklanmoqda...")
train_signals = pd.read_csv('train_signals.csv') 
train_transactions = pd.read_parquet('train_transactions.parquet')
test_signals = pd.read_csv('test_signals.csv')
test_transactions = pd.read_parquet('test_transactions.parquet')

print("2. Xususiyatlar yaratilmoqda...")
def extract_features(signals_df, transactions_df):
    agg_num = transactions_df.groupby('signal_id')['miqdor_indeksi'].agg(
        ['sum', 'mean', 'max', 'min', 'std', 'count']
    ).reset_index()
    
    turi_pivot = transactions_df.pivot_table(
        index='signal_id', 
        columns='tranzaksiya_turi', 
        values='miqdor_indeksi', 
        aggfunc=['count', 'sum'], 
        fill_value=0
    )
    turi_pivot.columns = [f"turi_{col[0]}_{col[1]}" for col in turi_pivot.columns]
    turi_pivot.reset_index(inplace=True)
    
    kc_pivot = transactions_df.pivot_table(
        index='signal_id', 
        columns='kirim_chiqim', 
        values='miqdor_indeksi', 
        aggfunc=['count', 'sum'], 
        fill_value=0
    )
    kc_pivot.columns = [f"kc_{col[0]}_{col[1]}" for col in kc_pivot.columns]
    kc_pivot.reset_index(inplace=True)
    
    features = signals_df[['signal_id']].merge(agg_num, on='signal_id', how='left')
    features = features.merge(turi_pivot, on='signal_id', how='left')
    features = features.merge(kc_pivot, on='signal_id', how='left')
    
    return features.fillna(0)

train_features = extract_features(train_signals, train_transactions)
test_features = extract_features(test_signals, test_transactions)

train_data = train_signals.merge(train_features, on='signal_id', how='left')
test_data = test_signals.merge(test_features, on='signal_id', how='left')

features_cols = [c for c in train_data.columns if c not in ['signal_id', 'signal_sanasi', 'eskalatsiya']]
X = train_data[features_cols]
y = train_data['eskalatsiya']
X_test = test_data[features_cols]

print("3. Model o'qitilmoqda...")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_data))
test_preds = np.zeros(len(test_data))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.03,
        random_state=42,
        verbose=-1
    )
    model.fit(X_tr, y_tr)
    
    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / n_splits

print(f"Modelning Validatsiya natijasi (ROC-AUC): {roc_auc_score(y, oof_preds):.5f}")

print("4. Bashorat fayli yaratilmoqda...")
submission = pd.DataFrame({
    'signal_id': test_data['signal_id'],
    'ehtimollik': test_preds
})

submission.to_csv('team_C9B71210.csv', index=False)
print("Tayyor! Papkangizda 'team_C9B71210.csv' fayli hosil bo'ldi.")

