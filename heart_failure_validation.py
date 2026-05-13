# External validation on Heart Failure dataset
# Classifiers: DT, XGBoost, META-DES, FIRE-META-DES
# Imbalance handling: none, smote, borderline1, borderline2, smoteenn, class_weight
# Validation: 80/20 stratified split + 10x5 repeated stratified cross-validation on training set
# Preprocessing: Min-Max normalization

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from deslib.des import METADES
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Heart Failure.csv', encoding='gbk', index_col=0)
X = df.drop(columns=['label'])
y = df['label']

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Imbalance ratio: {np.bincount(y)[1]/np.bincount(y)[0]:.2f}")

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train_full.shape[0]}, Test samples: {X_test_full.shape[0]}")

# Resampling utilities
def apply_smoteenn(X, y, random_state=42):
    smote = SMOTE(k_neighbors=7, random_state=random_state)
    enn = EditedNearestNeighbours(kind_sel='mode', n_neighbors=3)
    smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=random_state)
    return smote_enn.fit_resample(X, y)

def apply_resampling(X, y, method, random_state=42):
    if method == 'none':
        return X, y
    elif method == 'smote':
        return SMOTE(random_state=random_state).fit_resample(X, y)
    elif method == 'borderline1':
        return BorderlineSMOTE(kind='borderline-1', random_state=random_state).fit_resample(X, y)
    elif method == 'borderline2':
        return BorderlineSMOTE(kind='borderline-2', random_state=random_state).fit_resample(X, y)
    elif method == 'smoteenn':
        return apply_smoteenn(X, y, random_state)
    else:
        raise ValueError(f"Unknown method: {method}")

# FIRE-META-DES classifier
class FireMETA_DES:
    def __init__(self, k=9, dsel_ratio=0.3, acc_threshold=0.6, voting='soft', random_state=42):
        self.k = k
        self.dsel_ratio = dsel_ratio
        self.acc_threshold = acc_threshold
        self.voting = voting
        self.random_state = random_state
        self.pool = None
        self.DSEL_X = None
        self.DSEL_y = None
        self.fitted = False

    def _build_pool(self):
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=self.random_state)
        svm = SVC(probability=True, C=1.0, gamma='scale', class_weight='balanced', random_state=self.random_state)
        knn = KNeighborsClassifier(n_neighbors=5)
        dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=self.random_state)
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=self.random_state)
        xgb1 = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                 random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        xgb2 = xgb.XGBClassifier(n_estimators=30, max_depth=5, learning_rate=0.05,
                                 random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
        lgbm = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                                  random_state=self.random_state, verbose=-1)
        cat = CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1,
                                 random_seed=self.random_state, verbose=False)
        hgb = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=self.random_state)
        return [('lr', lr), ('svm', svm), ('knn', knn), ('dt', dt), ('rf', rf),
                ('xgb1', xgb1), ('xgb2', xgb2), ('lgbm', lgbm), ('cat', cat), ('hgb', hgb)]

    def fit(self, X, y):
        X_train_base, X_dsel, y_train_base, y_dsel = train_test_split(
            X, y, test_size=self.dsel_ratio, stratify=y, random_state=self.random_state
        )
        self.DSEL_X = X_dsel
        self.DSEL_y = y_dsel
        self.pool = self._build_pool()
        for name, clf in self.pool:
            clf.fit(X_train_base, y_train_base)
        self.fitted = True
        return self

    def _get_cr(self, x):
        dist = np.linalg.norm(self.DSEL_X - x, axis=1)
        neighbor_idx = np.argsort(dist)[:self.k]
        return neighbor_idx, self.DSEL_y[neighbor_idx]

    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted!")
        probas = []
        for x in X:
            neighbor_idx, y_local = self._get_cr(x)
            if len(np.unique(y_local)) > 1:
                selected = []
                for name, clf in self.pool:
                    cr_preds = clf.predict(self.DSEL_X[neighbor_idx])
                    acc_local = np.mean(cr_preds == y_local)
                    if acc_local >= self.acc_threshold:
                        selected.append(clf)
                if len(selected) == 0:
                    selected = [clf for _, clf in self.pool]
                if self.voting == 'hard':
                    votes = [clf.predict([x])[0] for clf in selected]
                    prob_pos = np.mean(votes)
                else:
                    prob_pos = np.mean([clf.predict_proba([x])[0][1] for clf in selected])
            else:
                if self.voting == 'hard':
                    votes = [clf.predict([x])[0] for _, clf in self.pool]
                    prob_pos = np.mean(votes)
                else:
                    prob_pos = np.mean([clf.predict_proba([x])[0][1] for _, clf in self.pool])
            probas.append([1 - prob_pos, prob_pos])
        return np.array(probas)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# META-DES builder
def build_meta_des(X_base, y_base, X_dsel, y_dsel, method, random_state=42):
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=random_state)
    svm = SVC(probability=True, C=1.0, gamma='scale', class_weight='balanced', random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=random_state)
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=random_state)
    xgb1 = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                             random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    xgb2 = xgb.XGBClassifier(n_estimators=30, max_depth=5, learning_rate=0.05,
                             random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    lgbm = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1,
                              random_state=random_state, verbose=-1)
    cat = CatBoostClassifier(iterations=50, depth=3, learning_rate=0.1,
                             random_seed=random_state, verbose=False)
    hgb = HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=random_state)
    pool = [lr, svm, knn, dt, rf, xgb1, xgb2, lgbm, cat, hgb]

    for clf in pool:
        clf.fit(X_base, y_base)

    if len(np.unique(y_dsel)) < 2:
        warnings.warn("DSEL has only one class, falling back to VotingClassifier")
        estimators = [(f'clf_{i}', clf) for i, clf in enumerate(pool)]
        voting = VotingClassifier(estimators=estimators)
        voting.fit(X_base, y_base)
        return voting

    meta = METADES(pool_classifiers=pool, random_state=random_state, k=7, selection_threshold=0.5)
    meta.fit(X_dsel, y_dsel)
    return meta

def get_classifier(clf_name, method, random_state=42, pos_scale=None):
    if clf_name == 'DT':
        clf = DecisionTreeClassifier(random_state=random_state)
        if method == 'class_weight':
            clf.set_params(class_weight='balanced')
        return clf
    elif clf_name == 'XGB':
        clf = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        if method == 'class_weight' and pos_scale is not None:
            clf.set_params(scale_pos_weight=pos_scale)
        return clf
    elif clf_name == 'META-DES':
        return 'META-DES'
    elif clf_name == 'FIRE-META-DES':
        return 'FIRE-META-DES'
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")

# 10x5 cross-validation
n_repeats = 10
n_splits = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

balance_methods = ['none', 'smote', 'borderline1', 'borderline2', 'smoteenn', 'class_weight']
classifier_names = ['DT', 'XGB', 'META-DES', 'FIRE-META-DES']

results = {m: {c: {'acc': [], 'prec': [], 'rec': [], 'auc': [], 'f1': [], 'brier': []}
               for c in classifier_names} for m in balance_methods}

for method in balance_methods:
    print(f"\n===== Imbalance method: {method} =====")
    for clf_name in classifier_names:
        print(f"  Classifier: {clf_name}")
        tmp_metrics = {k: [] for k in ['acc','prec','rec','auc','f1','brier']}

        for train_idx, val_idx in rskf.split(X_train_full, y_train_full):
            X_train_fold = X_train_full.iloc[train_idx].values
            y_train_fold = y_train_full.iloc[train_idx].values
            X_val_fold = X_train_full.iloc[val_idx].values
            y_val_fold = y_train_full.iloc[val_idx].values

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)

            if method == 'class_weight':
                X_res, y_res = X_train_scaled, y_train_fold
                neg, pos = np.bincount(y_res)
                scale_pos = neg / pos if pos > 0 else 1.0
            else:
                X_res, y_res = apply_resampling(X_train_scaled, y_train_fold, method)
                scale_pos = None

            if clf_name == 'META-DES':
                X_base, X_dsel, y_base, y_dsel = train_test_split(
                    X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
                )
            else:
                X_base, y_base = X_res, y_res
                X_dsel, y_dsel = None, None

            if clf_name == 'DT':
                clf = get_classifier('DT', method, random_state=42)
                clf.fit(X_base, y_base)
                y_pred = clf.predict(X_val_scaled)
                y_proba = clf.predict_proba(X_val_scaled)[:, 1]
            elif clf_name == 'XGB':
                clf = get_classifier('XGB', method, random_state=42, pos_scale=scale_pos)
                clf.fit(X_base, y_base)
                y_pred = clf.predict(X_val_scaled)
                y_proba = clf.predict_proba(X_val_scaled)[:, 1]
            elif clf_name == 'META-DES':
                meta = build_meta_des(X_base, y_base, X_dsel, y_dsel, method, random_state=42)
                y_pred = meta.predict(X_val_scaled)
                y_proba = meta.predict_proba(X_val_scaled)[:, 1]
            elif clf_name == 'FIRE-META-DES':
                fire = FireMETA_DES(k=9, dsel_ratio=0.3,
                                    acc_threshold=0.6, voting='soft', random_state=42)
                fire.fit(X_res, y_res)
                y_pred = fire.predict(X_val_scaled)
                y_proba = fire.predict_proba(X_val_scaled)[:, 1]

            acc = accuracy_score(y_val_fold, y_pred)
            prec = precision_score(y_val_fold, y_pred, zero_division=0)
            rec = recall_score(y_val_fold, y_pred, zero_division=0)
            auc = roc_auc_score(y_val_fold, y_proba)
            f1 = f1_score(y_val_fold, y_pred, zero_division=0)
            brier = brier_score_loss(y_val_fold, y_proba)

            for mkey in tmp_metrics:
                tmp_metrics[mkey].append(locals()[mkey])

        for mkey in tmp_metrics:
            results[method][clf_name][mkey] = np.mean(tmp_metrics[mkey])
        print(f"    Done, ACC={results[method][clf_name]['acc']:.4f}, AUC={results[method][clf_name]['auc']:.4f}")

# Output results
summary_rows = []
for method in balance_methods:
    for clf in classifier_names:
        row = {
            'Imbalance': method,
            'Classifier': clf,
            'ACC': results[method][clf]['acc'],
            'Precision': results[method][clf]['prec'],
            'Recall': results[method][clf]['rec'],
            'AUC': results[method][clf]['auc'],
            'F1': results[method][clf]['f1'],
            'Brier': results[method][clf]['brier']
        }
        summary_rows.append(row)

df_results = pd.DataFrame(summary_rows)
print("\n========== CV results (mean over 10x5 folds) ==========")
print(df_results.to_string(index=False))
