# External validation on Hepatitis dataset
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN
from deslib.des import METADES
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Hepatitis.csv', encoding='gbk', index_col=0)
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
        return SMOTEENN(random_state=random_state).fit_resample(X, y)
    else:
        raise ValueError(f"Unknown method: {method}")

def preprocess_with_scaling(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Base classifier pool (7 classifiers)
def create_pool(y_base=None, random_state=42):
    neg, pos = np.bincount(y_base) if y_base is not None else (0, 0)
    if pos > 0:
        scale_pos_weight = neg / pos
        class_weight = {0:1, 1:scale_pos_weight}
    else:
        scale_pos_weight = 1.0
        class_weight = 'balanced'

    lr = LogisticRegression(penalty='l1', solver='saga', max_iter=1000,
                            class_weight='balanced', random_state=random_state)
    svm = SVC(probability=True, C=0.8, gamma='scale',
              class_weight='balanced', random_state=random_state)
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
    dt = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=random_state)
    rf = RandomForestClassifier(n_estimators=80, max_depth=5,
                                class_weight='balanced', random_state=random_state)
    xgb_clf = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.1,
                                scale_pos_weight=scale_pos_weight,
                                random_state=random_state, use_label_encoder=False,
                                eval_metric='logloss')
    gnb = GaussianNB()
    pool = [lr, svm, knn, dt, rf, xgb_clf, gnb]
    return pool, scale_pos_weight

# FIRE-META-DES classifier
class FireMETA_DES:
    def __init__(self, k=5, dsel_ratio=0.3, acc_threshold=0.7,
                 voting='soft', use_weighted_voting=True, random_state=42):
        self.k = k
        self.dsel_ratio = dsel_ratio
        self.acc_threshold = acc_threshold
        self.voting = voting
        self.use_weighted_voting = use_weighted_voting
        self.random_state = random_state
        self.pool = None
        self.DSEL_X = None
        self.DSEL_y = None
        self.fitted = False

    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        X_train_base, X_dsel, y_train_base, y_dsel = train_test_split(
            X, y, test_size=self.dsel_ratio, stratify=y, random_state=self.random_state
        )
        self.DSEL_X = X_dsel
        self.DSEL_y = y_dsel
        self.pool, _ = create_pool(y_base=y_train_base, random_state=self.random_state)
        for clf in self.pool:
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
        if hasattr(X, 'values'):
            X = X.values

        probas = []
        for x in X:
            neighbor_idx, y_local = self._get_cr(x)
            local_accs = []
            for clf in self.pool:
                preds = clf.predict(self.DSEL_X[neighbor_idx])
                acc = np.mean(preds == y_local)
                local_accs.append(acc)

            if len(np.unique(y_local)) == 1:
                selected_indices = list(range(len(self.pool)))
            else:
                selected_indices = [i for i, acc in enumerate(local_accs) if acc >= self.acc_threshold]
                if len(selected_indices) == 0:
                    selected_indices = list(range(len(self.pool)))

            if self.use_weighted_voting:
                weights = [local_accs[i] for i in selected_indices]
                if np.sum(weights) == 0:
                    weights = np.ones(len(selected_indices))
                weights = np.array(weights) / np.sum(weights)
            else:
                weights = np.ones(len(selected_indices)) / len(selected_indices)

            if self.voting == 'hard':
                votes = [self.pool[i].predict([x])[0] for i in selected_indices]
                prob_pos = np.average(votes, weights=weights)
            else:
                proba_list = [self.pool[i].predict_proba([x])[0][1] for i in selected_indices]
                prob_pos = np.average(proba_list, weights=weights)

            probas.append([1 - prob_pos, prob_pos])
        return np.array(probas)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

# META-DES builder
def build_meta_des(X_base, y_base, X_dsel, y_dsel, random_state=42):
    pool, _ = create_pool(y_base=y_base, random_state=random_state)
    for clf in pool:
        clf.fit(X_base, y_base)

    unique, counts = np.unique(y_dsel, return_counts=True)
    if len(unique) < 2 or np.min(counts) < 3:
        warnings.warn("Invalid DSEL, falling back to VotingClassifier")
        estimators = [(f'clf_{i}', clf) for i, clf in enumerate(pool)]
        voting = VotingClassifier(estimators=estimators, voting='soft')
        voting.fit(X_base, y_base)
        return voting

    meta = METADES(pool_classifiers=pool, random_state=random_state, k=7, selection_threshold=0.5)
    meta.fit(X_dsel, y_dsel)
    if hasattr(meta, 'meta_classifier_') and meta.meta_classifier_.classes_.size < 2:
        warnings.warn("Meta-classifier degenerate, falling back to VotingClassifier")
        estimators = [(f'clf_{i}', clf) for i, clf in enumerate(pool)]
        voting = VotingClassifier(estimators=estimators, voting='soft')
        voting.fit(X_base, y_base)
        return voting
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

            X_train_scaled, X_val_scaled = preprocess_with_scaling(X_train_fold, X_val_fold)

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

            try:
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
                    meta = build_meta_des(X_base, y_base, X_dsel, y_dsel, random_state=42)
                    y_pred = meta.predict(X_val_scaled)
                    y_proba = meta.predict_proba(X_val_scaled)[:, 1]
                elif clf_name == 'FIRE-META-DES':
                    fire = FireMETA_DES(k=5, dsel_ratio=0.3,
                                        acc_threshold=0.7, voting='soft',
                                        use_weighted_voting=True, random_state=42)
                    fire.fit(X_res, y_res)
                    y_pred = fire.predict(X_val_scaled)
                    y_proba = fire.predict_proba(X_val_scaled)[:, 1]
            except Exception as e:
                warnings.warn(f"{clf_name} training failed ({e}), falling back to VotingClassifier")
                fallback = VotingClassifier(estimators=[
                    ('dt', DecisionTreeClassifier(class_weight='balanced', random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=80, class_weight='balanced', random_state=42))
                ], voting='soft')
                fallback.fit(X_base, y_base)
                y_pred = fallback.predict(X_val_scaled)
                y_proba = fallback.predict_proba(X_val_scaled)[:, 1]

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
