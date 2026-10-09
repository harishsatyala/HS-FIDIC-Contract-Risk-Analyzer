"""Train and evaluate 9 standalone classifiers and 3 soft-voting hybrids.

Run: python scripts/compare_models.py
Results are held-out workbook-label agreement, NOT external legal validation.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'FIDIC_Thesis_Style_Labeled_Dataset.xlsx'
OUT = ROOT / 'models'
SEED = 42


def candidates():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1500, class_weight='balanced'),
        'Multinomial Naive Bayes': MultinomialNB(),
        'Complement Naive Bayes': ComplementNB(),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(class_weight='balanced', random_state=SEED), cv=3),
        'SGD Logistic': SGDClassifier(loss='log_loss', class_weight='balanced', random_state=SEED),
        'Ridge Classifier': CalibratedClassifierCV(RidgeClassifier(class_weight='balanced'), cv=3),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=24, random_state=SEED, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=24, random_state=SEED, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(max_depth=30, random_state=SEED),
    }


def metrics(y, pred, elapsed):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    return dict(accuracy=round(float(accuracy_score(y, pred)), 5), macro_precision=round(float(p), 5), macro_recall=round(float(r), 5), macro_f1=round(float(f), 5), inference_seconds=round(float(elapsed), 4))


def main():
    df = pd.read_excel(DATA, sheet_name='Combined_Unique').dropna(subset=['Sentence', 'Sentence_Type'])
    df['Sentence'] = df['Sentence'].astype(str).str.strip()
    df['Sentence_Type'] = df['Sentence_Type'].astype(str).str.strip()
    df = df[df['Sentence_Type'].isin(['Heading','Definition','Obligation','Risk','Right']) & df['Sentence'].ne('')]
    df = df.drop_duplicates(subset=['Sentence'], keep=False)
    X_train, X_test, y_train, y_test = train_test_split(df.Sentence, df.Sentence_Type, test_size=.2, stratify=df.Sentence_Type, random_state=SEED)
    models, records = {}, []
    for name, clf in candidates().items():
        print('Training', name, flush=True)
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_features=12000, sublinear_tf=True), clf)
        pipe.fit(X_train, y_train)
        start = time.perf_counter()
        pred = pipe.predict(X_test)
        elapsed = time.perf_counter() - start
        models[name] = pipe
        records.append(dict(model=name, category='Individual', **metrics(y_test, pred, elapsed)))
    groups = {
        'Hybrid A (LR + NB + SVM)': ['Logistic Regression','Multinomial Naive Bayes','Linear SVM'],
        'Hybrid B (SGD + Complement NB + Ridge)': ['SGD Logistic','Complement Naive Bayes','Ridge Classifier'],
        'Hybrid C (LR + Forest + Extra Trees)': ['Logistic Regression','Random Forest','Extra Trees'],
    }
    classes = np.unique(y_train)
    for name, members in groups.items():
        start = time.perf_counter()
        probabilities = np.mean([models[m].predict_proba(X_test)[:, [list(models[m].classes_).index(c) for c in classes]] for m in members], axis=0)
        pred = classes[np.argmax(probabilities, axis=1)]
        records.append(dict(model=name, category='Hybrid', **metrics(y_test, pred, time.perf_counter()-start)))
    OUT.mkdir(exist_ok=True)
    joblib.dump({'models': models, 'hybrids': groups, 'classes': classes.tolist()}, OUT / 'comparison_models.joblib', compress=3)
    payload = {'evaluation_note': '80/20 stratified split of workbook labels, after removing duplicate sentence text. Not independently expert-validated; no guarantee on new contracts.', 'train_rows':len(X_train), 'test_rows':len(X_test), 'random_state':SEED, 'results':records}
    (OUT / 'comparison_metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(pd.DataFrame(records).sort_values('macro_f1', ascending=False).to_string(index=False))
    print('Saved model comparison artifacts.')

if __name__ == '__main__':
    main()
