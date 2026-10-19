"""New leakage-free, same-split comparison. Does not overwrite original results."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from preprocessing.preprocess import load_data
from feature_extraction.densenet_features import get_model, extract_features
from fuzzy_system.fuzzy_layer import fuzzy_transform
from genetic_algorithm.ga_optimizer import run_ga_controlled
from training.train_classifier import train_model

SEED = 42
OUT = Path('results/controlled')
DATA = 'dataset/chest_xray/train'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    images, y = load_data(DATA)
    if len(y) == 0:
        raise RuntimeError('No images found. See README dataset instructions.')
    # Feature extraction is label-independent; splitting/scaling are done afterward.
    features = extract_features(get_model(), images)
    idx = np.arange(len(y))
    train_val, test = train_test_split(idx, test_size=.2, stratify=y, random_state=SEED)
    train, val = train_test_split(train_val, test_size=.25, stratify=y[train_val], random_state=SEED)
    scaler = StandardScaler().fit(features[train])
    X_train, X_val, X_test = [scaler.transform(features[i]) for i in (train,val,test)]
    y_train, y_val, y_test = y[train], y[val], y[test]
    rng = np.random.default_rng(SEED)
    centers = rng.uniform(-1,1, X_train.shape[1]); spreads = rng.uniform(.1,2, X_train.shape[1])
    (best_centers,best_spreads), history = run_ga_controlled(X_train,y_train,X_val,y_val,fuzzy_transform,seed=SEED)
    with open(OUT/'ga_history.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['generation','generation_best','best_ever']); w.writeheader(); w.writerows(history)
    plt.figure(); plt.plot([h['generation'] for h in history],[h['best_ever'] for h in history]); plt.xlabel('Generation'); plt.ylabel('Best validation accuracy'); plt.tight_layout(); plt.savefig(OUT/'ga_best_validation.png'); plt.close()
    # Final evaluation: train each classifier on train+validation with fixed parameters.
    X_dev = np.concatenate([X_train,X_val]); y_dev = np.concatenate([y_train,y_val])
    transforms = {'baseline': lambda a:a,
                  'fuzzy':lambda a:fuzzy_transform(a,centers,spreads),
                  'ga_fuzzy':lambda a:fuzzy_transform(a,best_centers,best_spreads)}
    metrics = []
    for name, transform in transforms.items():
        clf = train_model(transform(X_dev), y_dev)
        pred = clf.predict(transform(X_test))
        proba = clf.predict_proba(transform(X_test))[:,list(clf.classes_).index(1)]
        row = {'model':name,'test_n':len(y_test),'accuracy':float(accuracy_score(y_test,pred)),
               'balanced_accuracy':float(balanced_accuracy_score(y_test,pred)),
               'auc':float(roc_auc_score(y_test,proba))}
        metrics.append(row)
        cm = confusion_matrix(y_test,pred,labels=[0,1])
        disp=ConfusionMatrixDisplay(cm,display_labels=['Normal','Pneumonia'])
        disp.plot(cmap='Blues',values_format='d'); plt.title(f'{name}: held-out test'); plt.tight_layout(); plt.savefig(OUT/f'{name}_confusion.png'); plt.close()
        RocCurveDisplay.from_predictions(y_test,proba,name=name); plt.title(f'{name}: held-out test'); plt.tight_layout(); plt.savefig(OUT/f'{name}_roc.png'); plt.close()
        print(name,row)
    with open(OUT/'metrics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
    with open(OUT/'run_details.json','w') as f:
        json.dump({'seed':SEED,'dataset_path':DATA,'sample_count':len(y),'train_n':len(train),'validation_n':len(val),'test_n':len(test),'train_class_counts':np.bincount(y_train).tolist(),'test_class_counts':np.bincount(y_test).tolist(),'ga_population':20,'ga_generations':20,'note':'All metrics from same held-out test split; scaler fitted on train only.'},f,indent=2)
    print('Saved controlled outputs in',OUT)

if __name__ == '__main__':
    main()
