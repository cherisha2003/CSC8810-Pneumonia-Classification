import numpy as np, matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from preprocessing.preprocess import *
from feature_extraction.densenet_features import *
from fuzzy_system.fuzzy_layer import *
from genetic_algorithm.ga_optimizer import *
from training.train_classifier import *
from evaluation.evaluate import *

# LOAD
X,y=load_data("dataset/chest_xray/train")
visualize_augmented(X)
class_distribution(y)

# FEATURES
model=get_model()
features=extract_features(model,X)

# SCALE (CRITICAL FIX)
scaler=StandardScaler()
features=scaler.fit_transform(features)

# TSNE
emb=TSNE(n_components=2).fit_transform(features)
plt.scatter(emb[:,0],emb[:,1],c=y)
plt.savefig("results/tsne_embeddings.png"); plt.close()

# SPLIT
Xtr,Xte,ytr,yte=train_test_split(features,y,test_size=0.2)

# BASELINE
m_base=train_model(Xtr,ytr)
base_acc=m_base.score(Xte,yte)

# FUZZY
means=np.random.rand(features.shape[1])
sigmas=np.random.rand(features.shape[1])
fuzzy=fuzzy_transform(features,means,sigmas)

ftr,fte,ytr,yte=train_test_split(fuzzy,y,test_size=0.2)
m_fuzzy=train_model(ftr,ytr)
fuzzy_acc=m_fuzzy.score(fte,yte)

plot_mf(means,sigmas)

# GA
(best_m,best_s),fit_curve,div=run_ga(features,y,fuzzy_transform)

plt.plot(fit_curve); plt.savefig("results/ga_fitness_curve.png"); plt.close()
plt.plot(div); plt.savefig("results/ga_population_diversity.png"); plt.close()

compare_mf(means,sigmas,best_m,best_s)

# FINAL MODEL
fuzzy_opt=fuzzy_transform(features,best_m,best_s)
ftr,fte,ytr,yte=train_test_split(fuzzy_opt,y,test_size=0.2)

m_final=train_model(ftr,ytr)
y_pred=m_final.predict(fte)
y_prob=m_final.predict_proba(fte)[:,1]

plot_conf(yte,y_pred)
plot_roc(yte,y_prob)

feature_dist(features,fuzzy_opt)
heatmap(fuzzy_opt)

# CV
kf=KFold(n_splits=5)
scores=[]
for tr,te in kf.split(fuzzy_opt):
    m=train_model(fuzzy_opt[tr],y[tr])
    scores.append(m.score(fuzzy_opt[te],y[te]))

plt.bar(range(5),scores)
plt.savefig("results/cv_fold_accuracies.png"); plt.close()

print("CV:",np.mean(scores),"±",np.std(scores))

# RANDOM
rand=np.random.rand(*features.shape)
rtr,rte,ytr,yte=train_test_split(rand,y,test_size=0.2)
m_rand=train_model(rtr,ytr)
rand_acc=m_rand.score(rte,yte)

# ABLATION
res={
    "DenseNet":base_acc,
    "Fuzzy":fuzzy_acc,
    "GA+Fuzzy":np.mean(scores),
    "Random":rand_acc
}
ablation(res)

joblib.dump(m_base, "model_baseline.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved for demo!")

print("FINAL:",res)