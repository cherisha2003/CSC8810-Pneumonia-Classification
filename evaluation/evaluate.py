import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_conf(y,p):
    cm=confusion_matrix(y,p)
    sns.heatmap(cm,annot=True,fmt='d')
    plt.savefig("results/confusion_matrix.png"); plt.close()

def plot_roc(y,prob):
    fpr,tpr,_=roc_curve(y,prob)
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
    plt.legend()
    plt.savefig("results/roc_curve.png"); plt.close()

def ablation(res):
    plt.bar(res.keys(),res.values())
    plt.xticks(rotation=30)
    plt.savefig("results/ablation_comparison.png"); plt.close()