import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

POP=20
GEN=20

def init(dim):
    return [(np.random.rand(dim),np.random.rand(dim)) for _ in range(POP)]

def fitness(ind,X,y,tf):
    m,s=ind
    s=np.clip(s,0.1,1.0)

    Xf=tf(X,m,s)
    Xtr,Xval,ytr,yval=train_test_split(Xf,y,test_size=0.2)

    clf=LogisticRegression(max_iter=500,class_weight='balanced')
    clf.fit(Xtr,ytr)
    return accuracy_score(yval,clf.predict(Xval))

def run_ga(X,y,tf):
    pop=init(X.shape[1])
    fit_curve=[]; div=[]

    for g in range(GEN):
        scores=[(i,fitness(i,X,y,tf)) for i in pop]
        scores.sort(key=lambda x:x[1],reverse=True)

        fit_curve.append(scores[0][1])
        div.append(np.std([np.mean(i[0]) for i,_ in scores]))

        top=[i for i,_ in scores[:POP//2]]

        new=[]
        for i in top:
            m,s=i
            new.append((m+np.random.normal(0,0.02,m.shape),
                        s+np.random.normal(0,0.02,s.shape)))

        pop=top+new

    return scores[0][0],fit_curve,div