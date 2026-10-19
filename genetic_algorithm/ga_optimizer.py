"""Gaussian membership parameter search; final test data must never enter fitness."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

POP = 20
GEN = 20

def run_ga_controlled(X_train, y_train, X_val, y_val, transform, seed=42,
                      population=POP, generations=GEN):
    """Return best-ever (centers, spreads), history. Search uses train/val only."""
    rng = np.random.default_rng(seed)
    dim = X_train.shape[1]
    pop = [(rng.uniform(-1, 1, dim), rng.uniform(0.1, 2, dim))
           for _ in range(population)]
    best_ind, best_score = None, -np.inf
    history = []

    def score(ind):
        centers, spreads = ind
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
        model.fit(transform(X_train, centers, spreads), y_train)
        pred = model.predict(transform(X_val, centers, spreads))
        return accuracy_score(y_val, pred)

    for generation in range(generations):
        ranked = sorted(((score(ind), ind) for ind in pop),
                        key=lambda item: item[0], reverse=True)
        if ranked[0][0] > best_score:
            best_score = ranked[0][0]
            best_ind = tuple(a.copy() for a in ranked[0][1])
        history.append({'generation': generation + 1,
                        'generation_best': float(ranked[0][0]),
                        'best_ever': float(best_score)})
        survivors = [ind for _, ind in ranked[:max(1, population // 2)]]
        next_pop = [(a.copy(), b.copy()) for a, b in survivors]
        while len(next_pop) < population:
            parent = survivors[rng.integers(len(survivors))]
            centers = np.clip(parent[0] + rng.normal(0, .05, dim), -3, 3)
            spreads = np.clip(parent[1] + rng.normal(0, .05, dim), .1, 3)
            next_pop.append((centers, spreads))
        pop = next_pop
    return best_ind, history

# Legacy original GA retained so original main.py still runs as before.
from sklearn.model_selection import train_test_split
POP_LEGACY = POP
GEN_LEGACY = GEN

def init(dim):
    return [(np.random.rand(dim), np.random.rand(dim)) for _ in range(POP_LEGACY)]

def fitness(ind, X, y, tf):
    m, s = ind
    Xf = tf(X, m, np.clip(s, .1, 1.))
    Xtr, Xval, ytr, yval = train_test_split(Xf, y, test_size=.2)
    clf = LogisticRegression(max_iter=500, class_weight='balanced')
    clf.fit(Xtr, ytr)
    return accuracy_score(yval, clf.predict(Xval))

def run_ga(X, y, tf):
    pop = init(X.shape[1]); curve = []; diversity = []
    for _ in range(GEN_LEGACY):
        ranked = sorted(((fitness(i, X, y, tf), i) for i in pop), reverse=True, key=lambda p:p[0])
        curve.append(ranked[0][0]); diversity.append(np.std([np.mean(i[0]) for _,i in ranked]))
        top = [i for _,i in ranked[:POP_LEGACY//2]]
        pop = top + [(m + np.random.normal(0,.02,m.shape), s + np.random.normal(0,.02,s.shape)) for m,s in top]
    return ranked[0][1], curve, diversity
