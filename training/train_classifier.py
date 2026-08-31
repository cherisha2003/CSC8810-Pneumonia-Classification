from sklearn.linear_model import LogisticRegression

def train_model(X,y):
    m=LogisticRegression(max_iter=1000,class_weight='balanced')
    m.fit(X,y)
    return m