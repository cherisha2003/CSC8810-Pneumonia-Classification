from tensorflow.keras.applications import DenseNet121

def get_model():
    return DenseNet121(weights='imagenet',include_top=False,pooling='avg')

def extract_features(model,X):
    return model.predict(X,batch_size=16)