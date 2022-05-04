import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from scipy.stats import zscore

np.random.seed(42)

def get_compiled_model():
    model = Sequential([
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.Adam(learning_rate=0.00001), 
        metrics=['AUC']
    )
    return model


def preprocess(df):
    df = df[df["uniq_Op"] != "?"]
    df = df.drop([0])

    df['uniq_Op'] = df['uniq_Op'].astype(str).astype(int)
    df['uniq_Opnd'] = df['uniq_Opnd'].astype(str).astype(int)
    df['total_Op'] = df['total_Op'].astype(str).astype(int)
    df['total_Opnd'] = df['total_Opnd'].astype(str).astype(int)
    df['branchCount'] = df['branchCount'].astype(str).astype(int)

    y = df.pop('defects').to_numpy().astype(int)
    df = df.apply(zscore)
    X = df.to_numpy().astype("float32")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, shuffle=True)
    
    oversampler = RandomOverSampler(sampling_strategy='minority')
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

df = pd.read_csv('Datasets/jm1.csv')
X_train, X_val, X_test, y_train, y_val, y_test = preprocess(df)

model = get_compiled_model()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=300)

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()