import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# 1. Load and preprocess data
iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)  # standardize features
y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1,1))  # one-hot encode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Build, train, evaluate model
def build_and_train_model(name,opt):
    print(f"\n--- Training with {name} ---")
    model = Sequential([
        Dense(10,input_dim=X_train.shape[1],activation='relu'),
        Dense(10,activation='relu'),
        Dense(3,activation='softmax')
    ])
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=50,batch_size=5,validation_split=0.2,verbose=0)
    y_pred = np.argmax(model.predict(X_test,verbose=0),axis=1)
    y_true = np.argmax(y_test,axis=1)
    acc = accuracy_score(y_true,y_pred)
    print(f"Test Accuracy with {name}: {acc:.4f}")
    return acc

# 3. Compare optimizers
optimizers = {"SGD":SGD(0.01),"Adam":Adam(0.001),"RMSprop":RMSprop(0.001)}
performance_results = {n: build_and_train_model(n,opt) for n,opt in optimizers.items()}

# 4. Summary
print("\n--- Summary of Performance ---")
for n,acc in performance_results.items():
    print(f"{n}: {acc:.4f} accuracy")
