from keras.datasets import mnist
import numpy as np 

def knn_predict(X_train , y_train , X_test , k):
    predictions = []
    for try_sample in X_test :

        distance = np.linalg.norm(X_train - try_sample , axis=1)
        near_index = np.argsort(distance)[:k]
        near_lable = y_train[near_index]
        values , counts = np.unique(near_lable , return_counts= True)
        predicted_lable = values[np.argmax(counts)]
        
        predictions.append(predicted_lable)

    return np.array(predictions)



(X_train , y_train) , (X_test,y_test) = mnist.load_data()

X_train = X_train/255
X_test = X_test/255

#Flat datas:
X_train = X_train.reshape(len(X_train) , -1 )
X_test = X_test.reshape(len(X_test) , -1)


y_pred = knn_predict(X_train, y_train, X_test[:100], k=5)

print(y_pred)