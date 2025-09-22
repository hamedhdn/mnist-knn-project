# Phase 2 – Manual KNN Implementation on MNIST 🧠

In this phase of the project, the goal is to manually implement the K-Nearest Neighbors (KNN) algorithm using NumPy and compare its performance with the built-in version from scikit-learn on the MNIST digit classification dataset.

## 🎯 Objectives
- Gain a deeper understanding of how KNN works internally
- Build a custom KNN classifier from scratch using NumPy
- Compare accuracy and performance with scikit-learn's optimized KNN

## 📁 Project Structure

phase2/
├── knn_numpy.py # Manual implementation of knn_predict 
├── compare_knn.py # Accuracy comparison between manual and sklearn 
├── README.md


## 📦 Required Libraries
```bash
pip install numpy scikit-learn keras



## 🛠️ Steps

1.Load MNIST Dataset
Using keras.datasets, the 28×28 grayscale digit images are loaded and flattened.

2.Normalize the Data
Pixel values are scaled to the range [0, 1] by dividing by 255.

3.Implement knn_predict Function

   .Compute Euclidean distances

   .Identify k nearest neighbors

   .Use majority voting to assign labels

4.Compare with scikit-learn's KNN 
Use KNeighborsClassifier to benchmark accuracy and speed.

5.Evaluate Accuracy 
Accuracy is calculated using accuracy_score from sklearn.metrics.



## 📊 Sample Results

Accuracy (Manual KNN): 0.9900
Accuracy (Scikit-learn KNN): 0.9900


## 🚀 Suggestions for Improvement

    .Use vectorized operations to speed up the manual implementation

    .Test on smaller subsets to reduce runtime

    .Explore fast nearest neighbor libraries like FAISS or HNSWlib for large-scale applications