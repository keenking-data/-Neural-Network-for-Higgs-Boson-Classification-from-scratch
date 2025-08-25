import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind

# Step 1: Load Real Dataset from URL for Higgs Boson Classification
# Dataset: 11M rows, but we take 10,000 for tractable from-scratch training.


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
df = pd.read_csv(url, header=None, compression='gzip')
data = df.sample(10000, random_state=42).reset_index(drop=True)  
X = data.iloc[:, 1:].values  
y = data[0].values  
feature_names = [f'Feature_{i}' for i in range(1, 29)]  #
data_df = pd.DataFrame(X, columns=feature_names)
data_df['Class'] = y.astype(int)  # For EDA.

# Step 2: Exploratory Data Analysis (EDA) with Statistics
# Descriptive: Means/std highlight scales (e.g., features normalized around 0-1); aids in detecting anomalies.
print("Basic Statistics:\n", data_df.describe())

# Grouped: Compare classes; signal often has higher means in certain features (e.g., invariant mass).
print("Grouped Statistics by Class:\n", data_df.groupby('Class').describe())

# Correlation: Matrix shows inter-feature relationships; stats: avg |corr| ~0.3-0.5, potential multicollinearity.
corr_matrix = data_df.corr()
print("Correlation Matrix:\n", corr_matrix)

# Inferential: T-tests for each feature; low p-values (<0.05) reject null of equal means, indicating separability.
ttest_results = {}
for col in feature_names:
    bg = data_df[data_df['Class'] == 0][col]
    sig = data_df[data_df['Class'] == 1][col]
    ttest_results[col] = ttest_ind(bg, sig)
print("T-test Results (statistic, p-value):")
for col, res in ttest_results.items():
    print(f"{col}: {res}")

# Visualizations: Plotly for interactivity; e.g., correlation heatmap reveals patterns.
fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='Viridis'))
fig_corr.update_layout(title='Correlation Matrix Heatmap')
fig_corr.show()

# PCA for 2D Visualization: Reduce dimensionality via SVD (first principles: eigen-decomposition of covariance).
# Statistics: Captures variance; top components explain class separation.
cov = np.cov(X.T)
eigvals, eigvecs = np.linalg.eig(cov)
top_idx = np.argsort(eigvals)[::-1][:2]
pca_projection = X @ eigvecs[:, top_idx]
pca_df = pd.DataFrame(pca_projection, columns=['PC1', 'PC2'])
pca_df['Class'] = data_df['Class']
fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Class',
                     title='PCA Projection of Higgs Features',
                     labels={'Class': 'Particle Type (0: Background, 1: Signal)'})
fig_pca.show()

# Example Box Plot for First Feature: Quantiles show distribution spread (IQR for variability).
fig_box = px.box(data_df, x='Class', y='Feature_1', title='Box Plot of Feature_1 by Class')
fig_box.show()

# Histogram for a High-Level Feature (e.g., Feature_22: missing energy magnitude).
fig_hist = px.histogram(data_df, x='Feature_22', color='Class', barmode='overlay', title='Histogram of Feature_22 (Missing Energy)')
fig_hist.show()

# Step 3: Data Preparation
# One-hot: For softmax output.
y_onehot = np.eye(2)[y.astype(int)]
# Train-test split: 80/20; stratified implicitly via shuffle.
n_samples = X.shape[0]
indices = np.random.permutation(n_samples)
split = int(0.8 * n_samples)
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y_onehot[indices[:split]], y_onehot[indices[split:]]
y_test_labels = np.argmax(y_test, axis=1)  # For evaluation.

# Step 4: Neural Network from Scratch
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Init: Xavier-like (scaled random) for better gradient flow.
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, z):
        return np.maximum(0, z)  # ReLU: Improves training over sigmoid (avoids vanishing gradients).
    
    def relu_deriv(self, a):
        return (a > 0).astype(float)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def compute_loss(self, y, y_pred):
        return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
    
    def backward(self, X, y, y_pred, learning_rate):
        error = y_pred - y
        dW2 = np.dot(self.a1.T, error) / X.shape[0]
        db2 = np.sum(error, axis=0, keepdims=True) / X.shape[0]
        d_hidden = np.dot(error, self.W2.T) * self.relu_deriv(self.a1)
        dW1 = np.dot(X.T, d_hidden) / X.shape[0]
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / X.shape[0]
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate, batch_size=64):
        losses = []
        n = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(n)
            X_shuf, y_shuf = X[indices], y[indices]
            for i in range(0, n, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred, learning_rate)
            y_pred_full = self.forward(X)
            loss = self.compute_loss(y, y_pred_full)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
        return losses
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Instantiate: 28 inputs, 64 hidden (balanced for capacity), 2 outputs.
nn = NeuralNetwork(input_size=28, hidden_size=64, output_size=2)
losses = nn.train(X_train, y_train, epochs=500, learning_rate=0.01)

# Step 5: Evaluation
y_pred_train = nn.predict(X_train)
y_pred_test = nn.predict(X_test)
train_acc = np.mean(y_pred_train == np.argmax(y_train, axis=1))
test_acc = np.mean(y_pred_test == y_test_labels)
print(f'Train Accuracy: {train_acc:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')

# Confusion Matrix: Statistical tool for error analysis (e.g., false negatives impact discovery rate).
cm = np.zeros((2, 2), dtype=int)
for true, pred in zip(y_test_labels, y_pred_test):
    cm[true, pred] += 1
fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Background', 'Signal'], y=['Background', 'Signal'], colorscale='Blues'))
fig_cm.update_layout(title='Confusion Matrix - Test Set')
fig_cm.show()

# Loss Curve: Descent trajectory; plateaus suggest learning rate adjustments.
fig_loss = px.line(x=range(len(losses)), y=losses, title='Training Loss Curve', labels={'x': 'Epoch', 'y': 'Loss'})
fig_loss.show()

# PCA with Predictions: Visualize test set decisions in reduced space.
test_pca = X_test @ eigvecs[:, top_idx]
pred_df = pd.DataFrame(test_pca, columns=['PC1', 'PC2'])
pred_df['True Class'] = y_test_labels
pred_df['Predicted Class'] = y_pred_test
fig_pred = px.scatter(pred_df, x='PC1', y='PC2', color='Predicted Class', symbol='True Class',
                      title='PCA with Predictions (Symbols: True, Colors: Pred)')
fig_pred.show()

print('Reflect: How might regularization mitigate overfitting, statistically reducing variance?.')