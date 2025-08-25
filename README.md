Neural-Network-for-Higgs-Boson-Classification-from-scratchas signal (Higgs Boson production, class 1) or background (noise, class 0) using the UCI Higgs Boson dataset.
Designed for Machine Learning for Science (ML4Sci) applications, it mirrors tasks in high-energy physics, such as those at CERN's Large Hadron Collider.

The implementation includes:
Data Handling: Loads the UCI Higgs dataset (10,000 subsampled instances, 28 features) directly from its URL.
Exploratory Data Analysis (EDA): Computes descriptive statistics (means, standard deviations), inferential tests (t-tests with p-values <<0.05 confirming feature discriminability), and correlations (~0.3-0.5 average magnitude).
Visualization: Interactive Plotly plots, including PCA projections, box plots, histograms, and confusion matrices, revealing class separation and model performance.
Neural Network: A from-scratch implementation with one hidden layer (64 neurons, ReLU activation), trained via mini-batch gradient descent with cross-entropy loss, achieving ~70-80% test accuracy.
Evaluation: Reports accuracy and visualizes decision boundaries via PCA, with confusion matrices highlighting precision-recall trade-offs.

Dataset
The UCI Higgs Boson dataset contains 11 million Monte Carlo simulated particle collision events, with 28 features (21 low-level kinematic properties, e.g., jet momenta, and 7 high-level derived quantities, e.g., invariant masses).
A subsample of 10,000 instances is used for computational efficiency, balancing signal and background events.

Features
Statistical EDA: Descriptive stats (e.g., means differ significantly, p-values <<0.05 via t-tests), correlation analysis (moderate collinearity, |r| ~0.3-0.5).
Interactive Visualizations: Plotly scatter plots (PCA-reduced), box plots, histograms, and heatmaps for correlations and confusion matrices.
Neural Network: Custom implementation with ReLU activation, softmax output, and mini-batch gradient descent (batch size 64, learning rate 0.01, 500 epochs).
Performance: Achieves ~70-80% test accuracy, with loss curves and PCA-based decision boundary visualizations.

Prerequisites
Python: 3.7+
Libraries: Install via pip install numpy pandas plotly scipy
No external ML frameworks required (e.g., TensorFlow, PyTorch).
