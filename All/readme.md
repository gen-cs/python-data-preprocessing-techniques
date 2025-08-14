# 🧹 Data Cleaning & ⚙️ Feature Engineering: Basic → Advanced

> **"80% of your ML project time is spent here — and for good reason."**  
This repository is a **complete Python-powered toolkit** for preparing raw data and transforming it into **machine-learning-ready features**.  
From **basic cleaning** to **advanced automated feature engineering**, this guide covers the techniques you need to turn messy datasets into predictive gold.

---

## 🧼 Data Cleaning Techniques

### **📍 Basic Data Cleaning**
#### 🧩 Missing Data Handling
- **Removal** → Delete observations with missing values  
- **Imputation** → Fill missing values with **mean**, **median**, **mode**, or statistical methods  
- **Forward / Backward Fill** → For time series data  
- **Interpolation** → Linear / polynomial interpolation for continuous variables  

#### 🔍 Duplicate Detection & Removal
- Identify identical or near-identical records  
- Remove exact duplicates while preserving unique info  
- Handle slight differences from data entry errors  

#### 📊 Outlier Management
- **Detection** → IQR, Z-score, visualizations  
- **Treatment** → Remove, transform, or cap outliers  
- **Contextual Assessment** → Decide if they are valid extremes or errors  

#### 🛠 Structural Error Correction
- Fix inconsistencies in formats and variable types  
- Standardize naming conventions & capitalization  
- Correct typos & inconsistent labels  

---

### **🚀 Advanced Data Cleaning**
#### 🔇 Noise Reduction
- **Binning** → Group data points, smooth with bin means  
- **Regression Smoothing** → Fit regression functions  
- **Clustering** → Detect and handle anomalies outside cluster bounds  

#### ⚠️ Inconsistency Detection
- **Pointwise** → Values outside expected ranges (negative ages, impossible dates)  
- **Multi-variable** → Conflicting info across features  
- **Domain-Specific** → Apply business rules & constraints  

---

## ⚙️ Feature Engineering Techniques

### **🔰 Basic Feature Engineering**
#### 🏷 Categorical Encoding
- **Label Encoding** → Assign numeric IDs to categories  
- **One-Hot Encoding** → Binary columns per category  
- **Binary Encoding** → Use binary digits  
- **Target Encoding** → Encode by target variable statistics  

#### 🔢 Numerical Transformations
- **Log / Power Transformations** → Fix skewness  
- **Scaling & Normalization**:  
  - Min-Max Scaling (0–1 range)  
  - Standardization (zero mean, unit variance)  
  - Robust Scaling (median, IQR — outlier resistant)  

#### ⏳ Date-Time Feature Extraction
- Extract **day, month, year, hour, weekday**  
- Create **cyclical features** (sin/cos for periodic patterns)  
- Calculate **time deltas** & elapsed periods  

---

### **📈 Intermediate Feature Engineering**
#### 📦 Binning & Discretization
- Equal-width intervals  
- Equal-frequency bins  
- Domain-knowledge custom bins  

#### 🔖 Boolean / Flag Features
- Convert conditions into binary indicators  
- Threshold-based anomaly flags  
- Status indicators via business rules  

#### ➗ Feature Interactions
- Multiplication, addition, subtraction  
- Ratios, percentages, proportions  

---

### **💎 Advanced Feature Engineering**
#### 📐 Polynomial Features
- Higher-order terms (x², x³), cross-products  
- Capture non-linear relationships (beware overfitting)  

#### 📊 Statistical Features
- Rolling statistics (moving averages, rolling std)  
- Group aggregations (mean, max, min, std)  
- Lag features for time series  

#### 🎯 Domain-Specific
- **Text Data** → TF-IDF, embeddings, n-grams  
- **Time Series** → Seasonal decomposition, FFT features  
- **Image Data** → Texture, edges, color histograms  

---

## 🧠 Feature Selection Techniques

### 🧮 Filter Methods
- Univariate tests: Chi-square, F-test, mutual info  
- Correlation-based: Pearson, Spearman  
- Variance thresholding (remove low-variance features)  

### 🔁 Wrapper Methods
- Forward selection  
- Backward elimination  
- Recursive Feature Elimination (RFE)  

### 🌲 Embedded Methods
- L1 Regularization (Lasso)  
- Tree-based feature importance  
- Gradient boosting importance  

---

## 📉 Dimensionality Reduction Techniques

### 📏 Linear
- PCA → Maximize variance in fewer dimensions  
- LDA → Maximize class separation  
- Factor Analysis → Extract latent factors  

### 🔀 Non-linear
- t-SNE → Preserve local structure in lower-dim space  
- UMAP → Uniform manifold approximation  
- Autoencoders → Neural-based compression  

---

## 🤖 Automated Feature Engineering

### 📦 Popular Tools
- **Featuretools** → Deep Feature Synthesis (DFS)  
- **AutoFeat** → Non-linear features + L1 selection  
- **TSFresh** → 60+ time-series features  
- **Feature Engine** → Advanced encoders, correlation filters  

### ✅ Benefits
- **Efficiency** → Scale across projects  
- **Bias Reduction** → Avoid human preconceptions  
- **Consistency** → Standardize approach  
- **Exploration** → Find hidden relationships  

---

## 📝 Text Data Preprocessing & Feature Engineering

### 🧼 Basic Text Cleaning
- Tokenization, stop word removal  
- Lowercasing, punctuation removal  

### 🔍 Advanced Text Processing
- Stemming, lemmatization  
- N-grams, POS tagging  
- Named Entity Recognition (NER)  

### 🔡 Text Feature Engineering
- TF-IDF  
- Word embeddings (Word2Vec, GloVe, FastText)  
- Sentence embeddings  
- Bag of Words (BoW)  

---

## 💡 Best Practices

- 📌 **Algorithm-dependent preprocessing** — tree models need less scaling, distance-based models need more  
- 🔒 **Prevent data leakage** — respect train/validation/test boundaries  
- 📊 **Validate impact** — cross-validate preprocessing effects  
- ⚡ **Balance performance & efficiency** — avoid over-engineering  

---

## 🎯 Final Thoughts
This **progression from basic cleaning to advanced automated feature engineering** is your complete data preprocessing arsenal.  
The techniques you choose will depend on:
- 📂 Data characteristics  
- 🏭 Problem domain  
- 🖥 Computational resources  
- 🧠 Algorithm choice  

---

🚀 **Turn raw data into predictive power — one feature at a time.**
