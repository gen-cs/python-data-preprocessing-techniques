# 🧹 Data Cleaning and Feature Engineering Techniques: Basic to Advanced

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python)](https://python.org)
[![ML](https://img.shields.io/badge/Machine%20Learning-Guide-green)](https://github.com)

> **💡 Key Insight**: Data cleaning and feature engineering form the foundation of successful machine learning projects, often taking up **50-80%** of the entire classification process. These preprocessing steps are crucial for preparing raw data and transforming it into meaningful features that machine learning models can effectively utilize.

## 📋 Table of Contents

- [🧽 Data Cleaning Techniques](#-data-cleaning-techniques)
  - [Basic Data Cleaning](#basic-data-cleaning)
  - [Advanced Data Cleaning](#advanced-data-cleaning)
- [🔧 Feature Engineering Techniques](#-feature-engineering-techniques)
  - [Basic Feature Engineering](#basic-feature-engineering)
  - [Intermediate Feature Engineering](#intermediate-feature-engineering)
  - [Advanced Feature Engineering](#advanced-feature-engineering)
- [🎯 Feature Selection Techniques](#-feature-selection-techniques)
- [📉 Dimensionality Reduction Techniques](#-dimensionality-reduction-techniques)
- [🤖 Automated Feature Engineering](#-automated-feature-engineering)
- [📝 Text Data Preprocessing](#-text-data-preprocessing)
- [✅ Best Practices and Considerations](#-best-practices-and-considerations)

---

## 🧽 Data Cleaning Techniques

### Basic Data Cleaning

#### 🔍 Missing Data Handling

| Method | Description | Use Case |
|--------|-------------|----------|
| **Removal** | Delete observations with missing values | Small datasets with few missing values |
| **Imputation** | Fill missing values using mean, median, mode, or statistical methods | Numerical data with pattern-based missing values |
| **Forward/Backward Fill** | For time series data | Sequential data where previous/next values are meaningful |
| **Interpolation** | Use linear or polynomial interpolation for continuous variables | Smooth continuous data |

#### 🔄 Duplicate Detection and Removal

- ✅ Identify identical or near-identical records
- ✅ Remove exact duplicates while preserving unique information  
- ✅ Handle cases where slight differences exist due to data entry errors

#### 📊 Outlier Management

```python
# Example: Outlier detection using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]
```

- **Detection**: Use statistical methods (IQR, Z-score) or visualization techniques
- **Treatment**: Remove, transform, or cap outliers based on domain knowledge
- **Contextual Assessment**: Determine if outliers represent valid extreme values or errors

#### 🔧 Structural Error Correction

- 🏷️ Fix inconsistencies in data formats and variable types
- 📝 Standardize naming conventions and capitalization
- ✏️ Correct typos and inconsistent category labels

### Advanced Data Cleaning

#### 🎚️ Noise Reduction

| Technique | Method | Best For |
|-----------|--------|----------|
| **Binning** | Group data points and smooth using bin means or boundaries | Categorical conversion |
| **Regression** | Fit data to regression functions for smoothing | Trend-based data |
| **Clustering** | Identify and handle outliers that fall outside cluster boundaries | Multi-dimensional outliers |

#### ⚠️ Inconsistency Detection

- **Pointwise Inconsistencies**: Identify values outside expected ranges (negative ages, impossible dates)
- **Multi-variable Inconsistencies**: Detect conflicting information across related features
- **Domain-specific Validation**: Apply business rules and constraints

---

## 🔧 Feature Engineering Techniques

### Basic Feature Engineering

#### 🏷️ Categorical Encoding

```python
# Example encoding methods
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import BinaryEncoder, TargetEncoder

# Label Encoding
le = LabelEncoder()
data['category_encoded'] = le.fit_transform(data['category'])

# One-Hot Encoding
ohe = OneHotEncoder(sparse=False)
encoded_features = ohe.fit_transform(data[['category']])
```

| Encoding Type | Description | When to Use |
|---------------|-------------|-------------|
| **Label Encoding** | Assign numerical values to categories | Ordinal categories |
| **One-Hot Encoding** | Create binary columns for each category | Nominal categories |
| **Binary Encoding** | Represent categories using binary digits | High cardinality categories |
| **Target Encoding** | Encode categories based on target variable statistics | Predictive encoding |

#### 📈 Numerical Transformations

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

# Log transformation for skewed data
data['feature_log'] = np.log1p(data['feature'])

# Scaling methods
scaler = StandardScaler()  # Zero mean, unit variance
scaler = MinMaxScaler()    # Scale to [0,1] range
scaler = RobustScaler()    # Median and IQR based
```

#### 📅 Date-Time Feature Extraction

```python
# Extract temporal features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day_of_week'] = data['date'].dt.dayofweek

# Cyclical features for periodic patterns
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
```

### Intermediate Feature Engineering

#### 📦 Binning and Discretization

| Binning Type | Description | Use Case |
|--------------|-------------|----------|
| **Equal-width** | Divide continuous variables into equal intervals | Uniform distribution |
| **Equal-frequency** | Create bins with equal number of observations | Skewed distributions |
| **Custom** | Use domain knowledge for meaningful categories | Business-driven categories |

#### 🚩 Boolean/Flag Features

- ✅ Convert conditions into binary indicators
- ✅ Create threshold-based features for anomaly detection  
- ✅ Generate status flags based on business logic

#### 🔗 Feature Interactions

```python
# Creating interaction features
data['feature1_x_feature2'] = data['feature1'] * data['feature2']  # Multiplication
data['feature1_plus_feature2'] = data['feature1'] + data['feature2']  # Addition
data['feature1_ratio'] = data['feature1'] / (data['feature2'] + 1e-8)  # Ratio
```

### Advanced Feature Engineering

#### 📐 Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[['feature1', 'feature2']])
```

> ⚠️ **Warning**: Use with caution to avoid overfitting

#### 📊 Statistical Feature Engineering

```python
# Rolling statistics for time series
data['rolling_mean'] = data['value'].rolling(window=7).mean()
data['rolling_std'] = data['value'].rolling(window=7).std()

# Group aggregations
group_stats = data.groupby('category')['value'].agg(['mean', 'std', 'min', 'max'])

# Lag features
data['lag_1'] = data['value'].shift(1)
data['lag_7'] = data['value'].shift(7)
```

#### 🎯 Domain-Specific Techniques

| Domain | Techniques | Tools |
|--------|------------|-------|
| **Text Data** | TF-IDF, word embeddings, n-grams, POS tagging | `nltk`, `spacy`, `transformers` |
| **Time Series** | Seasonal decomposition, frequency domain features | `statsmodels`, `tsfresh` |
| **Image Data** | Texture features, edge detection, color histograms | `opencv`, `skimage` |

---

## 🎯 Feature Selection Techniques

### 🔍 Filter Methods

```python
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest, VarianceThreshold

# Univariate tests
selector = SelectKBest(score_func=chi2, k=10)
selected_features = selector.fit_transform(X, y)

# Variance thresholding
var_threshold = VarianceThreshold(threshold=0.1)
X_filtered = var_threshold.fit_transform(X)
```

### 🔄 Wrapper Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Forward Selection** | Incrementally add features | Simple, interpretable | Computationally expensive |
| **Backward Elimination** | Recursively remove features | Considers feature interactions | May remove important features early |
| **Recursive Feature Elimination** | Use model coefficients for selection | Model-specific selection | Requires model training |

### 🏗️ Embedded Methods

```python
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

# L1 Regularization (Lasso)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]

# Tree-based importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
feature_importance = rf.feature_importances_
```

---

## 📉 Dimensionality Reduction Techniques

### 📏 Linear Methods

```python
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Principal Component Analysis
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X)

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
```

### 🌀 Non-linear Methods

```python
from sklearn.manifold import TSNE
import umap

# t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP for dimensionality reduction
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)
```

---

## 🤖 Automated Feature Engineering

### 🛠️ Popular Tools and Libraries

#### 🔧 Featuretools
```python
import featuretools as ft

# Automated feature synthesis
feature_matrix, feature_defs = ft.dfs(entityset=es, 
                                     target_entity="customers",
                                     max_depth=2)
```

- ✅ Uses Deep Feature Synthesis (DFS) for automated feature creation
- ✅ Handles temporal and relational datasets
- ✅ Generates features through primitive stacking

#### 🎯 AutoFeat
- ✅ Automatically synthesizes non-linear features (log, x², x³)
- ✅ Performs feature selection using L1 regularization
- ✅ Iterative process of feature generation and selection

#### ⏰ TSFresh
```python
from tsfresh import extract_features

# Time series feature extraction
extracted_features = extract_features(timeseries_df, 
                                    column_id="id", 
                                    column_sort="time")
```

#### 🏭 Feature Engine
- ✅ Advanced encoding methods (RareLabelEncoder)
- ✅ Smart correlation-based selection
- ✅ Compatible with scikit-learn pipelines

### 🎁 Benefits of Automation

| Benefit | Description |
|---------|-------------|
| **⚡ Efficiency** | Scales feature engineering across multiple projects |
| **🧠 Bias Reduction** | Removes human preconceptions and errors |
| **🔄 Consistency** | Standardized approach across teams and models |
| **🔍 Exploration** | Discovers complex relationships not immediately obvious |

---

## 📝 Text Data Preprocessing and Feature Engineering

### Basic Text Preprocessing

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Basic preprocessing pipeline
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    
    # Stemming/Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)
```

### Advanced Text Processing

```python
import spacy

# Advanced NLP pipeline
nlp = spacy.load("en_core_web_sm")

def advanced_text_processing(text):
    doc = nlp(text)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # POS tagging
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    # N-grams
    bigrams = [doc[i:i+2].text for i in range(len(doc)-1)]
    
    return entities, pos_tags, bigrams
```

### Text Feature Engineering

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# TF-IDF Features
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
tfidf_features = tfidf.fit_transform(text_data)

# Bag of Words
bow = CountVectorizer(max_features=1000)
bow_features = bow.fit_transform(text_data)
```

---

## ✅ Best Practices and Considerations

### 🎯 Algorithm-Dependent Preprocessing

| Algorithm Type | Preprocessing Needs | Key Considerations |
|----------------|--------------------|--------------------|
| **Tree-based** | Minimal preprocessing | Handle raw features well |
| **Distance-based** | Scaling, normalization | Sensitive to feature scales |
| **Deep Learning** | Basic cleaning | Learn representations automatically |

### 🚫 Data Leakage Prevention

> ⚠️ **Critical**: Prevent data leakage to ensure model validity

- ❌ Avoid using future information for feature engineering
- 📅 Maintain temporal order in time series data
- 🔒 Separate feature engineering for train/validation/test sets

### 📊 Evaluation and Validation

```python
from sklearn.model_selection import cross_val_score

# Evaluate feature engineering impact
def evaluate_features(X, y, model):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean(), scores.std()

# Compare before and after feature engineering
original_score = evaluate_features(X_original, y, model)
engineered_score = evaluate_features(X_engineered, y, model)
```

### 💻 Computational Considerations

| Aspect | Recommendation |
|--------|----------------|
| **Feature Richness** | Balance with computational efficiency |
| **Memory Usage** | Consider memory requirements for high-dimensional data |
| **Production** | Optimize preprocessing pipelines for deployment |

---

## 🎉 Conclusion

This comprehensive progression from basic data cleaning to advanced automated feature engineering provides a complete toolkit for data preprocessing. The choice of techniques depends on:

- 📊 **Data characteristics**
- 🏢 **Problem domain** 
- 💻 **Computational resources**
- 🤖 **Specific machine learning algorithms being employed**

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing tools and libraries
- Special thanks to contributors of scikit-learn, pandas, and other ML libraries

---

⭐ **Star this repository** if you found it helpful!
