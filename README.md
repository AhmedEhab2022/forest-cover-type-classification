# Forest Cover Type Classification Using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AhmedEhab2022/Forest-Cover-Type-Classification/blob/main/Forest_Cover_Type_Classification.ipynb)

A comprehensive machine learning project that predicts forest cover types based on cartographic variables using the UCI Forest Cover Type dataset. This project helps forest management and conservation efforts by accurately classifying forest cover types using environmental and geological features, enabling better resource allocation and ecological monitoring.

## Project Overview

This repository contains a complete forest cover type classification workflow that analyzes cartographic data to predict seven different forest cover types. The analysis compares multiple machine learning algorithms and implements hyperparameter tuning to achieve optimal classification performance for forestry applications.

### Algorithms Implemented

1. **Decision Tree Classifier**

   - Interpretable tree-based classification
   - Feature importance analysis
   - Hyperparameter optimization

2. **Random Forest Classifier**

   - Ensemble method with multiple decision trees
   - Robust performance with default parameters
   - Excellent for handling large datasets

3. **XGBoost Classifier**
   - Gradient boosting framework
   - Superior accuracy with tuned parameters
   - Advanced feature importance ranking

## Dataset

The project uses the UCI Forest Cover Type dataset (`covtype.data`) containing:

- **Dataset Size**: 581,012 observations with 54 features
- **Source**: US Forest Service (USFS) Region 2 Resource Information System (RIS)
- **Geographic Area**: Roosevelt National Forest, Colorado

### Feature Categories

| Feature Type             | Count | Description                                                  |
| ------------------------ | ----- | ------------------------------------------------------------ |
| **Continuous Variables** | 10    | Elevation, Aspect, Slope, Distance metrics, Hillshade values |
| **Wilderness Areas**     | 4     | Binary indicators for wilderness area designation            |
| **Soil Types**           | 40    | Binary indicators for different soil type classifications    |
| **Target Variable**      | 1     | Cover_Type (7 forest cover type classes)                     |

### Forest Cover Types

1. **Spruce/Fir** - High elevation coniferous forest
2. **Lodgepole Pine** - Fire-adapted pine species
3. **Ponderosa Pine** - Drought-tolerant pine forest
4. **Cottonwood/Willow** - Riparian woodland
5. **Aspen** - Deciduous forest type
6. **Douglas Fir** - Mixed coniferous forest
7. **Krummholz** - Stunted trees near treeline

## Methodology

### Data Preprocessing

- Comprehensive exploratory data analysis with visualizations
- Data cleaning and validation (duplicate/missing value checks)
- Feature correlation analysis and multicollinearity detection
- Class imbalance assessment and handling strategies

### Model Development

1. **Baseline Models**: Implementation with default parameters
2. **Hyperparameter Tuning**: Grid search optimization for optimal performance
3. **Cross-Validation**: Stratified sampling to maintain class distribution
4. **Feature Importance**: Analysis of most predictive cartographic variables

### Evaluation Metrics

- **Accuracy Score**: Overall classification performance
- **Confusion Matrix**: Detailed class-wise prediction analysis
- **Classification Report**: Precision, recall, and F1-score per class
- **Feature Importance**: Ranking of most influential variables

## Key Findings

### Model Performance Results

| Algorithm         | Accuracy      | Training Speed | Best Use Case                                    |
| ----------------- | ------------- | -------------- | ------------------------------------------------ |
| **XGBoost**       | **Highest**   | Moderate       | Best overall accuracy with hyperparameter tuning |
| **Decision Tree** | Good          | **Fastest**    | Best balance of speed and interpretability       |
| **Random Forest** | **Very Good** | Fast           | Excellent performance with default parameters    |

### Data Insights

- **Class Imbalance**: Significant variation in forest cover type distribution
- **Feature Importance**: Elevation and wilderness areas are top predictors
- **Data Quality**: Clean dataset with no missing values or duplicates
- **Correlations**: Strong relationships between hillshade features at different times

## Requirements

```txt
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
xgboost
jupyter
ipykernel
```

## Dataset Download

> **⚠️ Important**: The dataset file (`covtype.data`) is approximately **11MB** and is **not included** in this repository due to GitHub file size limitations.

### Download the Dataset:

**Option A - Direct Download:**

- Download from [UCI ML Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz)
- Extract the `.gz` file to get `covtype.data`
- Create `data/` folder
- Place it in the `data/` folder

**Option B - Command Line:**

```bash
# Create data directory
mkdir data

# Download and extract
curl -o data/covtype.data.gz https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz
gunzip data/covtype.data.gz
```

## How to Run

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. **Download the dataset** using the methods above OR upload to your Google Drive
3. Update the file path in the notebook if needed
4. Run all cells in sequence

### Option 2: Local Setup

1. Clone this repository: `git clone <repository-url>`
2. **Download the dataset** (see Dataset Download section above)
3. Install dependencies: `pip install -r requirements.txt`
4. Open `Forest_Cover_Type_Classification.ipynb` in Jupyter
5. Verify the data path points to your downloaded dataset
6. Run all cells

## Technologies Used

- **Python Libraries**: pandas, numpy, matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn (Decision Tree, Random Forest, GridSearchCV)
- **Advanced ML**: XGBoost for gradient boosting
- **Evaluation**: Classification metrics and confusion matrix analysis
- **Environment**: Jupyter Notebook

## Project Structure

```
forest-cover-type-classification/
├── Forest_Cover_Type_Classification.ipynb   # Main analysis notebook
├── data/                                    # Create it for the downloaded dataset files
│   ├── covtype.data                         # Dataset file (DOWNLOAD REQUIRED)
│   ├── covtype.info                         # Dataset information
│   └── old_covtype.info                     # Additional dataset info
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
└── .gitignore                               # Excludes large data files
```

> **📁 Note**: The `covtype.data` file must be downloaded separately (see Dataset Download section above)

## Key Visualizations

The project includes comprehensive visualizations:

- **Histograms**: Distribution analysis of elevation, aspect, and slope
- **Correlation Matrices**: Feature relationship heatmaps with plotly and seaborn
- **Feature Importance**: Ranking of most predictive cartographic variables
- **Confusion Matrices**: Detailed model performance evaluation
- **Class Distribution**: Cover type imbalance visualization
- **Soil Type Analysis**: Most common soil types across the dataset

## Forestry Applications

- **Forest Management**: Automated classification for large-scale forest inventory
- **Conservation Planning**: Identify areas needing protection based on cover types
- **Ecological Monitoring**: Track forest composition changes over time
- **Resource Allocation**: Optimize forestry operations based on cover type predictions
- **Climate Research**: Understand forest distribution patterns for climate studies

## Analysis Highlights

- **Optimal Performance**: XGBoost achieves highest accuracy with hyperparameter tuning
- **Efficiency Trade-off**: Decision Tree offers best speed vs accuracy balance
- **Feature Analysis**: Elevation and wilderness areas are most predictive features
- **Data Quality**: Clean dataset with 581K+ observations, no missing values
- **Hyperparameter Impact**: Grid search shows significant performance improvements

## Key Insights

1. **Elevation** is the most important predictor of forest cover type
2. **Wilderness area designation** significantly influences cover type distribution
3. **Class imbalance** exists but algorithms handle it well with stratified sampling
4. **Hillshade features** show temporal correlations that aid classification
5. **Soil types** provide additional discriminative power for certain cover types
6. **XGBoost** delivers superior performance but requires more computational resources

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Data Source

Dataset: [Forest Cover Type Dataset](https://archive.ics.uci.edu/ml/datasets/covertype) from UCI Machine Learning Repository
