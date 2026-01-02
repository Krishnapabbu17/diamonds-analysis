# Diamond Cut Quality Classification using Machine Learning

## Overview
Diamond cut quality plays a major role in a diamond’s brilliance and value, yet grading cut quality can be subjective.  
This project explores whether **machine learning models** can accurately classify a diamond’s cut as **High** or **Low** using measurable physical and quality attributes.

We frame cut quality as a **binary classification problem**:
- **High**: Ideal, Premium  
- **Low**: Very Good, Good, Fair  

Using a dataset of ~54,000 round-cut diamonds, we compare the performance of **Random Forest** and **XGBoost** models and analyze how different features contribute to cut classification.

📄 Full analysis and results are documented in the notebooks and written report 

---

## Dataset
- **Source:** Public “Diamonds” dataset
- **Size:** 53,940 diamonds (after cleaning ~53,900)
- **Diamond Shape:** Round cut only

### Original Features
- Carat weight  
- Cut grade  
- Color grade (D–J)  
- Clarity grade (IF–I1)  
- Depth percentage  
- Table percentage  
- Dimensions (x, y, z in mm)  
- Price (USD)

---

## Data Cleaning & Preprocessing
Key preprocessing steps:
- Removed rows with **invalid dimensions** (x, y, or z = 0)
- Removed extreme outliers in dimensions
- Re-encoded ordinal categorical features (color, clarity)
- Converted cut grade into **binary target** (`High` / `Low`)
- Dropped price and raw dimensions from modeling to focus on **physical quality features**, not economic outcomes

### Final Model Features
- `carat_weight`
- `depth_percent`
- `table_percent`
- `clarity_grade`
- `color_grade`

---

## Exploratory Data Analysis
EDA revealed clear differences between High and Low cut diamonds:
- **Low-cut diamonds tend to have higher carat weights**
- High-cut diamonds cluster closer to ideal depth and table proportions
- Better cut quality is associated with higher clarity and color grades
- Strong **class imbalance** (High cut is the majority class)

Statistical tests confirmed feature relevance:
- **t-tests** for numeric features (all p < 0.05)
- **Chi-square tests** for categorical features (color & clarity)

---

## Models
### 1. Random Forest
- 200 trees
- Manual **class weighting** to address imbalance
- Stratified train/test split (80/20)

**Performance**
- Accuracy: **~93.5%**
- High-cut recall: **95%**
- Low-cut recall: **81%**
- Low-cut F1 score: **~75%**

---

### 2. XGBoost
- Hyperparameter tuning via `RandomizedSearchCV`
- Imbalance handled with `scale_pos_weight`

**Performance**
- Accuracy: **~95.1%**
- High-cut recall: **98%**
- Low-cut recall: **75%**
- Low-cut F1 score: **~79%**

---

## Results & Evaluation
- Both models perform strongly on the **High-cut majority class**
- XGBoost slightly outperforms Random Forest overall
- Random Forest captures more Low-cut diamonds (higher recall)
- XGBoost provides better precision for Low-cut diamonds
- ROC curves and normalized confusion matrices confirm robust discrimination

---

## Key Findings
- Diamond cut quality can be predicted with **~95% accuracy** using basic physical and quality features
- **Carat weight** is the most influential feature
- High-cut diamonds often sacrifice carat weight to achieve better proportions
- Class imbalance remains the biggest modeling challenge

---

## Limitations
- Dataset includes **only round-cut diamonds**
- Important cut features (crown angle, pavilion depth, symmetry, polish) are not available
- Minority class (Low cut) is harder to classify accurately

---

## Future Work
- Address class imbalance using **SMOTE or resampling**
- Incorporate advanced geometric cut parameters
- Extend model to other diamond shapes
- Experiment with ensemble or stacking models
- Add explainability (SHAP feature importance)

---

## Technologies Used
- Python
- pandas, NumPy
- scikit-learn
- XGBoost
- matplotlib, seaborn
- Google Colab

---

## Author
**Krishna Pabbu**  
Machine Learning & Data Science Project

---

## Repository Structure
