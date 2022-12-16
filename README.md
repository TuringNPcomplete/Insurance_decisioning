# Insurance Claim Outcome Prediction

This project looks at a practical insurance question: can we predict whether a customer will file a claim based on basic profile and policy-related information?

The goal was not only to get a strong score, but also to show a clear, thoughtful workflow and then test whether the result still holds up under a more defensible evaluation setup.

## Project Snapshot

- Problem type: binary classification
- Target variable: `insuranceclaim`
- Raw dataset size: 1,338 rows
- Final modeling dataset: 1,276 rows and 14 columns
- Best baseline benchmark: Extreme Gradient Boosting (XGBoost)
- Best baseline result: 93.8% accuracy, 93.7% F1-score, 98.3% ROC-AUC
- Strongest train-first re-check: XGBoost without the medical billing amount feature
- Most reliable final result: 95.15% accuracy, 95.15% F1-score, 98.27% ROC-AUC

## Why This Matters

In an insurance environment, a model like this can help teams spot likely claim outcomes earlier, support risk review, and guide where to focus attention. It is not a replacement for underwriting or claims judgment, but it can be a useful decision-support tool.

## Dataset

The dataset used here is the Sample Insurance Claim Prediction Dataset, adapted from the Medical Cost Personal Datasets.

It includes variables such as:

- age
- sex
- BMI
- smoking status
- region
- number of children
- medical charges
- claim outcome

## Preprocessing and Training steps

The notebook follows a simple end-to-end workflow:

1. Loaded the dataset and checked its shape, data types, unique values, and summary statistics.
2. Explored the target variable and feature distributions to understand class balance and data spread.
3. Removed duplicates and handled outliers.
4. Encoded categorical features so the models could use them.
5. Standardized the feature set before training.
6. Balanced the target classes with SMOTE.
7. Reviewed feature relationships and tested feature reduction ideas.
8. Trained and compared eight classification models.
9. Re-ran the strongest model in a train-first setup, compared performance with and without the medical billing amount feature, and reviewed subgroup and error patterns.

## Data Preparation Highlights

- The raw data started with 1,338 rows.
- One duplicate row was removed during cleaning.
- Outlier handling reduced the working dataset from 1,337 rows to 1,190 rows.
- Before resampling, the target split was 638 claim cases vs 552 non-claim cases.
- After SMOTE, both classes were balanced at 638 each.
- The final modeling dataset contained 1,276 rows and 14 columns.

## Visual Summary

### Target Balance Before Resampling

![Target distribution](assets/target-distribution.png)

The target was only slightly imbalanced at the start, but still uneven enough to justify balancing before model comparison.

### Final Dataset After Cleaning and Resampling

![Final dataset samples](assets/final-dataset-samples.png)

Most of the data was retained. About 9.8% of rows were removed during cleanup, while 5.7% were added through SMOTE to balance the classes.

### Model Comparison

![Model comparison heatmap](assets/model-comparison-heatmap.png)

This heatmap gives a quick view of how each model performed across accuracy, precision, recall, F1-score, and ROC-AUC.

### Feature Importance From the Best Baseline Model

![XGBoost feature importance](assets/xgb-feature-importance.png)

In the baseline XGBoost model, smoking status, the encoded children features, BMI, age, and the medical billing amount feature carried much of the predictive signal.

## Model Results

The notebook compares eight models. The table below summarizes the main results.

| Model | Accuracy | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic Regression | 81.2% | 81.3% | 86.0% |
| Decision Tree | 88.7% | 88.7% | 92.6% |
| Random Forest | 90.6% | 90.6% | 97.2% |
| Naive Bayes | 68.8% | 68.7% | 79.0% |
| Support Vector Machine | 84.4% | 84.4% | 90.5% |
| K-Nearest Neighbors | 83.6% | 83.6% | 91.1% |
| Gradient Boosting | 92.6% | 92.6% | 96.7% |
| Extreme Gradient Boosting | 93.8% | 93.7% | 98.3% |

## Robustness Re-check

The original leaderboard is useful, but the more important question is whether the result still looks strong when the train/test split happens before SMOTE and preprocessing is fit on training data only.

| Setup | Accuracy | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: |
| XGBoost with medical billing amount feature | 91.42% | 91.47% | 96.96% |
| XGBoost without medical billing amount feature | 95.15% | 95.15% | 98.27% |

In this re-check, removing the medical billing amount feature improved F1-score by 3.68 points and ROC-AUC by 1.31 points. That makes the no-billing-feature version the more credible final result in this project.

## Segment Review

- Non-smokers reached 95.14% F1-score and 95.15% recall, while smokers dropped to 83.40% F1-score and 79.03% recall.
- The strongest BMI segment was up to 25, while the `Above 30` group fell to 87.27% F1-score and 86.71% recall.
- The weakest age segment in this split was 31 to 45, with 85.58% F1-score and 85.71% recall.

## Error Analysis

- The model got 245 predictions correct, with 19 false negatives and 4 false positives.
- Most mistakes were false negatives, so the bigger risk here is missing a true claimant rather than incorrectly flagging a non-claimant.
- The largest error pocket was the smoker plus `Above 30` BMI subgroup, which accounted for 12 of the 19 false negatives.

## Main Takeaways

- Tree-based ensemble models were the strongest group in the original model comparison.
- XGBoost led the first-pass benchmark, but the more useful conclusion came from the train-first re-check.
- The model remained strong after removing the medical billing amount feature, which suggests that field was not helping generalization in this setup.
- The more important remaining weakness is lower recall for smoker and higher-BMI segments, not overall score alone.
- Logistic Regression was still a reasonable benchmark, while Naive Bayes was the weakest model in the original comparison.

## Notes and Next Steps

This notebook is a strong exploratory case study, but there are a few things I would tighten in a production version:

- move preprocessing and resampling into a single training pipeline
- validate the workflow with stricter cross-validation
- test whether segment-specific thresholding or class weighting can improve recall for smokers and higher-BMI policyholders
- add threshold tuning based on business costs for false positives and false negatives

## Repository Contents

- `insurance-claim-prediction.ipynb`: full notebook analysis
- `Insurance.csv`: dataset used in the notebook
- `assets/`: exported plots used in this README