# Tanzanian Water Wells Classification Model

## Introduction

This is the process by which I generated a Random Forest Classifier ("RFC") model to classify water wells using data from both Taarifa and the Tanzanian Ministry of Water.

The water wells are divided up into three different categories, using the following labels: `Functional`, `Non-functional`, and `Functional, needs repair`. The goal is to be able to accurately predict which category each well falls into, based on a variety of factors, such as the kind of pump in operation, when it was installed, and the size of the surrounding population. Understanding which wells might fail will help the government and NGOs on the ground keep up with maintenance and ensure access to water to as many communities as possible.

What follows are my steps that lead me to my final model.

## Data Processing

### Column Selection
The training dataset for this problem had 40 different variable (not including the labels), but a handful of these columns were duplicates, or just different iterations of other columns. When I went through removing duplicative columns, I most often ended up keeping the column that had less factors. For instance, `extraction_type`, `extraction_type_group`, and `extraction_type_class` were very similar columns with 18, 13, and 7 factors, respectively. I ended up keeping the `extraction_type_class` column, as it had the fewest factors.

I also removed columns that had too many factors (like `wpt_name`, were proxies for other variables (like how `ward` and `lga` were proxies for `region`), were too specific to each well (the coordinates), or were a constant variable for the whole dataset (`recorded_by`). I also removed the `scheme_name` column, as almost half of the data was missing.

Now, ignoring both the `id` and `status_group` columns, I had 16 predictor columns.

### Cleaning Variables
With my columns selected, I removed all rows with `NA`s, which dropped about 5,000 rows. There were then a few columns that had "unknown" listed as entries, which I also removed. And finally, I replaced all the `0`s in the `construction_year` column with the median year.

### New Variables
I then went on to create a new variable, `days_since`, which used the `date_recorded` variable and counted the number of days prior to January 1st, 2014 it was last inspected (this date was chosen because none of the `date_recorded` values happened on or after January 1st, 2014).

I also needed to reduce the number of levels in both the `funder` and `installer` column.

## Modeling

With all my variables ready, I ran four different classifiers (Decision Tree, Random Forest, Logistic Regression, and XGBoost) using three differnt training sets (normal, upsampled, SMOTE). The differnt training sets were to account for the unbalance within the data. There were many more functional wells than unfunctional wells, and even more than wells in need of repair.

Below is an image with the overall accuracy and the recall and precision for the `Functional, needs repair` wells for each classifier:

<img src="Images-Used/Classifier Metrics.png" width = "500">

Looking through these metrics and cross-validation scores for all the different models generated, if I were to submit to the competition, I would submit the Random Forest Classifier (RFC) trained on the data balanced using the SMOTE technique. 

Looking at accuracy alone, all the RFCs performed better than any other classification method. Now of those three, even though the model trained on unbalanced data had a higher accuracy score, the model trained on SMOTE-balanced data maintained a relatively high accuracy score, while also improving on the recall score for our least-populated category `Functional-Needs Repair`. By improving the recall score, the model is correctly predicting a higher proportion of wells that are functional but need repair. Some of the other models have even higher recall scores for the "Functional-Needs Repair" category, but they lose out on the overall accuracy of the model.

The next step in this process would then be to adjust the hyperparameters for the `r_forest_smote` classifier to see if we can make it a little better.

## Hyperparameter Tuning

Having already established a base accuracy score of `0.779` for our RFC, using the SMOTE technique for balancing data, I went ahead with adjusting the hyperparameters of the model.

I performed a GridSearch on the RFC-SMOTE Classifier using the following parameters:

1. **'n_estimators'**: 10, 30, 100, 500
2. **'criterion'**: 'gini', 'entropy'
3. **'max_depth'**: None, 2, 6, 10
4. **'min_samples_split'**: 5, 10
5. **'min_samples_leaf'**: 3, 6

And the result of this run gave the following **"Optimal Parameters"**:

1. **'n_estimators'**: 100
2. **'criterion'**: 'gini'
3. **'max_depth'**: None
4. **'min_samples_split'**: 5
5. **'min_samples_leaf'**: 3

Using these optimal parameters, I generated another RFC model using the SMOTE-balanced data, which returned the following results.

<img src="Images-Used/Final RFC.png" width = "500">

After tuning our hyperparameters, our overall accuracy stayed the same, dropping only **0.1%** , giving our final model a **77.6%** accuracty. Though our accuracy stayed the same, our precision for the `Functional, needs repair` wells increased by **1.6** points and our recall for the same label increased by **3.6** points. It still isn't perfect, since almost half of the `Functional, needs repair` wells are mislabled as `Functional` wells, but it's better than where it was.

## Feature Importance

<img src="Images-Used/Feature Importance.png" width = "500">

Using the `feature_importances_` method, we can see that the top five important features in our model are:

1. `days_since` - the number of days before January 1st, 2014 the well was last inspected
2. `construction_year` - when the well was constructed
3. `population` - the surrounding population that uses the well
4. `quantity_enough` - whether or not there was enough water in the well
5. `extraction_type_class_other` - whether or not the well belongs to the `extraction_type_class` `other`

The first three features are all continuous variables, which the RFC model has a tendency to inflate the importance of, and the last two are dummy variables for the `quantity` and `extraction_type_class` categories.

After looking at the density plots for the top 5 most important variables, we're able to conclude the following:
1. `Non-functional` and `Functional wells that need repair` are more likely to have had an inspection closer to January of 2014 (most likely during 2013), while the `Functional` wells show equal probabilty of having last had their inspection in 2013 or a couple years earlier (most likely in 2011).
2. `Functional` wells are more likely to have been built in the year 2000 or later.
3. Wells with very large surrounding populations (outliers in the dataset) usually have `Functional` wells, while wells with lower surrounding populations, are likely to be `Non-functional` or `Functional, but in need of repair`.
4. `Functional wells` and `Functional wells that need repair` are more likely to have enough water in their aquifers, while `Non-functional` wells are more likely to not have enough water.
5. The last feature of importance is not too helpful, as all well types are less likely to be of the `extraction_type_class` `other`.

## Further Exploration: Data Reclassification

Currently, each well has one of the following labels:
1. Functional
2. Non functional
3. Functional, needs repair

I would argue that though the competition is looking for a model that most accurately classifies a well with one of these three labels, it might make more sense to create three different models (maybe even four). You could separate out the labels and run the following classification models in an attempt to get even more accurate results and better recall: 

1. `Functional` and `Non-functional`
2. `Functional` and `Functional, needs repair`
3. `Non-functional` and `Functional, needs repair`
4. `Functional` and `Non-functional`, where `Functional, needs repair` is relabeled as `Non-functional

I think the last one would be the most useful, and make the most sense, as one could argue that both a functional well in need of repair and a non-functional well would need to be identified for maintenance.