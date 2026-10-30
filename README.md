# ML on the Restaurant Dataset

Three small machine-learning tasks on `Dataset.csv`, a table of about 9,500 restaurants (location,
cuisines, cost for two, table booking and delivery options, aggregate rating, votes).

| Notebook | Task | Approach |
|---|---|---|
| `Task 01_Predict Restaurant Ratings.ipynb` | Predict the aggregate rating | Label-encoded categoricals; Decision Tree and Random Forest regressors. Saved output: R^2 0.978 (tree) and 0.988 (forest). |
| `Task 02_Restaurant Recommendation System.ipynb` | Group similar restaurants for recommendations | K-Means on aggregate rating and price range for one country's restaurants. Silhouette scores are computed for k = 2 to 50, then a K-Means model is fitted. |
| `Task 03_ Cuisine Classification.ipynb` | Classification with Gaussian Naive Bayes | Saved output: 0.894 train and 0.878 test score. Despite the title, the code uses `Restaurant Name` as the target, with the remaining columns (including `Cuisines`) as features. |

## Running

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

The notebooks read `Dataset.csv` from this folder.

## Caveat

Task 01 uses every column except the rating (and latitude/longitude) as a feature, including
`Rating color` and `Rating text`, which are derived from the rating itself. Its very high R^2 most
likely reflects that leakage and should not be read as predictive skill.
