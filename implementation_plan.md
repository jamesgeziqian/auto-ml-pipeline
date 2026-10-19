# Auto Machine Learning Pipeline

This repo contains code for an auto machine learning pipeline, including steps of:
- type inference
- feature filtering
- preprocessing: impute nan values, one-hot encoding, etc.
- train and tune hyper parameters
- report metrics

## Data

This is also one of the biggest problem: feature data are stored in multiple csv files, each containing 32000+ columns and 50000 rows, and they should be concatenated by row to make a big dataset of 32000 columns, ideally; the label data are stored separately in another path, where the column name is configurable. The label data and the feature data should be joined into one by some key column name (feature names), which shall also be configurable.

The label data also has a lot of features already stoerd in the label file, and they should also be put into the model.

## Feature filtering

The filtering should at least be based on these metrics:

- nan ratio,
- number of unique values (categorical features),
- linear correlation (pearson),
- information value (IV value),
- population stability index (psi)

## Preprocessing

For all features, impute the nan values. The policy can be controled via config file.

For categorical features, transform to one-hot if needed.

## Training

Use lightGBM for the training, but modular this part so we can swap to other models easily by just implement another adapter for the new moddel.

## Parameter tuning

Use `optuna`'s tpe tuning. The search space should also be configurable via file, like `search_space_<model_name>.yaml, etc.

## Report metrics

At least the following metrics should be collected:

- accuracy
- precision
- F-1 score
- recall
- auc
- ks
- feature importance

## Working Environment

You can use `uv` to manage all the dependencies or use `conda` to create a virtualenv and install all the dependencies there.

## Performance

With 32000 columns and 50000 rows for taking 1 file, it would at least take 3-4GB on disk. If we're loading all the features and all rows into memory, it would easily exceed the 64GB memory limit. In the target env (remote server), the server admin set a rule that all processes that exceeds 64GB of RAM would get killed directly.

For this problem, we need to perform chunked read in both rows and columns and all the calculations (feature filtering, model training) shall adapt to this.

## Liveliness

Print info between each stage and small steps in them. Also pring a hearbeat so that we know it is running.
