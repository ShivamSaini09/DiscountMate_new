# Discount Prediction Artifacts

This branch stores local reference datasets and generated model artifacts for the discount-price prediction work. It is intentionally separate from the pull-request branch and is not intended to be merged into the main repository.

Included artifacts:

- `all_catalogue_products.csv` — source catalogue extraction.
- `discount_price_final_dataset.csv` — generated weekly modelling dataset.
- `discount_special_model.joblib` — standalone next-week special classifier.
- `discount_price_prediction_model.joblib` — combined special classifier and discount-price regressor.

The model files were generated and verified with Python 3.14, scikit-learn 1.9.0 and XGBoost 3.4.1. Joblib models should be loaded with compatible dependency versions.

The notebooks and documentation remain on the `discount-price-prediction` branch, which is the branch intended for code review.
