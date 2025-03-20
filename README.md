# Yelp-Rating-Prediction


## Overview
This project analyzes Yelp business data to predict restaurant ratings using multiple linear regression. The goal is to understand which factors contribute to higher ratings and how well a model can estimate a restaurant’s star rating based on available features.

## Dataset
The dataset consists of multiple JSON files from Yelp, including:
- `yelp_business.json` – Business details such as name, location, category, and rating.
- `yelp_checkin.json` – Check-in counts over time.
- `yelp_photo.json` – Photos and captions associated with businesses.
- `yelp_review.json` – User reviews including text and ratings.
- `yelp_tip.json` – Tips left by users.
- `yelp_user.json` – User metadata, including total review count and votes.

## Feature Selection
The following features were selected for training based on their correlation with the target variable (restaurant rating):
- `average_review_sentiment`
- `average_review_length`
- `review_count`
- `number_useful_votes`
- `weekday_checkins`
- `weekend_checkins`
- `price_range`
- `take_reservations`

## Model Training
A multiple linear regression model was used:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
lm = LinearRegression()
lm.fit(X_train, y_train)
```

## Model Evaluation
The model was evaluated using the R² score:
```python
r2_score = lm.score(X_test, y_test)
print("R² Score:", r2_score)
```
A scatter plot was also used to visualize actual vs. predicted ratings.

## Results
The final model achieved an R² score of approximately **0.66**, indicating a moderate level of predictive accuracy.

## How to Run the Project
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
   ```
2. Place the Yelp JSON files in the project directory.
3. Run the script:
   ```bash
   python yelp_rating_prediction.py
   ```

## Future Improvements
- Incorporate additional features to improve accuracy.
- Experiment with feature engineering techniques.
- Analyze residual errors to identify patterns in prediction mistakes.

## Repository Structure
```
├── yelp_business.json
├── yelp_checkin.json
├── yelp_photo.json
├── yelp_review.json
├── yelp_tip.json
├── yelp_user.json
├── yelp_rating_prediction.py
├── cleaned_yelp_data.csv
├── README.md
```

## Conclusion
This project demonstrates how linear regression can be used to predict restaurant ratings on Yelp. While the model provides moderate accuracy, further refinements could improve its predictive power.

