# Predicting Startup Success with Twitter Traction

A data mining project that investigates whether **Twitter activity and engagement metrics can predict startup success** defined as advancing from the Inception stage to the next funding round.

Built for a real investment scenario: a fund with €1.25M capital and 5 investment slots.

## Business Problem

Early-stage investors face a fundamental challenge: at the Inception stage, startups have no revenue, no product-market fit, and no track record. How do you decide who to fund?

**Hypothesis:** Twitter traction, how actively a startup tweets and how much engagement it generates, is an indicator of real momentum in the market.

## Key Findings

| Finding | Detail |
|---|---|
| Strongest predictor | `has_twitter` — startups with a Twitter account succeed at **4x the rate** of those without |
| Best engagement signal | `reply_rate`, `retweet_ratio`, `engagement_ratio` |
| Surprising negative predictor | `tweet_length` (r = −0.306) — shorter tweets = more success |
| Model AUC | **0.81** — ranks a real success above a failure 81% of the time |
| Lift at top 5 | **5.09x** — top picks are 5x more likely to succeed than random |
| Fund profit (top 5 picks) | **€8,750,000** gross on €1.25M invested |

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Modelling:** Scikit-learn (Logistic Regression, Ridge, Lasso, Decision Tree)
- **Evaluation:** ROC/AUC, Lift Curve, Profit Curve, Confusion Matrix

## Project Structure

| Notebook | Description |
|---|---|
| `Predict_StartupSucces.ipynb` | Full pipeline: EDA → cleaning → modelling → business evaluation |

## Methodology

| Step | Description |
|---|---|
| 1. EDA | Explored Twitter vs non-Twitter startups separately to avoid distortion |
| 2. Cleaning | Engineered `has_twitter` binary flag, filled NaNs with 0 for inactive startups |
| 3. Feature Selection | Dropped raw counts in favour of normalised rate/ratio features to avoid multicollinearity |
| 4. Modelling | Compared Logistic Regression, Ridge, Lasso, and Decision Tree |
| 5. Business Evaluation | Built Profit Curve, ROC Curve, and Lift Curve to justify investment decisions |

## Business Evaluation

Beyond standard ML metrics, this project includes a full **investment decision framework:**

- **Profit Curve**: shows expected fund value at each funding threshold
- **ROC/AUC**: measures model reliability across all thresholds
- **Lift Curve**: shows how much better the model is vs random selection
- **Portfolio Selection**: picks the top 5 startups by predicted probability

The asymmetric cost structure (missing a unicorn costs as much as 9 failures) shaped the entire modelling strategy.

## Getting Started

**1. Clone the repository**
```bash
git clone git@github.com:claudia-moliner/startup-success-prediction.git
cd startup-success-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**
```bash
jupyter notebook notebooks/Predict_StartupSucces.ipynb
```

## Author

**Claudia Moliner** — Data Analyst
[GitHub](https://github.com/claudia-moliner) · [Email](mailto:cmolinercalvet@gmail.com)
