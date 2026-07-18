---
layout: post
title: "Modelling Football Matches with the Poisson Distribution"
date: 2026-07-18
author: Kibum
tags: [football, python, modelling, poisson, betting]
categories: modelling python
---

Football is a low-scoring sport, which makes it an unusually good fit for one of the simplest tools in a statistician's kit: the Poisson distribution. It solves a very specific problem — how do you turn team strength into a probability distribution over goals, and then into probabilities for match outcomes? It may not the best model available today, but it's certainly the right place to start, because everything more sophisticated is really just an improvement on top of it.

## The core idea

The basic assumption behind a Poisson goals model is that each team scores goals according to a Poisson process: goals happen independently of each other over the 90 minutes of a match, at a roughly constant average rate. If a two teams' average scoring rate — its expected goals, or _lambda_ — is known, the Poisson distribution tells you the probability of scoring exactly 0, 1, 2, 3, or more goals in the match between them.

That assumption is a massive simplification, and it's worth knowing where it can break:

- **Independence is questionable.** A team that goes 2-0 up often changes its approach — sitting deeper, taking fewer risks — which changes the scoring rate mid-match. Goals are not truly independent events.
- **The scoring rate isn't fixed.** In-game variables such as injuries, substitutions and red cards can shift the effective average scoring rate during a match, but the basic model treats it as constant.
- **Low-scoring outcomes are correlated.** In practice, 0-0 and 1-1 draws happen slightly more often than a naive independent Poisson model predicts. This is a well-known limitation, and it's the reason the Dixon-Coles extension of the model, more on that later.

Despite these issues, the Poisson model captures the first-order structure of football scoring remarkably well. It gives us a coherent, testable, and interpretable baseline — which is exactly what we want before adding complexity.

## Model structure

The standard approach, going back to [Maher's 1982 paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9574.1982.tb00782.x) and later refined by [Dixon and Coles](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9876.00065), decomposes each team's expected goals into three components: **attack strength**, **defense strength**, and **home advantage**.

For a match between a home team H and away team A, the expected goals are:
expected_goals_home = attack_H * defense_A * home_advantage
expected_goals_away = attack_A * defense_H

Here, `attack_H` represents how many goals the home team tends to score against an average opponent, and `defense_A` represents how many goals the away team tends to concede against an average opponent. A team with a high (strong) attack rating and an opponent with a high (weak) defense rating combine to produce high expected goals. Home advantage is typically fit as a single multiplicative constant applied to the home team's scoring rate — though in more advanced versions it can vary by league or even by team.

These attack/defense/home_advantage parameters are usually estimated by technique called maximum likelihood estimation (MLE), fitting attack and defense ratings for every team simultaneously so that the model's predicted goal distributions best match historical results.

## The data

For this post I used the **[International Football Results from 1872 to 2024](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)** dataset on Kaggle, which contains over 45,000 international match results with match date, home team, away team, and full-time score. All credit for data collection goes to the original dataset author on Kaggle.

Dataset looks like this.

| date       | home_team   | away_team   |   home_score |   away_score | tournament     | city            | country       | neutral   |
|:-----------|:------------|:------------|-------------:|-------------:|:---------------|:----------------|:--------------|:----------|
| 2026-07-11 | Argentina   | Switzerland |            3 |            1 | FIFA World Cup | Kansas City     | United States | True      |
| 2026-07-14 | France      | Spain       |            0 |            2 | FIFA World Cup | Arlington       | United States | True      |
| 2026-07-15 | England     | Argentina   |            1 |            2 | FIFA World Cup | Atlanta         | United States | True      |


## Python implementation

Below is a working implementation using `statsmodels`' Poisson regression, which is a clean way to estimate the attack/defense parameters without writing a custom MLE routine from scratch.

### Step 1: Prepare the data

The trick to fitting attack/defense strength with a standard Poisson regression is to reshape the data so that each match contributes two rows — one for the home team's goals, one for the away team's goals — with team and opponent as categorical predictors.

```python
import pandas as pd
import numpy as np

# Example historical results dataset
# columns: home_team, away_team, home_goals, away_goals
matches = pd.read_csv("results.csv")

# Reshape into long format: one row per team performance
home_df = matches.rename(
    columns={"home_team": "team", "away_team": "opponent", "home_goals": "goals"}
)[["team", "opponent", "goals"]]
home_df["is_home"] = 1

away_df = matches.rename(
    columns={"away_team": "team", "home_team": "opponent", "away_goals": "goals"}
)[["team", "opponent", "goals"]]
away_df["is_home"] = 0

model_df = pd.concat([home_df, away_df], ignore_index=True)
```

### Step 2: Fit the Poisson regression

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

poisson_model = smf.glm(
    formula="goals ~ is_home + team + opponent",
    data=model_df,
    family=sm.families.Poisson()
).fit()

print(poisson_model.summary())
```

This fits a single model where `team` captures attack strength, `opponent` captures the defense strength conceded against, and `is_home` captures the home advantage effect — all jointly, which is the practical equivalent of the classic Maher decomposition.

### Step 3: Predict expected goals for a new fixture

```python
def predict_expected_goals(home_team, away_team, model):
    home_input = pd.DataFrame({
        "team": [home_team], "opponent": [away_team], "is_home":
    })
    away_input = pd.DataFrame({
        "team": [away_team], "opponent": [home_team], "is_home": 
    })

    home_xg = model.predict(home_input).iloc
    away_xg = model.predict(away_input).iloc
    return home_xg, away_xg

home_xg, away_xg = predict_expected_goals("Team A", "Team B", poisson_model)
print(f"Expected goals — Home: {home_xg:.2f}, Away: {away_xg:.2f}")
```

### Step 4: Build the scoreline probability matrix

```python
from scipy.stats import poisson

def scoreline_matrix(home_xg, away_xg, max_goals=8):
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    matrix = np.outer(home_probs, away_probs)
    return matrix

matrix = scoreline_matrix(home_xg, away_xg)
```

The matrix's rows represent home goals, columns represent away goals, and each cell is the joint probability of that exact scoreline, assuming independence between the two teams' scoring processes.

### Step 5: Derive match outcome probabilities

```python
def outcome_probabilities(matrix):
    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))
    return {"home_win": home_win, "draw": draw, "away_win": away_win}

probs = outcome_probabilities(matrix)
print(probs)
```

This gives us home win, draw, and away win probabilities directly from the scoreline grid — the same quantities that bookmakers price into match odds.

## Real worked example: 2026 World Cup

Now to the fun bit - let's predict some outcomes! I'm writing this post on the 18th of July, with only two matches left to be played. Spain and Argentina fight for the championship, while France and England play the third-place match. Therefore, we will predict the outcome of the two matches using the Poisson goals model. With more than 100 matches played in the 2026 FIFA World Cup, there's a rich amount of up-to-date data we can use to fit our model.

### Data cut-off
The international football results dataset contains results from 1872, so you can imagine a large chunk of the dataset being less relevant to the prediction of the world cup matches in 2026. Therefore, we will cut off the dataset to use the international match results from 2022 until now. I chose this timeframe as national teams tend to build up the team from one major tournament to the next, often introducing changes of players, coaching staff and the playing style.

```python
cutoff_df = model_df[model_df['date'] > '2022-12-18']
```

### Predicting outcomes
Let's try fitting the model and predict the outcomes of the two matches. We first fit the Poisson model to generate attack and defense parameters of each team.

```python
Model fitted on 3730 matches. Coefficients:
  Spain: Attack 2.743, Defense -2.097
  Argentina: Attack 2.550, Defense -2.325
  France: Attack 2.573, Defense -2.000
  England: Attack 2.454, Defense -2.008
```
For example, France's parameters mean that their attacking strength adds on average 2.573 goals to their scoreline, while their defensive strength decreases 2 goals from the opponent's average attacking strength.

Then we use the fitted Poisson parameters to calculate the probabilities of each team's scores.
```Python
Scoreline probability matrix (rows = home goals, columns = away goals):
[[0.14375475 0.14187304 0.07000799 0.02303053 0.00568227]
 [0.13696036 0.13516759 0.06669915 0.02194203 0.0054137 ]
 [0.06524355 0.06438953 0.03177335 0.01045248 0.00257892]
 [0.02071997 0.02044875 0.01009054 0.00331949 0.00081901]
 [0.00493517 0.00487057 0.00240341 0.00079065 0.00019507]]
```
The rows show the probability of Spain's scores and the columns show that of Argentina's. For example, the model predicts that there is a 0.14187304 chance that the score will end 0-1.

Finally, we can add the probabilities of scorelines to derive the full home win, draw, away win probabilities.

```Python
Expected goals (neutral venue) — Spain: 0.95, Argentina: 0.99
Match outcome probabilities:
  home_win: 33.09%
  draw: 31.42%
  away_win: 34.85%

Expected goals (neutral venue) — France: 1.10, England: 0.99
Match outcome probabilities:
  home_win: 37.42%
  draw: 29.95%
  away_win: 31.74%
```

Our Poisson model thinks that Argentina will become back-to-back world champions, while France will save face by defeating England in the third-place match.

The full implementation, including the `PoissonGoalsModel` class, is available here:
[`poisson_model.py`](https://github.com/kvkwon/kvkwon.github.io/blob/main/code/poisson_model/poisson_model.py)

## Validation and limitations

The Poisson goals model is based on a lot of simplified assumptions. Let's be explicit about what this model does *not* capture:

- **No time decay.** A match from three seasons ago is weighted the same as one from last week, which is rarely appropriate given squad and form changes.
- **Independence bias.** As mentioned, low-scoring outcomes like 0-0 and 1-1 tend to be slightly underestimated because real goals aren't perfectly independent.
- **No player-level information.** Injuries, suspensions, and squad rotation are invisible to this model entirely.
- **Static home advantage.** Treating home advantage as a single constant across all teams ignores real variation — some teams have a much stronger home effect than others.

None of this makes the model useless. It makes it a *starting point* — a transparent, testable baseline you can measure improvements against.

One useful sanity check is comparing the model's implied probability outcomes against bookmaker odds (converted from odds to probabilities and de-vigged, which means to adjust the probabilities for the bookmaker's built-in profit margin). If the model is wildly out of line with the market on most matches, that's either an evidence that we've found a huge edge, or that something is off in your parameter estimation. Unfortunately, the latter usually holds, as markets are generally efficient enough that large, persistent discrepancies are more likely a bug than alpha.

## How to improve the model

- **Dixon-Coles adjustment**: add a low-score correlation correction to fix the underestimated probability of 0-0 and 1-1 draws.
- **Time-weighting**: apply exponential decay so recent matches carry more weight than older ones.
- **Team-specific home advantage**: instead of one global home constant, estimate it per team.
- **Rolling re-estimation**: refit the model on a rolling window rather than the full historical dataset, to better reflect current form.

## Closing

The point of walking through the Poisson goals model in full is not to claim it produces profitable betting picks — it doesn't, on its own. The point is to show the mechanics clearly enough that we can judge its assumptions, test its outputs against the market, and decide where it's useful and where it isn't. That's the standard I want every post on this blog to meet: show the method, show the code, show where it breaks, and let the evidence — not the confidence of the write-up — do the convincing.