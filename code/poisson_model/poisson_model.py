"""
poisson_model.py

Simple Poisson-based football match prediction model.

Fits team attack/defense strength and home advantage using Poisson regression,
then derives expected goals, a full scoreline probability matrix, and match
outcome probabilities (home win / draw / away win) for a given fixture.

Neutral-venue matches (e.g. World Cup finals at a third-party venue) skip the
home advantage term entirely, since neither team benefits from home support.

Usage:
    python poisson_model.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson


class PoissonGoalsModel:
    """
    Poisson-based football match prediction model.

    Wraps the full pipeline — data loading, reshaping, fitting a Poisson
    regression for attack/defense/home-advantage, and deriving match
    outcome probabilities — behind a simple object interface.

    Neutral matches are supported: when neutral=True, is_home is set to 0
    for both teams, so no home advantage is applied to either side.

    Example
    -------
    >>> model = PoissonGoalsModel()
    >>> model.fit("results.csv")
    >>> home_xg, away_xg = model.predict_expected_goals("Team A", "Team B")
    >>> probs = model.predict_outcome_probabilities("Team A", "Team B", neutral=True)
    """

    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.model = None
        self.matches = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load historical match results.

        Expected columns: home_team, away_team, home_goals, away_goals
        Optional column: neutral (bool) — if present, used to correctly
        zero out home advantage for neutral-venue matches during fitting.
        """
        matches = pd.read_csv(filepath)
        required_cols = {"home_team", "away_team", "home_score", "away_score"}
        missing = required_cols - set(matches.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return matches

    def _reshape_to_long_format(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape match data into long format: one row per team performance,
        with an is_home indicator. This is what Poisson regression needs to
        estimate attack/defense/home-advantage jointly.

        When the neutral column is True, is_home is forced
        to 0 for both sides on neutral-venue matches, so the fitted home
        advantage coefficient reflects true home matches only.
        """

        home_df = matches.rename(
            columns={"home_team": "team", "away_team": "opponent", "home_score": "goals"}
        )[["team", "opponent", "goals"]]
        home_df["is_home"] = 1

        away_df = matches.rename(
            columns={"away_team": "team", "home_team": "opponent", "away_score": "goals"}
        )[["team", "opponent", "goals"]]
        away_df["is_home"] = 0

        neutral_mask = matches["neutral"].astype(bool)
        home_df.loc[neutral_mask.values, "is_home"] = 0

        return pd.concat([home_df, away_df], ignore_index=True)

    def fit(self, filepath: str) -> "PoissonGoalsModel":
        """
        Load data from filepath and fit the Poisson regression:
        goals ~ is_home + team + opponent.

        'team' captures attack strength, 'opponent' captures defense
        strength conceded against, and 'is_home' captures home advantage.

        Returns self, so calls can be chained, e.g.:
            model = PoissonGoalsModel().fit("results.csv")
        """
        self.matches = self.load_data(filepath)
        model_df = self._reshape_to_long_format(self.matches)

        self.model = smf.glm(
            formula="goals ~ is_home + team + opponent",
            data=model_df,
            family=sm.families.Poisson(),
        ).fit()
        return self

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call .fit(filepath) first.")

    def predict_expected_goals(
        self, home_team: str, away_team: str, neutral: bool = False
    ) -> tuple[float, float]:
        """
        Predict expected goals (lambda) for both teams in a given fixture.

        If neutral=True, home advantage is skipped for both teams — the
        'home_team' argument is used only to label which side is listed
        first in the output, not to grant a home-field boost.
        """
        self._check_fitted()

        home_is_home_flag = 0 if neutral else 1

        home_input = pd.DataFrame({
            "team": [home_team], "opponent": [away_team], "is_home": [home_is_home_flag]
        })
        away_input = pd.DataFrame({
            "team": [away_team], "opponent": [home_team], "is_home": [0]
        })

        home_xg = self.model.predict(home_input).iloc[0]
        away_xg = self.model.predict(away_input).iloc[0]
        return home_xg, away_xg

    def scoreline_matrix(self, home_xg: float, away_xg: float, max_goals: int = None) -> np.ndarray:
        """
        Build a full scoreline probability matrix assuming independent
        Poisson processes for home and away goals.

        Rows = home goals, columns = away goals.
        """
        max_goals = max_goals if max_goals is not None else self.max_goals
        home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
        return np.outer(home_probs, away_probs)

    def outcome_probabilities(self, matrix: np.ndarray) -> dict:
        """
        Derive home win / draw / away win probabilities from the scoreline matrix.
        """
        home_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        return {"home_win": home_win, "draw": draw, "away_win": away_win}

    def predict_outcome_probabilities(
        self, home_team: str, away_team: str, max_goals: int = None, neutral: bool = False
    ) -> dict:
        """
        Convenience method: predict expected goals for a fixture, build the
        scoreline matrix, and return home/draw/away win probabilities in
        one call.
        """
        home_xg, away_xg = self.predict_expected_goals(home_team, away_team, neutral=neutral)
        matrix = self.scoreline_matrix(home_xg, away_xg, max_goals=max_goals)
        return self.outcome_probabilities(matrix)

    def predict_fixture(
        self, home_team: str, away_team: str, max_goals: int = None, neutral: bool = False
    ) -> dict:
        """
        Full prediction for a single fixture: expected goals, scoreline
        matrix, and outcome probabilities, bundled into one dict.
        """
        home_xg, away_xg = self.predict_expected_goals(home_team, away_team, neutral=neutral)
        matrix = self.scoreline_matrix(home_xg, away_xg, max_goals=max_goals)
        probs = self.outcome_probabilities(matrix)

        return {
            "home_xg": home_xg,
            "away_xg": away_xg,
            "neutral": neutral,
            "scoreline_matrix": matrix,
            "outcome_probabilities": probs,
        }


if __name__ == "__main__":
    DATA_PATH = "code/poisson_model/cutoff_results.csv"
    HOME_TEAM = "Spain"
    AWAY_TEAM = "Argentina"
    NEUTRAL = True  # set True for a match played at a neutral venue

    model = PoissonGoalsModel(max_goals=4)
    model.fit(DATA_PATH)

    print(f"Model fitted on {len(model.matches)} matches. Coefficients:")
    for team in (HOME_TEAM, AWAY_TEAM):
        print(f"  {team}: Attack {model.model.params[f"team[T.{team}]"]:.3f}, Defense {model.model.params[f"opponent[T.{team}]"]:.3f}")

    result = model.predict_fixture(HOME_TEAM, AWAY_TEAM, neutral=NEUTRAL)

    venue_note = " (neutral venue)" if NEUTRAL else ""
    print(f"Expected goals{venue_note} — {HOME_TEAM}: {result['home_xg']:.2f}, "
          f"{AWAY_TEAM}: {result['away_xg']:.2f}\n")

    print("Match outcome probabilities:")
    for outcome, prob in result["outcome_probabilities"].items():
        print(f"  {outcome}: {prob:.2%}")

    print("\nScoreline probability matrix (rows = home goals, columns = away goals):")
    print(result["scoreline_matrix"])
