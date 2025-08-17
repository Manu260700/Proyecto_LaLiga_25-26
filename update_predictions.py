"""
Script to update LaLiga match predictions using a simple Poisson model.

This script reads historical data for the Spanish LaLiga from multiple seasons,
optionally incorporates results from the current season, computes attack and
defensive strengths for each team and then predicts the outcome of upcoming
fixtures. The results are written to a JSON file which can be consumed by
a simple static website.

Usage:
    python update_predictions.py

The script assumes the following directory structure:

    predictions_site/
      ├── data/
      │     ├── season-2223.csv
      │     ├── season-2324.csv
      │     ├── season-2425.csv
      │     ├── current_season.csv   # optional, same format as other seasons
      │     └── fixtures.csv         # upcoming fixtures with HomeTeam,AwayTeam columns
      └── predictions.json

If ``current_season.csv`` is present, its results will be included with a
weight equal to the most recent season plus one. For example, if the latest
historical season weight is 3 (as in 2024–25), the current season will be
weighted 4. This allows the model to adapt as new matches are played.

The output ``predictions.json`` contains a list of dictionaries with the
expected number of goals for each team and the most probable scoreline
computed from a Poisson distribution truncated at 6 goals.

"""

import json
import math
import os
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def compute_team_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute relative attack/defence metrics and other statistics for each team.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing match results with columns: 'HomeTeam', 'AwayTeam',
        'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC'.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Mapping from team name to a dictionary of statistics. Keys include
        attack and defence strengths (home and away) relative to league
        averages, shot, shots on target and corner statistics, and league
        ranking.
    """
    league_home_goals = df['FTHG'].mean()
    league_away_goals = df['FTAG'].mean()
    league_home_shots = df['HS'].mean()
    league_away_shots = df['AS'].mean()
    league_home_sot = df['HST'].mean()
    league_away_sot = df['AST'].mean()
    league_home_corners = df['HC'].mean()
    league_away_corners = df['AC'].mean()

    team_stats: Dict[str, Dict[str, float]] = {}
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    for team in teams:
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]
        # Goals for and against
        goals_scored_home = home_matches['FTHG'].sum()
        goals_scored_away = away_matches['FTAG'].sum()
        goals_conceded_home = home_matches['FTAG'].sum()
        goals_conceded_away = away_matches['FTHG'].sum()
        # Shots and corners
        shots_home = home_matches['HS'].sum()
        shots_away = away_matches['AS'].sum()
        shots_conceded_home = home_matches['AS'].sum()
        shots_conceded_away = away_matches['HS'].sum()
        sot_home = home_matches['HST'].sum()
        sot_away = away_matches['AST'].sum()
        sot_conceded_home = home_matches['AST'].sum()
        sot_conceded_away = away_matches['HST'].sum()
        corners_home = home_matches['HC'].sum()
        corners_away = away_matches['AC'].sum()
        corners_conceded_home = home_matches['AC'].sum()
        corners_conceded_away = away_matches['HC'].sum()
        # Points
        points = 0
        for _, row in home_matches.iterrows():
            if row['FTHG'] > row['FTAG']:
                points += 3
            elif row['FTHG'] == row['FTAG']:
                points += 1
        for _, row in away_matches.iterrows():
            if row['FTAG'] > row['FTHG']:
                points += 3
            elif row['FTAG'] == row['FTHG']:
                points += 1

        def safe(avg: float, denom: float) -> float:
            return (avg / denom) if denom > 0 else 0.0

        team_stats[team] = {
            'attack_home': safe((goals_scored_home / len(home_matches)) if len(home_matches) else 0.0,
                                league_home_goals),
            'attack_away': safe((goals_scored_away / len(away_matches)) if len(away_matches) else 0.0,
                                league_away_goals),
            'defense_home': safe((goals_conceded_home / len(home_matches)) if len(home_matches) else 0.0,
                                 league_away_goals),
            'defense_away': safe((goals_conceded_away / len(away_matches)) if len(away_matches) else 0.0,
                                 league_home_goals),
            'shots_attack_home': safe((shots_home / len(home_matches)) if len(home_matches) else 0.0,
                                      league_home_shots),
            'shots_attack_away': safe((shots_away / len(away_matches)) if len(away_matches) else 0.0,
                                      league_away_shots),
            'shots_defense_home': safe((shots_conceded_home / len(home_matches)) if len(home_matches) else 0.0,
                                       league_away_shots),
            'shots_defense_away': safe((shots_conceded_away / len(away_matches)) if len(away_matches) else 0.0,
                                       league_home_shots),
            'sot_attack_home': safe((sot_home / len(home_matches)) if len(home_matches) else 0.0,
                                    league_home_sot),
            'sot_attack_away': safe((sot_away / len(away_matches)) if len(away_matches) else 0.0,
                                    league_away_sot),
            'sot_defense_home': safe((sot_conceded_home / len(home_matches)) if len(home_matches) else 0.0,
                                     league_away_sot),
            'sot_defense_away': safe((sot_conceded_away / len(away_matches)) if len(away_matches) else 0.0,
                                     league_home_sot),
            'corner_attack_home': safe((corners_home / len(home_matches)) if len(home_matches) else 0.0,
                                       league_home_corners),
            'corner_attack_away': safe((corners_away / len(away_matches)) if len(away_matches) else 0.0,
                                       league_away_corners),
            'corner_defense_home': safe((corners_conceded_home / len(home_matches)) if len(home_matches) else 0.0,
                                        league_away_corners),
            'corner_defense_away': safe((corners_conceded_away / len(away_matches)) if len(away_matches) else 0.0,
                                        league_home_corners),
            'points': float(points)
        }
    # Ranking based on points (higher is better)
    sorted_teams = sorted(team_stats.items(), key=lambda x: x[1]['points'], reverse=True)
    for rank, (team, stats) in enumerate(sorted_teams, start=1):
        stats['rank'] = float(rank)
    return team_stats


def aggregate_team_stats(stats_per_season: Dict[str, Dict[str, Dict[str, float]]],
                         weights: Dict[str, float]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Aggregate per-season statistics across multiple seasons using given weights.

    Parameters
    ----------
    stats_per_season : dict
        Mapping from season filename to its team statistics (see ``compute_team_stats``).
    weights : dict
        Mapping from season filename to weight used in the aggregation.

    Returns
    -------
    aggregated : dict
        Aggregated statistics per team.
    bottom_average : dict
        Average statistics of the four lowest-ranked teams. Used to estimate
        metrics for teams that do not appear in the historical data (e.g. newly
        promoted clubs).
    """
    aggregated: Dict[str, Dict[str, float]] = {}
    all_teams: Iterable[str] = set().union(*[stats.keys() for stats in stats_per_season.values()])
    for team in all_teams:
        total_weight = 0.0
        accumulator: defaultdict = defaultdict(float)
        for season, stats in stats_per_season.items():
            if team not in stats:
                continue
            w = float(weights.get(season, 1.0))
            total_weight += w
            for k, v in stats[team].items():
                accumulator[k] += w * v
        if total_weight > 0:
            aggregated[team] = {k: (v / total_weight) for k, v in accumulator.items()}
    # Compute bottom four average to estimate unknown teams
    bottom4 = sorted(aggregated.items(), key=lambda x: x[1].get('rank', 9999), reverse=True)[:4]
    bottom_average: Dict[str, float] = defaultdict(float)
    for _, team_stats in bottom4:
        for k, v in team_stats.items():
            bottom_average[k] += v
    for k in bottom_average:
        bottom_average[k] /= float(len(bottom4)) if bottom4 else 1.0
    return aggregated, bottom_average


def most_likely_score(lambda_home: float, lambda_away: float) -> Tuple[int, int]:
    """Compute the most probable (home_goals, away_goals) under independent Poisson models.

    Evaluates all scorelines from 0–6 goals for both teams and returns the
    combination with the highest probability.
    """
    def poisson_p(lam: float, k: int) -> float:
        return (lam ** k) * math.exp(-lam) / math.factorial(k)

    best_score = (0, 0)
    max_prob = 0.0
    for i, j in product(range(7), repeat=2):
        p = poisson_p(lambda_home, i) * poisson_p(lambda_away, j)
        if p > max_prob:
            max_prob = p
            best_score = (i, j)
    return best_score


def predict_fixture(home_team: str, away_team: str,
                    agg_stats: Dict[str, Dict[str, float]],
                    bottom_stats: Dict[str, float],
                    avg_home_goals: float,
                    avg_away_goals: float) -> Dict[str, object]:
    """Predict expected goals and most likely scoreline for a fixture.

    Parameters
    ----------
    home_team, away_team : str
        Names of the clubs.
    agg_stats : dict
        Aggregated team statistics.
    bottom_stats : dict
        Default statistics for teams absent from the dataset.
    avg_home_goals, avg_away_goals : float
        League-average goals scored by home and away teams (used in Poisson model).

    Returns
    -------
    dict
        Contains the expected home goals, expected away goals and predicted
        most likely scoreline.
    """
    home = agg_stats.get(home_team, bottom_stats)
    away = agg_stats.get(away_team, bottom_stats)
    lambda_home = avg_home_goals * home['attack_home'] * away['defense_away']
    lambda_away = avg_away_goals * away['attack_away'] * home['defense_home']
    predicted = most_likely_score(lambda_home, lambda_away)
    return {
        'home_team': home_team,
        'away_team': away_team,
        'expected_home_goals': lambda_home,
        'expected_away_goals': lambda_away,
        'predicted_score': predicted
    }


def load_fixtures(path: Path) -> List[Tuple[str, str]]:
    """Load upcoming fixtures from a CSV file with 'HomeTeam' and 'AwayTeam' columns."""
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    return list(df[['HomeTeam', 'AwayTeam']].itertuples(index=False, name=None))


def generate_predictions(base_dir: Path = None) -> List[Dict[str, object]]:
    """
    Compute predictions for upcoming fixtures.

    This function aggregates historical and current-season data, calculates
    league averages, loads fixtures and returns the list of predictions.

    Parameters
    ----------
    base_dir : Path, optional
        Base directory containing the ``data`` folder. If omitted, it is
        inferred relative to this file.

    Returns
    -------
    List[dict]
        A list of prediction dictionaries ready to be serialized.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / 'data'
    # Historical seasons and weights
    season_files = ['season-2223.csv', 'season-2324.csv', 'season-2425.csv']
    weights: Dict[str, float] = {
        'season-2223.csv': 1.0,
        'season-2324.csv': 2.0,
        'season-2425.csv': 3.0,
    }
    # Load season stats
    stats_per_season: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fname in season_files:
        path = data_dir / fname
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        stats_per_season[fname] = compute_team_stats(df)
    # Include current season if present
    current_path = data_dir / 'current_season.csv'
    if current_path.is_file() and current_path.stat().st_size > 0:
        current_df = pd.read_csv(current_path)
        stats_per_season['current_season'] = compute_team_stats(current_df)
        weights['current_season'] = max(weights.values()) + 1.0
    # Aggregate stats and compute bottom for missing teams
    aggregated_stats, bottom_stats = aggregate_team_stats(stats_per_season, weights)
    # Compute league-average goals across loaded results (including current season)
    total_fthg = total_ftag = 0.0
    count = 0
    for fname in season_files + (['current_season.csv'] if current_path.is_file() else []):
        fpath = data_dir / fname
        if fpath.is_file() and fpath.stat().st_size > 0:
            df = pd.read_csv(fpath)
            total_fthg += df['FTHG'].sum()
            total_ftag += df['FTAG'].sum()
            count += len(df)
    avg_home_goals = (total_fthg / count) if count else 0.0
    avg_away_goals = (total_ftag / count) if count else 0.0
    # Load fixtures
    fixtures_file = data_dir / 'fixtures.csv'
    fixtures = load_fixtures(fixtures_file)
    predictions: List[Dict[str, object]] = []
    for home_team, away_team in fixtures:
        pred = predict_fixture(home_team, away_team, aggregated_stats, bottom_stats,
                               avg_home_goals, avg_away_goals)
        predictions.append(pred)
    # Return cleaned predictions (convert to plain Python types)
    return [
        {
            'home_team': p['home_team'],
            'away_team': p['away_team'],
            'expected_home_goals': round(float(p['expected_home_goals']), 3),
            'expected_away_goals': round(float(p['expected_away_goals']), 3),
            'predicted_score': [int(p['predicted_score'][0]), int(p['predicted_score'][1])],
        }
        for p in predictions
    ]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    predictions = generate_predictions(base_dir=base_dir)
    output_path = base_dir / 'predictions.json'
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)
    print(f"Generated {len(predictions)} predictions at {output_path}")


if __name__ == '__main__':
    main()