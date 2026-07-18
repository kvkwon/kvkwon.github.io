Poisson Model
Companion code for the FC Quantitative Edge blog post:
"Modelling Football Matches with the Poisson Distribution"

Setup
pip install -r requirements.txt

Usage
Place a results.csv file (columns: home_team, away_team, home_goals, away_goals)
in this folder, then run:

python poisson_model.py

Edit the HOME_TEAM / AWAY_TEAM variables at the bottom of the script to test
different fixtures.