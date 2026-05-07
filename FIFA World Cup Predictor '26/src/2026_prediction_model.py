import pandas as pd
import numpy as np
from collections import defaultdict
from xgboost import XGBRegressor
import os, csv
from datetime import datetime

# Config
n_simulations = 10000 # set to N simulations
data_path = "FIFA World Cup Predictor '26/Data/Processed/training_dataset2.csv"
train_cutoff = 2022 # train on matches before this year

# ELO K-factor for in-tournament updates
ELO_K = 32

# Team Name Map
# Names in the groups dict map to names used in the dataset
NAME_MAP = {
    "Czechia": "Czech Republic",
    "Turkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
}

# Load & prep data
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

train = df[df["date"].dt.year < train_cutoff].copy().reset_index(drop=True)

# Build Current Team Stats from full dataset
# (use all data up to Jan 2026 for simulation lookups,
# but only train the model on pre-2022 data)
full = df.copy()

# Latest ELO per team — from both home and away columns
elo_h = full[["date","home_team","home_elo"]].rename(
    columns={"home_team":"team","home_elo":"elo"})
elo_a = full[["date","away_team","away_elo"]].rename(
    columns={"away_team":"team","away_elo":"elo"})
elo_all = pd.concat([elo_h, elo_a]).dropna(subset=["elo"])
LATEST_ELO = elo_all.sort_values("date").groupby("team")["elo"].last().to_dict()

# Latest rolling attack/defense per team (pre-engineered in dataset)
# Use the most recent row where the team appeared as home
latest_home = full.sort_values("date").groupby("home_team").last().reset_index()
latest_away = full.sort_values("date").groupby("away_team").last().reset_index()

team_avg_scored = {}
team_avg_conceded = {}

for _, r in latest_home.iterrows():
    team_avg_scored[r["home_team"]] = r["home_avg_scored"]
    team_avg_conceded[r["home_team"]] = r["home_avg_conceded"]

for _, r in latest_away.iterrows():
    t = r["away_team"]
    # Away stats are generally slightly lower than home; use as fallback only
    if t not in team_avg_scored:
        team_avg_scored[t] = r["away_avg_scored"]
        team_avg_conceded[t] = r["away_avg_conceded"]


# FEATURE ENGINEERING FOR TRAINING SET
# Uses the pre-engineered columns already in the dataset
# + neutral flag + separate home/away ELO
features = [
    "home_avg_scored", "home_avg_conceded",
    "away_avg_scored", "away_avg_conceded",
    "home_elo", "away_elo", "elo_diff",
    "neutral", # 1 = neutral venue, 0 = home advantage
    "home_win_rate", "away_win_rate",
    "home_goal_diff_avg", "away_goal_diff_avg",
]

train["elo_diff"] = train["home_elo"] - train["away_elo"]
train["neutral"] = train["neutral"].astype(int)
train = train.dropna(subset=features + ["home_score","away_score"]).copy()

# Train Models
model_home = XGBRegressor(
    n_estimators=600, max_depth=5, learning_rate=0.05,
    random_state=42, subsample=0.8, colsample_bytree=0.8
)
model_away = XGBRegressor(
    n_estimators=600, max_depth=5, learning_rate=0.05,
    random_state=42, subsample=0.8, colsample_bytree=0.8
)
model_home.fit(train[features], train["home_score"])
model_away.fit(train[features], train["away_score"])
print("Models trained.")

# LIVE ELO — updates during tournament simulation
# Reset at the start of each simulation run
_live_elo = {}

def reset_elo():
    """Reset ELO to latest known values before each simulation."""
    global _live_elo
    _live_elo = dict(LATEST_ELO)

def get_elo(team):
    ds_name = NAME_MAP.get(team, team)
    # Scotland has no ELO in dataset — use ~average of similar UEFA teams
    fallback = 1700.0
    return _live_elo.get(ds_name, _live_elo.get(team, fallback))

def update_elo(home, away, home_goals, away_goals):
    """Standard ELO update after a match."""
    r_h = get_elo(home)
    r_a = get_elo(away)
    exp_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
    exp_a = 1 - exp_h
    if home_goals > away_goals:
        s_h, s_a = 1.0, 0.0
    elif away_goals > home_goals:
        s_h, s_a = 0.0, 1.0
    else:
        s_h, s_a = 0.5, 0.5

    ds_home = NAME_MAP.get(home, home)
    ds_away = NAME_MAP.get(away, away)
    _live_elo[ds_home] = r_h + ELO_K * (s_h - exp_h)
    _live_elo[ds_away] = r_a + ELO_K * (s_a - exp_a)

# FEATURE BUILDER — neutral venue aware, uses live ELO
_warned_teams = set()

def get_stats(team):
    ds_name = NAME_MAP.get(team, team)
    scored = team_avg_scored.get(ds_name,   team_avg_scored.get(team,   1.2))
    conceded = team_avg_conceded.get(ds_name, team_avg_conceded.get(team, 1.2))
    if ds_name not in team_avg_scored and team not in team_avg_scored \
            and team not in _warned_teams:
        print(f" WARNING: '{team}' has no historical stats — using defaults")
        _warned_teams.add(team)
    return scored, conceded

def build_features(home, away, neutral=True):
    h_scored, h_conceded = get_stats(home)
    a_scored, a_conceded = get_stats(away)
    h_elo = get_elo(home)
    a_elo = get_elo(away)

    # win_rate and goal_diff_avg — use dataset values or neutral defaults
    ds_h = NAME_MAP.get(home, home)
    ds_a = NAME_MAP.get(away, away)

    # Pull from latest_home / latest_away dicts built above
    h_wr = latest_home.set_index("home_team")["home_win_rate"].get(ds_h, 0.45)
    a_wr = latest_away.set_index("away_team")["away_win_rate"].get(ds_a, 0.45)
    h_gda = latest_home.set_index("home_team")["home_goal_diff_avg"].get(ds_h, 0.0)
    a_gda = latest_away.set_index("away_team")["away_goal_diff_avg"].get(ds_a, 0.0)

    return np.array([[
        h_scored, h_conceded,
        a_scored, a_conceded,
        h_elo, a_elo,
        h_elo - a_elo,
        int(neutral), # World Cup = neutral venue
        h_wr, a_wr,
        h_gda, a_gda,
    ]])

# PENALTY SHOOTOUT
def simulate_penalties(home, away, p=0.75):
    h = sum(np.random.random() < p for _ in range(5))
    a = sum(np.random.random() < p for _ in range(5))
    while h == a:
        h += int(np.random.random() < p)
        a += int(np.random.random() < p)
    return h, a, (home if h > a else away)

# MATCH SIMULATION
# World Cup matches are always neutral=True
# ELO is updated after every match
def simulate_match(home, away, knockout=False):
    x = build_features(home, away, neutral=True)
    lam_h = max(0.1, float(model_home.predict(x)[0]))
    lam_a = max(0.1, float(model_away.predict(x)[0]))
    hg = np.random.poisson(lam_h)
    ag = np.random.poisson(lam_a)

    # Update live ELO after this match
    update_elo(home, away, hg, ag)

    if knockout and hg == ag:
        hp, ap, winner = simulate_penalties(home, away)
        return hg, ag, winner, True, hp, ap

    return hg, ag, (home if hg > ag else away), False, None, None

# GROUP STAGE
def simulate_group(teams):
    matches = [
        (teams[0], teams[1]), (teams[2], teams[3]),
        (teams[0], teams[2]), (teams[1], teams[3]),
        (teams[0], teams[3]), (teams[1], teams[2]),
    ]
    table = {t: {"pts":0, "gf":0, "ga":0, "gd":0} for t in teams}
    for h, a in matches:
        hg, ag, _, _, _, _ = simulate_match(h, a, knockout=False)
        table[h]["gf"] += hg; table[a]["gf"] += ag
        table[h]["ga"] += ag; table[a]["ga"] += hg
        table[h]["gd"] += hg-ag; table[a]["gd"] += ag-hg
        if   hg > ag: table[h]["pts"] += 3
        elif ag > hg: table[a]["pts"] += 3
        else: table[h]["pts"] += 1; table[a]["pts"] += 1

    return sorted(
        table.items(),
        key=lambda x: (x[1]["pts"], x[1]["gd"], x[1]["gf"], np.random.random()),
        reverse=True
    )

# BEST 8 THIRD-PLACE TEAMS
def get_best_thirds(group_results):
    thirds = [(g, group_results[g][2][0], group_results[g][2][1])
              for g in group_results]
    thirds_sorted = sorted(
        thirds,
        key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gf"], np.random.random()),
        reverse=True
    )[:8]
    return [t[1] for t in thirds_sorted]

# KNOCKOUT ROUND
def knockout_round(teams, round_name=""):
    winners, losers, log = [], [], []
    for i in range(0, len(teams), 2):
        h, a = teams[i], teams[i+1]
        hg, ag, winner, pens, hp, ap = simulate_match(h, a, knockout=True)
        loser = a if winner == h else h
        winners.append(winner); losers.append(loser)
        log.append({
            "round": round_name, "home": h, "away": a,
            "home_score": hg, "away_score": ag,
            "penalties": pens, "home_pens": hp, "away_pens": ap,
            "winner": winner,
        })
    return winners, losers, log

# FIFA 2026 BRACKET SEEDING
def build_bracket(group_results):
    firsts  = {g: group_results[g][0][0] for g in group_results}
    seconds = {g: group_results[g][1][0] for g in group_results}
    thirds  = get_best_thirds(group_results)
    pairs   = [
        (firsts["A"], seconds["B"]), (firsts["C"], seconds["D"]),
        (firsts["E"], seconds["F"]), (firsts["G"], seconds["H"]),
        (firsts["I"], seconds["J"]), (firsts["K"], seconds["L"]),
        (firsts["B"], seconds["A"]), (firsts["D"], seconds["C"]),
        (firsts["F"], seconds["E"]), (firsts["H"], seconds["G"]),
        (firsts["J"], seconds["I"]), (firsts["L"], seconds["K"]),
        (thirds[0], thirds[1]), (thirds[2], thirds[3]),
        (thirds[4], thirds[5]), (thirds[6], thirds[7]),
    ]
    ordered = []
    for h, a in pairs:
        ordered.append(h); ordered.append(a)
    return ordered

# FULL TOURNAMENT
def simulate_tournament(groups, verbose=True):
    reset_elo() # fresh ELO for every simulation

    group_results = {}
    group_log = []

    if verbose: print("\n=== GROUP STAGE ===\n")

    for g in groups:
        table = simulate_group(groups[g])
        group_results[g] = table
        for rank, (team, stats) in enumerate(table, 1):
            group_log.append({
                "group": g, "rank": rank, "team": team,
                "pts": stats["pts"], "gf": stats["gf"],
                "ga":  stats["ga"],  "gd": stats["gd"],
            })
        if verbose:
            print(f"Group {g}")
            print(f"  {'#':<4}{'Team':<32} {'Pts':>4} {'GF':>4} {'GA':>4} {'GD':>5}")
            print(f"  {'-'*53}")
            for i, (t, s) in enumerate(table):
                print(f"  {i+1:<4}{t:<32} {s['pts']:>4} {s['gf']:>4} {s['ga']:>4} {s['gd']:>+5}")
            print()

    current = build_bracket(group_results)
    if verbose: print("32 qualified teams seeded into bracket.\n")

    knockout_log = []
    semi_losers  = []
    runner_up = None   # variable to capture the Final loser
    round_names = ["Round of 32","Round of 16","Quarter-Finals","Semi-Finals","Final"]

    for r in round_names:
        current, losers, matches = knockout_round(current, r)
        knockout_log.extend(matches)
        if r == "Semi-Finals":
            semi_losers = losers
        if r == "Final":
            runner_up = losers[0]   # capture the runner-up here
        if verbose:
            print(f"=== {r} ===")
            for m in matches:
                pen_str = f" (pens: {m['home_pens']}-{m['away_pens']})" if m["penalties"] else ""
                print(f"  {m['home']:32s} {m['home_score']} - {m['away_score']}"
                      f"  {m['away']:32s}  {m['winner']}{pen_str}")
            print()

    # 3rd place playoff
    t3h, t3a = semi_losers[0], semi_losers[1]
    hg, ag, t3_winner, pens, hp, ap = simulate_match(t3h, t3a, knockout=True)
    fourth = t3a if t3_winner == t3h else t3h
    t3_match = {
        "round": "3rd Place Playoff", "home": t3h, "away": t3a,
        "home_score": hg, "away_score": ag,
        "penalties": pens, "home_pens": hp, "away_pens": ap,
        "winner": t3_winner,
    }
    knockout_log.append(t3_match)

    champion = current[0]

    if verbose:
        pen_str = f" (pens: {hp}-{ap})" if pens else ""
        print(f"=== 3RD PLACE PLAYOFF ===")
        print(f"  {t3h:32s} {hg} - {ag}  {t3a:32s}  → {t3_winner}{pen_str}\n")
        print("=" * 65)
        print(f" Champion:  {champion}")
        print(f" Runner-up: {runner_up}")   # ← FIX: now prints correctly
        print(f" 3rd Place: {t3_winner}")
        print(f" 4th Place: {fourth}")
        print("=" * 65 + "\n")

    return champion, runner_up, t3_winner, fourth, group_log, knockout_log

# MONTE CARLO
round_names = ["Round of 32","Round of 16","Quarter-Finals","Semi-Finals","Final"]
all_stages = round_names + ["Champion"]

def run_simulations(groups, n=n_simulations):
    wins = defaultdict(int)
    thirds = defaultdict(int)
    stage_reached = defaultdict(lambda: defaultdict(int))
    checkpoints = {int(n * p) for p in [0.25, 0.5, 0.75, 1.0]}

    print(f"Running {n:,} simulations...\n")
    for i in range(1, n+1):
        champion, runner_up, t3_winner, _, _, knockout_log = simulate_tournament(
            groups, verbose=False)

        for match in knockout_log:
            if match["round"] != "3rd Place Playoff":
                stage_reached[match["home"]][match["round"]] += 1
                stage_reached[match["away"]][match["round"]] += 1

        wins[champion] += 1
        stage_reached[champion]["Champion"] += 1
        thirds[t3_winner] += 1

        if i in checkpoints:
            leader = max(wins, key=wins.get)
            print(f"  {i/n*100:5.0f}% complete — current leader: "
                  f"{leader} ({wins[leader]/i*100:.1f}%)")

    results = sorted(wins.items(),   key=lambda x: x[1], reverse=True)
    third_results = sorted(thirds.items(), key=lambda x: x[1], reverse=True)

    # Progression table
    col_w  = 11
    labels = ["R32","R16","QF","SF","Final","Champion"]
    print("\n")
    print("=" * 100)
    print(" KNOCKOUT PROGRESSION ANALYSIS — % chance to reach each stage")
    print("=" * 100)
    print(f"  {'Team':<32}" + "".join(f"{l:>{col_w}}" for l in labels))
    print("  " + "-" * (32 + col_w * len(labels)))

    all_teams = sorted(
        stage_reached.keys(),
        key=lambda t: (stage_reached[t].get("Round of 32",0),
                       stage_reached[t].get("Champion",0)),
        reverse=True
    )
    for team in all_teams:
        row = f"  {team:<32}"
        for stage in all_stages:
            pct = stage_reached[team].get(stage, 0) / n * 100
            row += f"{pct:>{col_w-1}.1f}%"
        print(row)
    print("  " + "-" * (32 + col_w * len(labels)))
    print()

    print("=" * 60)
    print(" 3RD PLACE FINISH PROBABILITIES")
    print("=" * 60)
    for team, count in third_results[:20]:
        print(f"  {team:<32} {count/n*100:5.2f}%")
    print()

    print("=" * 60)
    print(" WIN PROBABILITIES")
    print("=" * 60)
    return results, third_results, stage_reached

# Export as CSVs + Excel file
def export_results(champion, runner_up, t3_winner, fourth,
                   group_log, knockout_log,
                   sim_results, third_results, stage_reached,
                   out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    total = sum(c for _, c in sim_results)

    # 1. Group stage
    group_csv = os.path.join(out_dir, f"group_stage_{ts}.csv")
    pd.DataFrame(group_log)[["group","rank","team","pts","gf","ga","gd"]]\
      .to_csv(group_csv, index=False)
    print(f" Group stage: {group_csv}")

    # 2. Knockout
    ko_csv = os.path.join(out_dir, f"knockout_{ts}.csv")
    pd.DataFrame(knockout_log)[
        ["round","home","away","home_score","away_score",
         "penalties","home_pens","away_pens","winner"]
    ].to_csv(ko_csv, index=False)
    print(f" Knockout: {ko_csv}")

    # 3. Win probabilities
    sim_csv = os.path.join(out_dir, f"win_probabilities_{ts}.csv")
    with open(sim_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team","wins","win_pct"])
        for team, count in sim_results:
            w.writerow([team, count, round(count/total*100, 2)])
    print(f" Win probabilities: {sim_csv}")

    # 4. Full results CSV
    full_csv = os.path.join(out_dir, f"full_results_{ts}.csv")
    rows = []
    rows.append({"section":"CHAMPION","team":champion, "detail":""})
    rows.append({"section":"RUNNER_UP", "team":runner_up, "detail":""})
    rows.append({"section":"3RD_PLACE", "team":t3_winner, "detail":""})
    rows.append({"section":"4TH_PLACE", "team":fourth, "detail":""})
    rows.append({"section":"SIMULATED_CHAMPION", "team":sim_results[0][0], "detail":""})
    for r in group_log:
        rows.append({
            "section": "GROUP_STAGE", "team": r["team"],
            "detail": (f"Group {r['group']} | Rank {r['rank']} | Pts {r['pts']} | "
                       f"GF {r['gf']} | GA {r['ga']} | GD {r['gd']:+d}")
        })
    for r in knockout_log:
        pen_str = f" (pens: {r['home_pens']}-{r['away_pens']})" if r["penalties"] else ""
        rows.append({
            "section": r["round"].upper().replace(" ","_"),
            "team": r["winner"],
            "detail": f"{r['home']} {r['home_score']}-{r['away_score']} {r['away']}{pen_str}"
        })
    for team, count in sim_results:
        rows.append({
            "section": "WIN_PROBABILITY", "team": team,
            "detail": f"{round(count/total*100,2)}%"
        })
    pd.DataFrame(rows).to_csv(full_csv, index=False)
    print(f"Full results: {full_csv}")

    # 5. Progression CSV
    col_keys = {
        "Round of 32":"r32_pct","Round of 16":"r16_pct",
        "Quarter-Finals":"qf_pct","Semi-Finals":"sf_pct",
        "Final":"final_pct","Champion":"champion_pct",
    }
    prog_rows = []
    for team in sorted(
        stage_reached.keys(),
        key=lambda t:(stage_reached[t].get("Round of 32",0),
                      stage_reached[t].get("Champion",0)),
        reverse=True
    ):
        row = {"team": team}
        for stage, col in col_keys.items():
            row[col] = round(stage_reached[team].get(stage,0)/total*100, 2)
        prog_rows.append(row)

    prog_df  = pd.DataFrame(prog_rows)
    prog_csv = os.path.join(out_dir, f"progression_{ts}.csv")
    prog_df.to_csv(prog_csv, index=False)
    print(f" Progression table: {prog_csv}")

    # 6. Excel workbook
    xlsx_path = os.path.join(out_dir, f"wc26_results_{ts}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(group_log)[["group","rank","team","pts","gf","ga","gd"]]\
          .to_excel(writer, sheet_name="Group Stage", index=False)
        pd.DataFrame(knockout_log)[
            ["round","home","away","home_score","away_score",
             "penalties","home_pens","away_pens","winner"]
        ].to_excel(writer, sheet_name="Knockout Rounds", index=False)
        pd.DataFrame([
            {"team":t,"wins":c,"win_pct":round(c/total*100,2)}
            for t,c in sim_results
        ]).to_excel(writer, sheet_name="Win Probabilities", index=False)
        pd.DataFrame([
            {"team":t,"third_place_finishes":c,
             "third_place_pct":round(c/total*100,2)}
            for t,c in third_results
        ]).to_excel(writer, sheet_name="3rd Place Probs", index=False)
        prog_df.to_excel(writer, sheet_name="Progression Analysis", index=False)
        pd.DataFrame([{
            "Champion":champion,"Runner-up":runner_up,   # ← FIX
            "3rd Place":t3_winner,"4th Place":fourth
        }]).to_excel(writer, sheet_name="Summary", index=False)
    print(f" Excel workbook: {xlsx_path}")

# GROUPS (2026 format — 12 groups of 4)
groups = {
    "A": ["Mexico",        "South Africa",           "South Korea",   "Czechia"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Qatar",         "Switzerland"],
    "C": ["Brazil",        "Morocco",                "Haiti",         "Scotland"],
    "D": ["United States", "Paraguay",               "Australia",     "Turkiye"],
    "E": ["Germany",       "Curaçao",                "Côte d'Ivoire", "Ecuador"],
    "F": ["Netherlands",   "Japan",                  "Sweden",        "Tunisia"],
    "G": ["Belgium",       "Egypt",                  "Iran",          "New Zealand"],
    "H": ["Spain",         "Cape Verde",             "Saudi Arabia",  "Uruguay"],
    "I": ["France",        "Senegal",                "Iraq",          "Norway"],
    "J": ["Argentina",     "Algeria",                "Austria",       "Jordan"],
    "K": ["Portugal",      "DR Congo",               "Uzbekistan",    "Colombia"],
    "L": ["England",       "Croatia",                "Ghana",         "Panama"],
}

# RUN
np.random.seed(99)

print("Running single tournament simulation...")
champion, runner_up, t3_winner, fourth, group_log, knockout_log = simulate_tournament(
    groups, verbose=True)

sim_results, third_results, stage_reached = run_simulations(groups, n=n_simulations)

print("\nExporting results...")
export_results(champion, runner_up, t3_winner, fourth,
               group_log, knockout_log,
               sim_results, third_results, stage_reached,
               out_dir="output")