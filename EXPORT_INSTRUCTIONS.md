# FanGraphs CSV export instructions

FanGraphs blocks programmatic scrapes from this machine. Export from your browser instead — it takes about 5 minutes total. Two files is enough to start.

---

## File 1 — MLB batting, 2015–2025, split by season

1. Open: <https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&qual=0&type=8&season1=2015&season=2025&ind=1>
2. Verify these filters on the page:
   - **Group:** Player Stats
   - **Stats:** Batting
   - **League:** All
   - **Position:** All
   - **Active Roster Only:** No
   - **Min PA:** 0
   - **Season range:** 2015 → 2025
   - **Split Seasons:** ON (each player-season is its own row)
3. Scroll down. Below the table, click **Export Data**.
4. Save the CSV as: `data/raw/mlb/mlb_dashboard_2015_2025.csv`

The CSV must include columns `Name`, `playerid`, `Season`, `Age`, `PA`, `wRC+`. (FG's default Dashboard view includes all of these.)

---

## File 2 — AAA batting, 2015–2024

The MiLB leaderboard's URL parameters changed when MLB rebranded the minor-league
system. The cleanest path is to open the bare leaderboard URL and set filters by
hand on the page:

1. Open: <https://www.fangraphs.com/leaders/minor-league>
2. Set these filters in the page UI (top of the page):
   - **Group:** Player Stats
   - **Stats:** Batting
   - **Level:** AAA  (the dropdown will say "Triple-A" — pick that)
   - **League:** All (or leave blank — picks up IL/PCL pre-2021 and AAA East/West 2021+)
   - **Stat Type / View:** Dashboard (default)
   - **Min PA (qual):** 0
   - **Season range:** 2015 → 2024
   - **Split Seasons:** ON (one row per player-season; sometimes labeled "Multiple Seasons" or "Individual Seasons")
3. Click **Export Data** below the table.
4. Save as: `data/raw/aaa/aaa_dashboard_2015_2024.csv`

**If FanGraphs's MiLB leaderboard does not let you span multiple seasons in one
export** (this is common for MiLB — the multi-season toggle often only works for
MLB), then export **one CSV per season**:
- Set the season to 2015, Export Data → save as `aaa_2015.csv`
- Repeat for 2016, 2017, ... 2024
- Drop all 10 files into `data/raw/aaa/`. The loader script merges them automatically.

CSV must include: `Name`, `playerid` (sometimes `IDfg`), `Season` (or `Year`), `Age`, `Team`, `PA`, `AVG`, `OBP`, `SLG`, `wOBA`, `wRC+`, `BB%`, `K%`, `ISO`, `BABIP`.
If your export doesn't have all of these by default, look for a "Customize Columns" / "Edit Columns" button on the leaderboard and add the missing ones before exporting.

---

## Once your files are in place

Run from the project root:

```
.venv/bin/python scripts/01_load_and_match.py
```

It prints the matched sample size and writes the merged dataset to `data/processed/matched_players.csv`.
