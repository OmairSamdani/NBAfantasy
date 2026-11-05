import pandas as pd
from nba_api.stats.endpoints import drafthistory

d = drafthistory.DraftHistory(league_id="00", timeout=60).get_data_frames()[0]
print(d.columns.tolist())
print(len(d))
d.to_csv("draft_history.csv", index=False)