"""Summarize completed development rounds, excluding repeated control models."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
seen={'v6_no_id_features'}
best=0.8157894736842105
rows=[]
for folder in sorted((ROOT/'reports').glob('run_*')):
    source=folder/'screen_results.csv'
    if not source.exists():
        continue
    frame=pd.read_csv(source)
    new=frame.loc[~frame.version.isin(seen)]
    seen.update(frame.version)
    if new.empty:
        continue
    winner=new.sort_values('accuracy',ascending=False).iloc[0]
    gain=winner.accuracy-best
    best=max(best,float(winner.accuracy))
    gate_path=folder/'gate.json'
    eligibility_path=folder/'stability/eligibility.json'
    rows.append(dict(run=folder.name,candidate=winner.version,
        accuracy=float(winner.accuracy),gain_over_previous_best=float(gain),
        running_best=best,
        online_gate_passed=json.loads(gate_path.read_text())['passed'] if gate_path.exists() else None,
        top5_research_eligible=json.loads(eligibility_path.read_text())['A_eligible'] if eligibility_path.exists() else None))
result=pd.DataFrame(rows)
result.to_csv(ROOT/'reports/local_iterations.csv',index=False)
fig,ax=plt.subplots(figsize=(10,4.5))
rounds=[int(x.split('_')[-1]) for x in result.run]
ax.plot(rounds,result.accuracy*100,'o-',label='Best new candidate in round')
ax.plot(rounds,result.running_best*100,'s-',label='Best development score so far')
ax.axhline(83,color='firebrick',ls='--',label='Historical 83% gate (new A policy is separate)')
ax.set(xlabel='Development round',ylabel='3-fold OOF accuracy (%)',xticks=sorted(set(rounds[::3]+rounds[-1:])),
       title='Spaceship Titanic: fixed development split (not leaderboard scores)')
ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'reports/iteration_progress.png',dpi=160);plt.close(fig)
print(result[['run','candidate','accuracy','running_best']].to_string(index=False))
