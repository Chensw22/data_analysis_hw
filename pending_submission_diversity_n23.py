"""Compare decision diversity among unsubmitted files with four locally positive replays."""
import json
import numpy as np
import pandas as pd
from broad_partition_evidence_n23 import ART, ROOT, SEEDS, write
from export_budget_quarter_n23 import source
from export_service_bag_n23 import PERSONAL, THREE, NAME as LINEAR
from export_service_bag_tree_n23 import NAME as TREE
from kaggle_workflow import sha256

REPORT=ROOT/'reports/nov23_recent_family_transfer'
PAIRS=ROOT/'reports/nov23_passenger_service_pairs'
FIXED=ROOT/'reports/nov23_fixed_vote_totals'
PAIR='proposal_n23_service_pair_'
DONOR=PAIR+'zero_donor'


def main():
    read=lambda path:json.loads(path.read_text())
    sources={
        PERSONAL:(ROOT/'reports/nov23_dependence_anchors/three_corrections.json',('plans',),('personal',)),
        THREE:(ROOT/'reports/nov23_dependence_anchors/three_corrections.json',('plans',),('mean_probability',)),
        LINEAR:(FIXED/'service_bag.json',('plans',),('changes','quarter_fixed')),
        TREE:(FIXED/'service_bag_trees.json',('plans',),('changes','quarter_initial')),
        PAIR+'quarter':(PAIRS/'test_budget.json',('plans',),('quarter',)),
        PAIR+'change_150':(PAIRS/'test_budget.json',('plans',),('change_150',)),
        DONOR:(ROOT/'reports/nov23_donor_direction/overlay.json',(),('0.1','matched','combined')),
        PAIR+'corroborated':(PAIRS/'portfolio.json',('plans',),('metrics','pair_plus_corroborated'))}
    test=pd.read_csv(ROOT/'data/raw/test.csv',usecols=['PassengerId'])
    _,incumbent,_=source('proposal_v131_party_consensus',test)
    receipts={read(p)['file_sha256'] for p in (ROOT/'reports').glob('*/online_attempts/*.json')}
    votes,info={},{}
    for name,(report,prefix,suffix) in sources.items():
        path=ART/name/'submission.csv'
        digest=sha256(path)
        if digest in receipts:
            continue
        assert digest==read(ROOT/'reports'/name/'manifest.json')['sha256']
        table=pd.read_csv(path)
        np.testing.assert_array_equal(table.PassengerId,test.PassengerId)
        assert table.Transported.dtype==bool
        fold_values=[]
        data=read(report)
        for seed in [*map(str,SEEDS),'v131']:
            m=data
            for key in (*prefix,seed,*suffix):
                m=m[key]
            fold_values.append(m['fixed']-m['harmed'])
        if min(fold_values)<=0:
            continue
        votes[name]=table.Transported.to_numpy()
        info[name]=dict(source=str(report.relative_to(ROOT)),net_correct=fold_values,sha256=digest,
                        incumbent_changes=int((votes[name]!=incumbent).sum()))
    distances={n:{k:int((v!=u).sum()) for k,u in votes.items()} for n,v in votes.items()}
    # Reserve the first test for the established priority. Then maximize the minimum
    # test-file distance to all choices already selected (ties use weakest local gain).
    order=[]
    if votes:
        first=DONOR if DONOR in votes else max(votes,key=lambda k:(min(info[k]['net_correct']),k))
        order.append(first)
    while len(order)<len(votes):
        score=lambda k:(min(distances[k][j] for j in order),min(info[k]['net_correct']),k)
        order.append(max((k for k in votes if k not in order),key=score))
    write(REPORT/'pending_diversity.json',dict(files=info,distances=distances,exploratory_order=order))
    for i,k in enumerate(order):
        print(k,info[k]['net_correct'],[distances[k][v] for v in order[:i]],flush=True)


if __name__=='__main__':
    main()
