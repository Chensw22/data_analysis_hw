import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import audit_top5_references as audit

root=Path.cwd()
dev=np.sort(np.load('artifacts/run_002/split_indices.npz')['development']).astype(np.int64)
assert len(dev)==6954 and audit.row_hash(dev)==audit.DEV_HASH
train,test,y=audit.read_development_only(root/'data/raw/train.csv',root/'data/raw/test.csv',dev)
def attrs(frame):
 d=pd.DataFrame(index=frame.index)
 d['group']=frame.PassengerId.str.split('_').str[0]
 d['cabin']=frame.Cabin
 d['surname']=frame.Name.str.split().str[-1]
 return d
tr,te=attrs(train),attrs(test);d=tr.iloc[dev].reset_index(drop=True)
report={'kind':'exploratory_only_no_candidate_score','read_targets':'original6954dev_only','development_sha':audit.row_hash(dev),'coverage':{},'crossgroup_surname_diagnostics':{},'source_sha':{}}
for relation in ['group','cabin','surname']:
 values=set(tr[relation].dropna())
 counts=tr[relation].value_counts()
 matched=te[relation].map(counts).fillna(0).to_numpy()
 report['coverage'][relation]={'shared_keys':len(values&set(te[relation].dropna())), 'test_rows_with_train_label_neighbor':int((matched>0).sum()),'test_rows':len(test),'neighbor_count_percentiles':np.percentile(matched,[0,25,50,75,90,100]).tolist()}

# Each unordered cross-GroupId same-surname pair, aggregate by surname first.
def pair_totals(residual):
 frame=d.assign(r=residual).dropna(subset=['surname'])
 out=[]
 for surname,rows in frame.groupby('surname',sort=True):
  g=rows.groupby('group').r.agg(['size','sum'])
  pairs=(len(rows)**2-np.square(g['size']).sum())/2
  if pairs:
   numerator=(rows.r.sum()**2-np.square(g['sum']).sum())/2
   out.append((pairs,numerator))
 return np.asarray(out,dtype=float)
strata=train.iloc[dev][['HomePlanet','CryoSleep']].fillna('__MISSING__').astype(str).agg('|'.join,axis=1).to_numpy()
def center(values):
 series=pd.Series(values);return values-series.groupby(strata).transform('mean').to_numpy()
for reference in ['v44','mitra']:
 path=root/f'artifacts/top5_reference_audit/{reference}/group_20261112.npz'
 summary=json.loads((path.parent/'summary.json').read_text())
 entry=next(v for v in summary['audits'] if v['kind']=='group')
 assert audit.file_hash(path)==entry['oof_sha256']
 z=np.load(path)
 assert np.array_equal(z['dev_ids'],dev)
 assert np.array_equal(z['passenger_ids'].astype(str),train.PassengerId.iloc[dev].astype(str))
 assert np.array_equal(z['group_ids'].astype(str),d.group.astype(str))
 report['source_sha'][str(path.relative_to(root))]=audit.file_hash(path)
 residual=center(y-z['prob'])
 pairs=pair_totals(residual)
 rng=np.random.default_rng(20261116)
 boot=[]
 for _ in range(2000):
  sampled=pairs[rng.integers(len(pairs),size=len(pairs))]
  boot.append(sampled[:,1].sum()/sampled[:,0].sum())
 counts=d.surname.value_counts()
 groupcounts=d.groupby(['surname','group']).size()
 crosscount=np.asarray([0 if pd.isna(s) else counts[s]-groupcounts[s,g] for s,g in zip(d.surname,d.group)])
 folds=z['fold_id']
 coverage=[]
 for fold in np.unique(folds):
  query=folds==fold; support=d.loc[~query,'surname'].value_counts()
  n=d.loc[query,'surname'].map(support).fillna(0).to_numpy()
  coverage.append({'fold':int(fold),'query_rows':int(query.sum()),'with_neighbor':int((n>0).sum()),'count_percentiles':np.percentile(n,[0,25,50,75,90,100]).tolist()})
 report['crossgroup_surname_diagnostics'][reference]={'reference':'frozen GroupCV20261112 OOF','residual':'y-p centered within HomePlanet×CryoSleep (exploratory)','surname_clusters':len(pairs),'crossgroup_pairs':int(pairs[:,0].sum()),'dev_rows_with_othergroup_same_surname':int((crosscount>0).sum()),'pair_residual_product':float(pairs[:,1].sum()/pairs[:,0].sum()),'surname_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'groupfold_neighbor_coverage':coverage}
report['limitations']=['Not a candidate OOF score or independent hypothesis test; reused development labels.','Cluster bootstrap by surname is descriptive; names/groups can create dependent clusters.','No relation threshold, alpha, beta, weights or model fitted/selected.']
(root/'artifacts/relational_review/diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
