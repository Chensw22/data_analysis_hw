"""Replace the original fare posterior in the complete scored recipe; preserve directional edits."""
import json
import numpy as np
import pandas as pd
from booking_mix_confidence_transfer_n26 import partition
from directional_foundation_transfer_n27 import calibrate
from export_directional_foundation_n27 import NAME as DIRECTIONAL,apply
from export_consensus_review_n26 import ART,dig,write
from audit_clean_fare_anchor_bounds_n26 import read_sources
from held_expert_portfolios_n26 import DECK,SIX,panel_predictions,sources
from leave_booking_out_fare_n27 import OUT,ROOT,REPORT,roster
from rebase_fare_six_n26 import predict,NAME as BASE
from repeated_archive_misses_n26 import cohort,count
from kaggle_workflow import validate_submission
from six_way_against_scored_nine_n26 import test_decisions

NAME='proposal_n27_leave_order_fare'


def update(nine,raw,original_fine,new_fine,six,families,paid,champion,protect,only_add,direction="both"):
    old=predict(nine,raw,original_fine,six,families,paid)['joint']
    new=predict(nine,raw,new_fine,six,families,paid)['joint']
    active=(new>=.5)!=(old>=.5)
    if only_add:active&=(old>=.5)==nine
    if protect:active&=(champion>=.5)==(old>=.5)
    if direction=="up":active&=champion<.5
    if direction=="down":active&=champion>=.5
    return np.where(active,new,champion)


def run(export=None):
    people,y,g,f=roster()
    n=len(y)
    raw=people.iloc[:n].copy()
    known,versions=read_sources(raw,y,g[:n],f)
    a161,blind,certain,fresh=cohort(raw,y,f)
    np.testing.assert_array_equal(known,certain)
    full=sources()[-1]['full']
    p=panel_predictions(full,'full')
    challenger=np.column_stack([p[n] for n in SIX])
    with np.load(DECK/'deck_estimates.npz') as z:paid=z['eligible'][:n]
    fine=np.empty(len(people))
    for k in range(9):
        with np.load(OUT/f'fold{k}.npz') as z:
            q=z['q']
            np.testing.assert_array_equal(z['ids'],people.PassengerId.iloc[q])
            fine[q]=z['fine']
    with np.load(ART/'nov26_latent_passenger_difficulty/panel.npz') as z:
        np.testing.assert_array_equal(z['passenger'],people.PassengerId)
        med=np.median(z['scores'][:n,-3:],axis=1)
    part=partition(g[:n],'directional_foundation',4)
    baseline=versions['111'][0]
    risk=np.where(baseline,1-med,med)
    cuts=np.empty((n,2))
    for k in range(8):
        for side in range(4):
            t=np.flatnonzero(known&(f==k)&(part!=side));q=np.flatnonzero((f==k)&(part==side))
            cuts[q]=calibrate(risk,baseline,y,t,1.)
    args=dict(raw=p['neural_deck_raw'],original_fine=p['neural_deck_fine'],new_fine=fine[:n],
              six=challenger,families=full['votes'][:,:18],paid=paid)
    with np.load(ART/BASE/'scores.npz') as z:
        np.testing.assert_array_equal(z['passenger'],people.PassengerId.iloc[n:])
        test=dict(nine=z['anchor'],raw=z['raw'],original_fine=z['fine'],new_fine=fine[n:],
                  six=z['challengers'],families=z['family_votes'],paid=z['eligible'])
    old=predict(test['nine'],test['raw'],test['original_fine'],test['six'],test['families'],test['paid'])['joint']
    test111,h111=test_decisions(BASE,people.PassengerId.iloc[n:])
    testdir,hd=test_decisions(DIRECTIONAL,people.PassengerId.iloc[n:])
    champ=pd.read_csv(ART/DIRECTIONAL/'test_probabilities.csv',float_precision='round_trip').probability.to_numpy()
    np.testing.assert_array_equal(old>=.5,test111)
    np.testing.assert_array_equal(champ>=.5,testdir)
    with np.load(OUT/'posterior.npz') as z:
        deck,eligible=z['deck'],z['eligible']
        calc=(deck>=0)&eligible
        with np.load(DECK/'deck_estimates.npz') as previous:
            prior_deck=previous['fine'].argmax(1)
            new_deck=z['p'].argmax(1)
            restored=calc & (new_deck==deck) & (prior_deck!=deck)
            withdrawn=calc & (new_deck!=deck) & (prior_deck==deck)
            deck_correct=[int((v[calc]==deck[calc]).sum()) for v in (prior_deck,new_deck)]
        summary=dict(deck_correct=deck_correct,known_deck=int(calc.sum()),direct_checks=len(z['checked']),
                     max_error=float(z['errors'].max()),newly_correct_deck=int(restored.sum()),newly_wrong_deck=int(withdrawn.sum()),transport_correct=[int((((v>=.5)==y)).sum()) for v in (args['original_fine'],fine[:n])])
    rows={}
    all_scores={}
    for parent,protect in [('111',False),('directional',True)]:
        for tag,add,direction in [('replace',False,'both'),('add',True,'both'),('up',False,'up'),('down',False,'down')]:
            key=parent+'_'+tag
            target=champ if protect else old
            prob=update(**test,champion=target,protect=protect,only_add=add,direction=direction)
            all_scores[key]=prob
            eff=[];base_oof=[];action=[]
            for j,nine in enumerate(versions['nine']):
                pure=versions['111'][j]
                strength=np.where(pure,1-med,med)
                known_direction=(strength>=cuts[np.arange(n),pure.astype(int)])
                base=(pure^known_direction) if protect else pure
                pred=update(nine=nine,**args,champion=base,protect=protect,only_add=add,direction=direction)>=.5
                action.append(base!=pred)
                base_oof.append(base)
                eff.append((pred==y).astype(int)-(base==y))
            wrong=base_oof[0]!=y
            effect=np.stack(eff)
            np.testing.assert_array_equal(effect[0,known],effect[1,known])
            rows[key]=dict(test_changes=int(((prob>=.5)!=(target>=.5)).sum()),
                test_new=int((((prob>=.5)!=(target>=.5))&((old>=.5)==test['nine'])).sum()),
                reversals_on_test=int((((prob>=.5)!=(target>=.5))&((old>=.5)!=test['nine'])).sum()),
                scopes={s:count(action[0],m,wrong) for s,m in [('known',known),('fresh',fresh&known),('blind',blind),
                    ('paid',paid&known),('deck_newly_correct',restored[:n]&known),('deck_newly_wrong',withdrawn[:n]&known),('positive_parent',known&base_oof[0]),('negative_parent',known&~base_oof[0])]},
                full_net_range=[int(effect.min(0).sum()),int(effect.max(0).sum())],folds=[int(effect[0,known&(f==i)].sum()) for i in range(8)])
    np.savez_compressed(OUT/'test_scores.npz',ids=people.PassengerId.iloc[n:].to_numpy(str),**all_scores)
    REPORT.write_text(json.dumps(dict(summary=summary,replay=rows),indent=2)+'\n')
    print(summary,rows,flush=True)
    if export is not None:
        assert export in ('directional_replace','directional_add','directional_up','directional_down')
        dest=ART/NAME
        dest.mkdir(exist_ok=True)
        model=dest/'inputs.npz'
        fields=dict(**test,champion=champ,protect=True,only_add=export.endswith('_add'),direction={'directional_up':'up','directional_down':'down'}.get(export,'both'))
        np.savez_compressed(model,passenger=people.PassengerId.iloc[n:].to_numpy(str),**fields)
        pr=all_scores[export]
        with np.load(model) as z:replay=update(**{k:z[k] for k in fields})
        np.testing.assert_array_equal(pr,replay)
        ids=people.PassengerId.iloc[n:]
        pd.DataFrame(dict(PassengerId=ids,Transported=pr>=.5)).to_csv(dest/'submission.csv',index=False)
        pd.DataFrame(dict(PassengerId=ids,probability=pr,reload_probability=replay)).to_csv(dest/'test_probabilities.csv',index=False)
        check=validate_submission(dest/'submission.csv',ROOT/'data/raw/test.csv')
        assert all(json.loads(fl.read_text())['file_sha256']!=check['sha256'] or fl.parents[1].name==NAME for fl in (ROOT/'reports').glob('*/online_attempts/*.json'))
        manifest=dict(**check,version='n27_leave_order_fare',parents={BASE:h111,DIRECTIONAL:hd},training_rows=n,model_sha256=dig(model),reload_difference=0,
                      changed=rows[export]['test_changes'],source_file='evaluate_leave_booking_out_fare_n27.py')
        selection=dict(version=manifest['version'],policy=export,evidence=str(REPORT.relative_to(ROOT)))
        for path in [dest,ROOT/'reports'/NAME]:
            path.mkdir(exist_ok=True)
            write(path/'manifest.json',manifest);write(path/'selection.json',selection)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',choices=['directional_replace','directional_add','directional_up','directional_down'])
    run(parser.parse_args().export)
