"""Aggregate remaining hard cases and draw repair/collateral counts from caches."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/error_rescue_20261117'
LOCAL=ROOT/'artifacts/error_rescue_20261117'


def main():
    methods=pd.read_csv(OUT/'catalog_methods.csv')
    catalog=pd.read_csv(LOCAL/'catalog_rows.csv',dtype={'PassengerId':str})
    hard=pd.read_csv(LOCAL/'shared_error_cases.csv',dtype={'PassengerId':str})
    merged=hard.merge(catalog[['row_id','all_primary_still_wrong','all_recipes_still_wrong']],on='row_id',validate='one_to_one')
    profiles={}
    for name in ['all_primary_still_wrong','all_recipes_still_wrong']:
        x=merged[merged[name]]
        profiles[name]=dict(rows=len(x),planet_counts=x.HomePlanet.fillna('missing').value_counts().to_dict(),
            no_group_peers=int(x.GroupSize.eq(1).sum()),
            earth_asleep=int((x.HomePlanet.eq('Earth')&x.CryoSleep.eq(True)).sum()),
            limitation='Fixed recipe collection failure, not a noise floor or proof of mislabels.')
    v93=methods.set_index('method').loc['v93']
    profiles['v93_gain_decomposition_vs_v81']=dict(total_net_correct=int(v93.net_correct_vs_v81),
        five_wrong_repaired=int(v93.repaired_five_common_wrong),five_correct_broken=int(v93.damaged_five_common_correct),
        net_on_disagreement_rows=int(v93.net_correct_vs_v81-v93.repaired_five_common_wrong+v93.damaged_five_common_correct))
    (OUT/'remaining_error_profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    selected={'v70_SCARF':'SCARF','v22_mlp':'MLP','v68_TabR_S':'TabR','v67_FTTransformer_default':'FT-Transformer',
              'v41_TabM':'TabM','v65_TabPFN_default32':'TabPFN32','v77_Mitra_context20':'Mitra + context','v93':'Mitra long context'}
    data=methods[methods.method.isin(selected)]
    fig,ax=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    x=data.repaired_five_common_wrong.to_numpy();y=data.damaged_five_common_correct.to_numpy()
    ax[0].scatter(x,y,s=65,c=np.where(y>x,'#d65a4a','#2a9d8f'))
    ax[0].plot([0,145],[0,145],ls='--',color='0.6',lw=1,label='Equal counts')
    for (_,row),a,b in zip(data.iterrows(),x,y):
        offset=(5,5)
        if row.method=='v65_TabPFN_default32':offset=(5,10)
        if row.method=='v77_Mitra_context20':offset=(7,0)
        ax[0].annotate(selected[row.method],(a,b),xytext=offset,textcoords='offset points',fontsize=8)
    ax[0].set(xlim=(-5,175),ylim=(-5,270),xlabel='Correct among 914 previously shared errors',
        ylabel='Wrong among 5,263 previously shared successes',title='Repair comes with collateral errors')
    ax[0].text(.02,.97,'777 disagreement rows omitted here',transform=ax[0].transAxes,va='top',fontsize=8,color='0.4')
    counts=[298,616]
    bars=ax[1].bar(['Correct by at least one\nof 16 representatives','Still wrong for all\n16 representatives'],counts,color=['#2a9d8f','#d65a4a'],width=.6)
    for bar,n in zip(bars,counts):ax[1].text(bar.get_x()+bar.get_width()/2,n+8,str(n),ha='center',fontsize=13)
    ax[1].set(ylim=(0,720),ylabel='Passengers',title='Within the original 914 shared errors')
    ax[1].text(.5,.96,'Truth-aware union is diagnostic only,\nnot an achievable ensemble score',transform=ax[1].transAxes,ha='center',va='top',fontsize=9,color='0.4')
    fig.savefig(OUT/'repair_tradeoff.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
