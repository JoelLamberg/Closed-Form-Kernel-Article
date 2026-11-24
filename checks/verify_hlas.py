import json, glob, numpy as np
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R=[json.load(open(f)) for f in glob.glob(os.path.join(REPO, 'data', 'stronger_H', 'unz', 'results_map', 'pt_*.json'))]
ref=json.load(open(os.path.join(REPO, 'stronger_H', 'reference', 'map_cell_values.json')))
K=2*np.pi
print('cells',len(R),'unique',len({(r['i'],r['j']) for r in R}),'devices',{r['device'] for r in R})
bad=[r for r in R if not all(np.isfinite(r[k]) for k in ('err_dbl_sc','err_sclas_sc','err_sclpwa_sc'))]; print('nonfinite',len(bad))
# reproduce reference
dl=[];dd=[]
for r in R:
    q=ref['%03d_%03d'%(r['i'],r['j'])]
    dl.append(abs(r['err_sclpwa_sc']/q['err_sclpwa_sc']-1)); dd.append(abs(r['err_dbl_sc']/q['err_dbl_sc']-1))
dl=np.array(dl); dd=np.array(dd)
print('vs verified map: LPWA rel diff max %.1e (median %.1e); one-sided max %.1e, >1%%: %d'%(dl.max(),np.median(dl),dd.max(),(dd>0.01).sum()))
a=np.array([r['a'] for r in R]); s=np.array([r['s'] for r in R])
e1=np.array([r['err_dbl_sc'] for r in R]); e3=np.array([r['err_sclas_sc'] for r in R]); e2=np.array([r['err_sclpwa_sc'] for r in R])
fl=np.array([r['err_sc_exact'] for r in R])
u1=np.array([r.get('unc_dbl',np.nan) for r in R]); u3=np.array([r.get('unc_sclas',np.nan) for r in R])
r1=e1>10*fl; r3=e3>10*fl; both=r1&r3
print('resolved: one-sided',r1.sum(),'LAS2c',r3.sum(),'both',both.sum(),'either',(r1|r3).sum())
print('unc on resolved: one-sided max %.2e (p99 %.1e), LAS2c max %.2e (p99 %.1e)'%(u1[r1].max(),np.percentile(u1[r1],99),u3[r3].max(),np.percentile(u3[r3],99)))
i=np.nanargmax(u1); print(' worst one-sided unc cell a=%.2f s=%.3g e1=%.1e fl=%.1e resolved=%s'%(a[i],s[i],e1[i],fl[i],r1[i]))
i=np.nanargmax(u3); print(' worst LAS2c unc cell a=%.2f s=%.3g e3=%.1e fl=%.1e resolved=%s'%(a[i],s[i],e3[i],fl[i],r3[i]))
q=e3/e1
win=both&(q<1); print('LAS2c wins (both resolved):',win.sum(),'of',both.sum())
# robust win: ratio beyond combined uncertainty
rob=both&(q*(1+u3+u1)<1); print('LAS2c wins beyond uncertainty:',rob.sum())
lose=both&(q>1)
print('one-sided wins:',lose.sum())
# one resolved, other not
print('only one-sided resolved:',(r1&~r3).sum(),' only LAS2c resolved:',(r3&~r1).sum())
x=K*(a/np.where(s>0,s,np.nan))/(K*a/5.5)**2
for lo,hi in ((0,.1),(.1,.3),(.3,.6),(.6,1)):
    m=both&(s>=lo)&(s<hi)
    if m.any(): print(' a/Rc [%g,%g): n=%d wins %d median q %.2f min %.2f max %.2f'%(lo,hi,m.sum(),(q[m]<1).sum(),np.median(q[m]),q[m].min(),q[m].max()))
print('crossover in x=kRc/(kw0)^2:')
for lo,hi in ((0,.3),(.3,.6),(.6,1),(1,1.5),(1.5,2),(2,3),(3,10),(10,1e9)):
    m=both&(x>=lo)&(x<hi)
    if m.any(): print('  [%g,%g): n=%d LAS2c wins %d, median q %.2f'%(lo,hi,m.sum(),(q[m]<1).sum(),np.median(q[m])))
print('LAS2c beats LPWA everywhere?',(e3<e2)[r3].mean())
print('max LAS2c advantage (min q) %.3f at a=%.1f s=%.2f'%(q[both].min(),a[both][q[both].argmin()],s[both][q[both].argmin()]))
print('wins with x>2:',(win&(x>2)).sum(),' losses with x<0.5:',(lose&(x<0.5)).sum())
print('surface H: LAS2c/LPWA median %.2e'%np.median(np.array([r['surfH_err_las']/r['surfH_err_lpwa'] for r in R])))
print('ladder levels: ppw_used dist',np.unique([r['ppw_used'] for r in R],return_counts=True))
np.savez(os.path.join(REPO, 'data', 'stronger_H', 'map_las.npz'),a=a,s=s,i=[r['i'] for r in R],j=[r['j'] for r in R],e1=e1,e2=e2,e3=e3,fl=fl,u1=u1,u3=u3)
