import unittest
import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative
from spatial_offset_candidate import objective, fit_offset, predict, cells


class OffsetTests(unittest.TestCase):
    def test_eligibility_and_rank_bins(self):
        roster=pd.DataFrame({'HomePlanet':['Earth']*7,'Cabin':['G/1/P','G/2/P','G/3/P','G/4/P','G/1/S',None,'F/1/P'],
                             'CryoSleep':[False,False,True,False,False,False,False]})
        np.testing.assert_array_equal(cells(roster),[1,2,-1,3,7,-1,-1])

    def test_analytic_gradient(self):
        b=np.array([.2,-.1]); logits=np.array([-.4,.5,1.,-.3]); y=np.array([0,1,0,1]); cells=np.array([0,1,0,1])
        actual=objective(b,logits,y,cells)[1]
        numerical=approx_derivative(lambda x: objective(x,logits,y,cells)[0],b).ravel()
        np.testing.assert_allclose(actual,numerical,atol=1e-8)

    def test_excluded_rows_unchanged_and_empty_cells_zero(self):
        p=np.array([.2,.3,.4,.5,.6,.7]); y=np.array([0,0,1,1,1,1]); cells=np.array([-1,0,0,1,1,-1])
        fitted=fit_offset(p,y,cells)
        self.assertEqual(fitted['training_rows'],4)
        np.testing.assert_array_equal(np.array(fitted['beta'])[2:],0)
        predicted=predict(p,cells,fitted)
        np.testing.assert_array_equal(predicted[cells<0],p[cells<0])

    def test_no_eligible_training_rejected(self):
        with self.assertRaises(ValueError): fit_offset(np.array([.5,.5]),np.array([0,1]),np.array([-1,-1]))

    def test_noneligible_labels_do_not_affect_fit(self):
        p=np.array([.2,.3,.4,.5,.6,.7]); y=np.array([0,0,1,1,1,1]); groups=np.array([-1,0,0,1,1,-1])
        alternative=y.copy(); alternative[groups<0]=1-alternative[groups<0]
        np.testing.assert_array_equal(fit_offset(p,y,groups)['beta'],fit_offset(p,alternative,groups)['beta'])


if __name__=='__main__': unittest.main()
