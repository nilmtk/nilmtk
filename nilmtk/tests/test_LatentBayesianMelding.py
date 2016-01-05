#!/usr/bin/python
from __future__ import print_function, division
import unittest
from os.path import join
from os import remove
from testingtools import data_dir
from nilmtk.datastore import HDFDataStore
from nilmtk import DataSet
from nilmtk.disaggregate import FHMM

import pandas as pd
from pandas import HDFStore
#import matplotlib.pyplot as plt
from nilmtk.disaggregate.latent_Bayesian_melding import LatentBayesianMelding

class TestLatentBayesianMelding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        filename_meterdata_ukdale = join(data_dir(), 'meterdata_ukdale.h5')
        cls.meterdata_ukdale = HDFStore(filename_meterdata_ukdale)

    @classmethod
    def tearDownClass(cls):
        cls.meterdata_ukdale.close()

    def test_LatentBayesianMelding_correctness(self):
        """
        Created on Friday January 1 20:27:27 2016
        
        @author: Mingjun Zhong, School of Informatics, University of Edinburgh
        Email: mingjun.zhong@gmail.com
        
        Requirements: 
        1) the NILMTK: Non-Intrusive Load Monitoring Toolkit, 
                https://github.com/nilmtk/nilmtk;
        2) the (free academic) MOSEK license: 
                https://www.mosek.com/resources/academic-license 
        
        References:
         [1] Mingjun Zhong, Nigel Goddard and Charles Sutton. 
         Latent Bayesian melding for integrating individual and population models.
         In Advances in Neural Information Processing Systems 28, 2015.
         [2] Mingjun Zhong, Nigel Goddard and Charles Sutton. 
         Signal Aggregate Constraints in Additive Factorial HMMs, 
         with Application to Energy Disaggregation.
         In Advances in Neural Information Processing Systems 27, 2014.
         
        Note: 
        1)  The optimization problems described in the papers were transformed to
            a second order conic programming (SOCP) which suits the MOSEK solver; 
            I should write a technical report for this.
        2)  Any questions please drop me an email.
        """
        
        ############# some global variables ##########################
        # Sampling time was 2 minutes
        sample_seconds = 120
        
        ####### Select a house: this is the house 2 in UKDALE #########
        # This is a demo for applying Latent Bayesian Melding to energy disaggregation.
        # The HES data were used for training the model parameters, and the population
        # models. This demo shows how to use the LBM to disaggregate the mains readings
        # of a day in UKDALE.
        # Note that since HES data were read every 2 minutes, so UKDALE data were 
        # resampled to 2 minutes.
        # Note that all the readings were tranferred to the unit: deciwatt-hour, 
        # which is identical to the unit of HES data. 
        ################################################################
        
        # Note that this demo is only for a chunk (=a day).
        # This should be easy to amend for all the data you want to disaggregate.
        #########################################################################
        
        ## This is the building information: dataset/building_number/date
        building_information = 'ukdale/building2/2013-06-08'
        
        # Read the data from h5 file.
        # meterdata is a DataFrame and its columns are: 1) appliance readings;
        # 2) the mains readings; 3) the synthetic mains readings which are the sum
        # of all the appliance readings.
        meterdata = self.meterdata_ukdale['meterdata']
        
        # Map the keywords of appliances between HES and UKDALE
        appliance_map = {'cooker':"('cooker', 1)",
                         'kettle':"('kettle', 1)",
                         'dishwasher':"('dish washer', 1)",
                         'toaster':"('toaster', 1)",
                         'washingmachine':"('washing machine', 1)",
                         'fridgefreezer':"('fridge', 1)", 
                         'microwave':"('microwave', 1)"}
                         
        ########### Appliances to be disaggregated: not all of the appliance #########
        meterlist = ['cooker', 'kettle', 'dishwasher','toaster',
                     'washingmachine','fridgefreezer', 'microwave']
        
        ## the ground truth reading for those appliances to be disaggregated ########
        appliancedata = meterdata
        groundTruthApplianceReading = pd.DataFrame(index=meterdata.index)
        for meter in appliance_map:
            groundTruthApplianceReading[meter] = meterdata[appliance_map[meter]]
            appliancedata = appliancedata.drop(appliance_map[meter],axis=1)    
        ## the sum of other meter readings which will not be disaggregated
        groundTruthApplianceReading['othermeters'] = appliancedata.sum(axis=1)
        
        ## The mains readings to be disaggregated ###
        mains = meterdata['mains']
        
        #### declare an instance for lbm ################################
        lbm = LatentBayesianMelding()
        
        # to obtain the model parameters trained by using HES data
        individual_model = lbm.import_model(meterlist,join(data_dir(), 'appliance_model_induced_density.json'))
        
        # use lbm to disaggregate mains readings into appliances
        results = lbm.disaggregate_chunk(mains)
        
        # the inferred appliance readings
        infApplianceReading=results['inferred appliance energy']       


if __name__ == '__main__':
    unittest.main()
