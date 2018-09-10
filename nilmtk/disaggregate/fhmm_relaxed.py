from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame

import pandas as pd
import numpy as np
import json
from collections import OrderedDict
import timeit

# mosek fusion modules
import mosek.fusion
from mosek.fusion import *
from mosek.array import *

SEED = 42

# Fix the seed for repeatibility of experiments
np.random.seed(SEED)

# set the maximum depth of the Python interpreter stack.
sys.setrecursionlimit(100000)       

class FHMM_Relaxed(object):
    """Provides a common interface to all disaggregation classes.

    See https://github.com/nilmtk/nilmtk/issues/271 for discussion, and
    nilmtk/docs/manual/development_guide/writing_a_disaggregation_algorithm.md
    for the development guide.

    Attributes
    ----------
    model :
        Each subclass should internally store models learned from training.

    MODEL_NAME : string
        A short name for this type of model.
        e.g. 'CO' for combinatorial optimisation.
        Used by self._save_metadata_for_disaggregation.
    """
    ##########################################################################
    # This class implements the relaxed factorial hmm for
    # energy disaggregation. The fhmm 
    #
    # References:
    # [1] Mingjun Zhong, Nigel Goddard and Charles Sutton. 
    #     Latent Bayesian melding for integrating individual and population models.
    #     In Advances in Neural Information Processing Systems 28, 2015.
    # [2] Mingjun Zhong, Nigel Goddard and Charles Sutton. 
    #     Signal Aggregate Constraints in Additive Factorial HMMs, 
    #     with Application to Energy Disaggregation.
    #     In Advances in Neural Information Processing Systems 27, 2014.
    ##########################################################################
    
    def __init__(self):

        self.model = {}
        self.predictions = pd.DataFrame()
        self.trainMeterData = {}
        
        # initial the noise variances               
        self.varModel = 1.0
        self.varLatentModel = 1.0
        self.varSac = 1.0
        self.varDuration = 1.0
        self.varOffOnTran = 1.0
        self.varPieceWiseNoise = 1.0
        self.varPriorNosCycles = 1.0
        self.mains_chunk = 0
        # sampling seconds
        self.sample_seconds = 120
        
        # number of iterations
        self.NosOfIters = 3
        
        # shape parameter for Gamma distribution
        self.alpha = 1 + 1e-6
        # rate parameter for Gamma distribution
        self.beta = 1e-6
        
        # number of states
        self.numberOfStates = 3
        
        self.primalObjValue = 0.0
        self.dualObjValue = 0.0    
        
    def train(self, metergroup):
        """Trains the model given a metergroup containing appliance meters
        (supervised) or a site meter (unsupervised).  Will have a
        default implementation in super class.  Can be overridden for
        simpler in-memory training, or more complex out-of-core
        training.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        """
        raise NotImplementedError()

    def train_on_chunk(self, chunk, meter):
        """Signature is fine for site meter dataframes (unsupervised
        learning). Would need to be called for each appliance meter
        along with appliance identifier for supervised learning.
        Required to be overridden to provide out-of-core
        disaggregation.

        Parameters
        ----------
        chunk : pd.DataFrame where each column represents a
            disaggregated appliance
        meter : ElecMeter for this chunk
        """
        raise NotImplementedError()

    def disaggregate(self, mains, output_datastore):
        """Passes each chunk from mains generator to disaggregate_chunk() and
        passes the output to _write_disaggregated_chunk_to_datastore()
        Will have a default implementation in super class.  Can be
        overridden for more simple in-memory disaggregation, or more
        complex out-of-core disaggregation.

        Parameters
        ----------
        mains : nilmtk.ElecMeter (single-phase) or
            nilmtk.MeterGroup (multi-phase)
        output_datastore : instance of nilmtk.DataStore or str of
            datastore location
        """
        raise NotImplementedError()

    def disaggregate_chunk(self, mains_chunk):
        """In-memory disaggregation.

        Parameters
        ----------
        mains : pd.DataFrame

        Returns
        -------
        appliances : pd.DataFrame where each column represents a
            disaggregated appliance
        """
        
        # define the noise variances - either treat them as variables or constants
        # constants
        computetime = []
        self.mains_chunk = mains_chunk
        self.varLatentModel = {}
        self.varSac = {}
        for (i,appliance) in enumerate(self.individual):
            self.varLatentModel[appliance] = 1.0
            self.varSac[appliance] = 1.0
        
        objectiveOptimized = []
        for iters in range(self.NosOfIters):
            # infer the model variables
            prediction = self.disaggregate_fixedPrecision(mains_chunk)
            
            # estimate the regulation parameters - noise variance
            self.estimate_noisevariance(mains_chunk, prediction)
            
            # compute the objective
            optimalObjective = self.objective(mains_chunk, prediction)
            objectiveOptimized.append(optimalObjective)
            computetime.append(prediction['time'])
            # check results
            #self.checkconstraints(prediction)
        
        prediction = self.disaggregate_fixedPrecision(mains_chunk)     
        computetime.append(prediction['time'])
        # compute the objective
        optimalObjective = self.objective(mains_chunk, prediction)
        objectiveOptimized.append(optimalObjective)
        prediction['optimized objective'] = objectiveOptimized
        prediction['time'] = computetime
        return prediction        
        
        raise NotImplementedError()
        
    def disaggregate_fixedPrecision(self, mains_chunk):
        # This method is to disaggregate a chunk when fixing the precisions for
        # all the sub-models (constraints)
    
        print("Employing the Mosek solver to solve the problem:\n")
        print("Declaring variables and constraints...\n")
        # Mosek fusion for second-oder cone programming -- a composite model
        nosOfTimePoints = int(len(mains_chunk))       
        with Model("composite model") as M:

            # record computing time
            start_time = timeit.default_timer()            
            
            # define latent variable of appliance
            latentVariableOfAppliance = OrderedDict()
            
            # define state variables for each appliance
            stateVariableOfAppliance = OrderedDict()            

            # define relaxed variables related to state transitions
            relaxedVariableOfAppliance = OrderedDict()
            
            # define epigraph for latent variable term
            tauLatent = OrderedDict()
                                    
            # declare the variables and the constraints
            nosOfVariables = 0
            nosOfConstrs = 0
            for i, appliance in enumerate(self.individual):
                #print("\n Declare variables and constraints for appliance: '{}'"
                #        .format(appliance))
                        
                nosOfStates = int(self.individual[appliance]['numberOfStates'])                
                
                ######### necessary constant matrix for the model #############
                tempVector = np.zeros((nosOfStates,nosOfStates))
                tempVector[:,0][1:]=1.0
                constMatOffOn = np.kron(np.ones((1,nosOfTimePoints-1)),
                                        tempVector)
                
                ######### Declare the variables ###############################
                # declare latent variables 
                latentVariableOfAppliance[appliance] = \
                                M.variable(NDSet(1, nosOfTimePoints),
                                            Domain.greaterThan(0.0))
                                                                                       
                # declare the state variables of HMMs
                stateVariableOfAppliance[appliance] = \
                                M.variable(NDSet(nosOfStates, nosOfTimePoints),
                                           Domain.inRange(0.0, 1.0))
                
                # declare the relaxed variables for state transitions of HMMs 
                # variable H_i = [H_{i1},H_{i2},...,H_{i,T-1}] for the ith HMM
                relaxedVariableOfAppliance[appliance] = \
                M.variable(NDSet(nosOfStates,nosOfStates*(nosOfTimePoints-1)),
                           Domain.inRange(0.0, 1.0))

                ############## Make constraints for variables #################
                # make constraints on each appliance states
                # sum over state variables at time t to 1
                c_stateSumToOne = M.constraint( Expr.mul(np.ones(nosOfStates), 
                                        stateVariableOfAppliance[appliance]), 
                                        Domain.equalsTo(1.0) )
                                        
                # make constraints on the relaxed variables and the state variables
                # the relaxed variable should be constrained to match state variables
                # For the summation of rows
                c_relaxStateSumRow = M.constraint(
                        Expr.sub( Expr.mul(np.ones((1,nosOfStates)), 
                        relaxedVariableOfAppliance[appliance]), 
                        Variable.reshape(stateVariableOfAppliance[appliance].
                        slice([0,0],[nosOfStates,nosOfTimePoints-1]).transpose(), 
                        1, nosOfStates*(nosOfTimePoints-1)) ), 
                        Domain.equalsTo(0.0))
                        
                # For the summation of collumns
                for j in range(nosOfStates):
                    row_relaxedV = relaxedVariableOfAppliance[appliance].\
                        slice([j,0],[j+1,nosOfStates*(nosOfTimePoints-1)])
                    row_relaxedV = Variable.reshape(row_relaxedV,
                                                 nosOfTimePoints-1,nosOfStates)
                    if j == 0:
                        vtemp_hstack = row_relaxedV
                    else:
                        vtemp_hstack = Variable.hstack(vtemp_hstack,row_relaxedV)
                vtemp = Variable.reshape(vtemp_hstack,
                                nosOfStates*(nosOfTimePoints-1),nosOfStates)
                c_relaxStateSumColumn = M.constraint( 
                        Expr.sub( Expr.mul(vtemp, np.ones((nosOfStates,1))),
                        Variable.reshape(stateVariableOfAppliance[appliance].
                        slice([0,1],[nosOfStates,nosOfTimePoints]).transpose(),
                        1,nosOfStates*(nosOfTimePoints-1)).transpose()), 
                        Domain.equalsTo(0.0) )
                    
                ######### Declare the rotated quadratic cone ##################                
                # define the rotated quadratic cone for latent variables
                # connected to the HMMs
                tauLatent[appliance] = M.variable(1, Domain.greaterThan(0.0))
                diffLatent = Expr.sub(latentVariableOfAppliance[appliance],
                    Expr.mul(1.0*np.array(self.individual[appliance]['means']).
                             reshape((1,nosOfStates)),
                             stateVariableOfAppliance[appliance]))
                rqc_latent = M.constraint(Expr.hstack(
                    Expr.constTerm(self.varLatentModel[appliance]),
                    tauLatent[appliance].asExpr(),
                    diffLatent), Domain.inRotatedQCone())
                
                ############ The objective functions #########################                    
                # objective function for the initial probability
                if i == 0:
                    sumInitialProb = Expr.dot(stateVariableOfAppliance[appliance].
                        slice([0,0],[nosOfStates,1]), 
                        -np.log(np.maximum(1e-300,
                    np.array(self.individual[appliance]['startprob']))).flatten())
                else:
                    sumInitialProb = Expr.add( sumInitialProb, 
                        Expr.dot(stateVariableOfAppliance[appliance].
                        slice([0,0],[nosOfStates,1]), 
                        -np.log(np.maximum(1e-300,
                    np.array(self.individual[appliance]['startprob']))).flatten()))
            
                # objective function for the transition probability matrix            
                if i == 0:
                    sumTransProb = Expr.sum(Expr.mulElm( 
                        relaxedVariableOfAppliance[appliance], 
                        -np.log(np.maximum(1e-300,
                                np.kron(np.ones((1,nosOfTimePoints-1)), 
                                self.individual[appliance]['transprob'])))) )
                else:
                    sumTransProb = Expr.add( sumTransProb,Expr.sum(Expr.mulElm( 
                        relaxedVariableOfAppliance[appliance], 
                        -np.log(np.maximum(1e-300,
                        np.kron(np.ones((1,nosOfTimePoints-1)), 
                        self.individual[appliance]['transprob']))) ) ) )
                                                             
                # Summation of the latent variables for all appliances accross time
                if i == 0:
                    sumLatentVariable = latentVariableOfAppliance[appliance]
                else:
                    sumLatentVariable = Expr.add(sumLatentVariable,
                                        latentVariableOfAppliance[appliance])
                ###############################################################
                # Forming objective function by adding up the epigraphs #####
                if i == 0:
                    sumTauLatent = tauLatent[appliance]
                else:
                    sumTauLatent = Expr.add(sumTauLatent,tauLatent[appliance])
                       
                ##### Counting the number of variables #######################
                nosOfVariables = nosOfVariables \
                                + stateVariableOfAppliance[appliance].size() \
                                + relaxedVariableOfAppliance[appliance].size()\
                                + latentVariableOfAppliance[appliance].size() \
                                + tauLatent[appliance].size()
                                
                ##### Counting the number of constraints #####################             
                nosOfConstrs = nosOfConstrs \
                                + c_stateSumToOne.size() \
                                + c_relaxStateSumRow.size() \
                                + c_relaxStateSumColumn.size()\
                                + rqc_latent.size()
            
            #################################################################            
            ######### The variable of total variation regularization ######## 
            ######### or piece-wise variable ################################
            variableOfPiecewise = M.variable(NDSet(1, nosOfTimePoints),
                                            Domain.greaterThan(0.0))
            # Log Laplacian distribution on piece-wise variable (the variation)
            # using quadratic cones
            tauPiecewise = M.variable(NDSet(1, nosOfTimePoints-1),
                                            Domain.greaterThan(0.0))
            
            diffPiecewise = Expr.sub(
                variableOfPiecewise.slice([0,1],[1,nosOfTimePoints]),
                variableOfPiecewise.slice([0,0],[1,nosOfTimePoints-1]))  
            qc_piecewise = M.constraint(
                Expr.hstack(Variable.reshape(tauPiecewise,nosOfTimePoints-1,1),
                            Expr.reshape(diffPiecewise,NDSet(nosOfTimePoints-1,1))), 
                            Domain.inQCone(nosOfTimePoints-1,2))
            ########## another way to form product of quadratic cone sets#######
            #v = Variable.stack([ [tauPiecewise.index(t),
            #                      Expr.sub(variableOfPiecewise.index(t+1),
            #                      variableOfPiecewise.index(t))] 
            #                      for t in range(nosOfTimePoints-1)])
            #qc_piecewise = M.constraint(v, Domain.inQCone(nosOfTimePoints-1,2))
            #################################################################
            nosOfVariables = nosOfVariables + variableOfPiecewise.size() \
                             + tauPiecewise.size()
            nosOfConstrs = nosOfConstrs + qc_piecewise.size()
                
            sumTauPiecewise = Expr.sum(Expr.mul(1/(2.0*self.varPieceWiseNoise),
                                                tauPiecewise))  
                                                
            ###################################################################
            ###### The log data likelihood ###################################     
            delta = M.variable('logdatalikelihood', 1, Domain.greaterThan(0.0))
            rqc_logDataLikelihood = M.constraint(
                Expr.hstack(self.varModel,delta,
                    Expr.sub(DenseMatrix(
                        mains_chunk.values.ravel().reshape(1,nosOfTimePoints)),
                             Expr.add(sumLatentVariable,variableOfPiecewise))),
                Domain.inRotatedQCone())
                
            nosOfVariables = nosOfVariables + delta.size()
            nosOfConstrs = nosOfConstrs + rqc_logDataLikelihood.size()

            ######### Performing the optimization ##########################
            # the objective function to minimize
            #print("\n The final objective function")
            M.objective('objectiveFunction', 
                        ObjectiveSense.Minimize, 
                        Expr.sum(Expr.vstack([sumInitialProb,
                                              sumTransProb,
                                              delta.asExpr(),
                                              sumTauLatent,
                                              sumTauPiecewise]
                                              )))
            
            # solving the problem
            print("\n Solving the problem ...")
            
            # This defines which solution status values are accepted 
            # when fetching solution values
            M.acceptedSolutionStatus(AccSolutionStatus.Anything)
            M.solve()
            
            ####### Print the optimization status ############################
            print("\n ++++++++++++++++++++++++++++++++++")
            print("Number of variables:{}".format(nosOfVariables))
            print("Number of constraints:{}".format(nosOfConstrs))
            print("Primal solution status:{}".format(M.getPrimalSolutionStatus()))
            print("Primal value:{}".format(M.primalObjValue()))
            print("Dual solution status:{}".format(M.getDualSolutionStatus()))
            print("Dual value:{}".format(M.dualObjValue()))
            print("Accepted solution status:{}".format(M.acceptedSolutionStatus()))
            print("\n ++++++++++++++++++++++++++++++++++")
            ###################################################################
            
            self.primalObjValue = M.primalObjValue()
            self.dualObjValue = M.dualObjValue()

            # recording the computing time
            stop_time = timeit.default_timer()
            print("Solving this problem took '{0}' seconds".format(stop_time-start_time))           

            ###### Reading the inference results ##############################
            inferred_appliance_mains_energy = pd.DataFrame(index=mains_chunk.index)
            inferred_latent_energy = pd.DataFrame(index=mains_chunk.index)
            inferred_states = OrderedDict()
            inferred_relaxedStates = OrderedDict()
            inferred_total_energy = OrderedDict()
            prediction = {}
            for (i, appliance) in enumerate(self.individual):
                nosOfStates = int(self.individual[appliance]['numberOfStates'])
                ######### necessary constant matrix for the model #############
                tempVector = np.zeros((nosOfStates,nosOfStates))
                tempVector[:,0][1:]=1.0
                constMatOffOn = np.kron(np.ones((1,nosOfTimePoints-1)),
                                        tempVector)
                                        
                inferred_appliance_mains_energy[appliance] = \
                    np.dot(np.array(self.individual[appliance]['means']).flatten(), 
                    np.reshape(np.array(stateVariableOfAppliance[appliance].level()),
                               (nosOfStates,nosOfTimePoints)))
                               
                inferred_latent_energy[appliance] = \
                    latentVariableOfAppliance[appliance].level()
                    
                # the inferred variables
                inferred_states[appliance] = \
                    np.reshape(np.array(stateVariableOfAppliance[appliance].level()),
                               (nosOfStates,nosOfTimePoints))
                    
                inferred_relaxedStates[appliance] = \
                    np.reshape(np.array(relaxedVariableOfAppliance[appliance].level()),
                               constMatOffOn.shape)
                               
            inferred_appliance_mains_energy['inferred mains'] = \
                inferred_appliance_mains_energy.iloc[:,0:len(self.individual)].\
                sum(axis=1).values
            inferred_appliance_mains_energy['mains'] = mains_chunk.values
                            
            inferred_latent_energy['inferred mains'] = \
                inferred_latent_energy.iloc[:,0:len(self.individual)].sum(axis=1).values
            inferred_latent_energy['mains'] = mains_chunk.values
            inferred_latent_energy['piecewise noise'] = \
                variableOfPiecewise.level()                
                
            prediction['inferred appliance energy'] = inferred_appliance_mains_energy
            prediction['inferred latent energy'] = inferred_latent_energy
            prediction['inferred states'] = inferred_states
            prediction['inferred relaxed states'] = inferred_relaxedStates
            prediction['inferred total energy'] = inferred_appliance_mains_energy.sum(axis=0).to_dict()
            prediction['time'] = stop_time-start_time
            return prediction

    def estimate_noisevariance(self, mains_chunk, prediction):
        # estimate the noise variances
        nosOfTimePoints = int(len(mains_chunk))
        inferred_appliance_mains_energy = prediction['inferred appliance energy']
        inferred_latent_energy = prediction['inferred latent energy']
        for (i, appliance) in enumerate(self.individual):
            nosOfStates = int(self.individual[appliance]['numberOfStates'])
            if i == 0:
                sumAppliance = inferred_appliance_mains_energy[appliance]
            else:
                sumAppliance = sumAppliance + \
                                    inferred_appliance_mains_energy[appliance]
                
            self.varLatentModel[appliance] = (self.beta +
                0.5*np.sum((inferred_latent_energy[appliance]
                -inferred_appliance_mains_energy[appliance])**2))\
                /(0.5*nosOfTimePoints+self.alpha-1)
            #print('Latent noise Var:{}\n'.format(self.varLatentModel[appliance]))
                            
        # Estimate the model noise variance
        self.varModel = (self.beta + 
           0.5*np.sum((mains_chunk.values.ravel().reshape(1,nosOfTimePoints)
           - inferred_latent_energy['inferred mains'].values 
           - inferred_latent_energy['piecewise noise'].values )**2))\
           /(0.5*nosOfTimePoints+self.alpha-1)
        #print('Model noise Var:{}\n'.format(self.varModel))
        
        # Estimate the noise variance for piecewise prior
        self.varPieceWiseNoise = (self.beta +
            0.5*np.sum(np.abs(inferred_latent_energy['piecewise noise'].\
            iloc[1:nosOfTimePoints].values
            -inferred_latent_energy['piecewise noise'].\
            iloc[0:nosOfTimePoints-1].values)))\
            /(nosOfTimePoints-1+self.alpha-1)
        #print('Piecewise Noise Var:{}\n'.format(self.varPieceWiseNoise))

    def objective(self, mains_chunk, prediction):
        # estimate the noise variances
        nosOfTimePoints = int(len(mains_chunk))
        inferred_appliance_mains_energy = prediction['inferred appliance energy']
        inferred_states = prediction['inferred states']
        inferred_relaxedStates = prediction['inferred relaxed states']
        inferred_latent_energy = prediction['inferred latent energy']
        optimalObjective = 0.0
        for (i, appliance) in enumerate(self.individual):
            nosOfStates = int(self.individual[appliance]['numberOfStates'])
            if i == 0:
                sumAppliance = inferred_appliance_mains_energy[appliance]
            else:
                sumAppliance = sumAppliance + \
                                inferred_appliance_mains_energy[appliance]
                
            ############# latent variable objective #####################
            optimalObjective = optimalObjective\
                -0.5*nosOfTimePoints*np.log(self.varLatentModel[appliance])\
                -0.5*(1.0/self.varLatentModel[appliance])*\
                                    np.sum((inferred_latent_energy[appliance]
                -inferred_appliance_mains_energy[appliance])**2)\
                -(self.alpha-1.0)*np.log(self.varLatentModel[appliance])\
                -self.beta*(1.0/self.varLatentModel[appliance])

            ########## initial probability ################################
            optimalObjective = optimalObjective\
                +np.dot(inferred_states[appliance][:,1].flatten(), 
                        np.log(np.maximum(1e-300,
                np.array(self.individual[appliance]['startprob']))).flatten())
                    
            ########## transition probabilities ##########################
            optimalObjective = optimalObjective\
                +np.sum(np.multiply(inferred_relaxedStates[appliance], 
                        np.log(np.maximum(1e-300,
                                np.kron(np.ones((1,nosOfTimePoints-1)), 
                                self.individual[appliance]['transprob'])))) )
                                
        # The data likelihood and prior
        optimalObjective = optimalObjective \
            -0.5*nosOfTimePoints*np.log(self.varModel)-0.5*(1/self.varModel)*\
            np.sum((mains_chunk.values.ravel().reshape(1,nosOfTimePoints)
           - inferred_latent_energy['inferred mains'].values 
           - inferred_latent_energy['piecewise noise'].values )**2)\
           - (self.alpha-1.0)*np.log(self.varModel) - self.beta*(1/self.varModel)               
        
        # The piecewise prior
        optimalObjective = optimalObjective \
            -(nosOfTimePoints-1)*np.log(self.varPieceWiseNoise)\
            -0.5*(1/self.varPieceWiseNoise)*\
                    np.sum(np.abs(inferred_latent_energy['piecewise noise'].\
                    iloc[1:nosOfTimePoints].values
            -inferred_latent_energy['piecewise noise'].\
                    iloc[0:nosOfTimePoints-1].values))\
            -(self.alpha-1)*np.log(self.varPieceWiseNoise) \
            - self.beta*(1/self.varPieceWiseNoise)            
                 
        print("\n Objective:{}\n".format(optimalObjective))
        return optimalObjective
            
    def _pre_disaggregation_checks(self, load_kwargs):
        if not self.model:
            raise RuntimeError(
                "The model needs to be instantiated before"
                " calling `disaggregate`.  For example, the"
                " model can be instantiated by running `train`.")

        if 'resample_seconds' in load_kwargs:
            DeprecationWarning("'resample_seconds' is deprecated."
                               "  Please use 'sample_period' instead.")
            load_kwargs['sample_period'] = load_kwargs.pop('resample_seconds')

        return load_kwargs

    def _save_metadata_for_disaggregation(self, output_datastore,
                                          sample_period, measurement,
                                          timeframes, building,
                                          meters=None, num_meters=None,
                                          supervised=True):
        """Add metadata for disaggregated appliance estimates to datastore.

        This method returns nothing.  It sets the metadata
        in `output_datastore`.

        Note that `self.MODEL_NAME` needs to be set to a string before
        calling this method.  For example, we use `self.MODEL_NAME = 'CO'`
        for Combinatorial Optimisation.

        Parameters
        ----------
        output_datastore : nilmtk.DataStore subclass object
            The datastore to write metadata into.
        sample_period : int
            The sample period, in seconds, used for both the
            mains and the disaggregated appliance estimates.
        measurement : 2-tuple of strings
            In the form (<physical_quantity>, <type>) e.g.
            ("power", "active")
        timeframes : list of nilmtk.TimeFrames or nilmtk.TimeFrameGroup
            The TimeFrames over which this data is valid for.
        building : int
            The building instance number (starting from 1)
        supervised : bool, defaults to True
            Is this a supervised NILM algorithm?
        meters : list of nilmtk.ElecMeters, optional
            Required if `supervised=True`
        num_meters : int
            Required if `supervised=False`
        """

        # TODO: `preprocessing_applied` for all meters
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.

        # DataSet and MeterDevice metadata:
        building_path = '/building{}'.format(building)
        mains_data_location = building_path + '/elec/meter1'

        meter_devices = {
            self.MODEL_NAME : {
                'model': self.MODEL_NAME,
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            },
            'mains': {
                'model': 'mains',
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            }
        }

        merged_timeframes = merge_timeframes(timeframes, gap=sample_period)
        total_timeframe = TimeFrame(merged_timeframes[0].start,
                                    merged_timeframes[-1].end)

        date_now = datetime.now().isoformat().split('.')[0]
        dataset_metadata = {
            'name': self.MODEL_NAME,
            'date': date_now,
            'meter_devices': meter_devices,
            'timeframe': total_timeframe.to_dict()
        }
        output_datastore.save_metadata('/', dataset_metadata)

        # Building metadata

        # Mains meter:
        elec_meters = {
            1: {
                'device_model': 'mains',
                'site_meter': True,
                'data_location': mains_data_location,
                'preprocessing_applied': {},  # TODO
                'statistics': {
                    'timeframe': total_timeframe.to_dict()
                }
            }
        }

        def update_elec_meters(meter_instance):
            elec_meters.update({
                meter_instance: {
                    'device_model': self.MODEL_NAME,
                    'submeter_of': 1,
                    'data_location': (
                        '{}/elec/meter{}'.format(
                            building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict()
                    }
                }
            })

        # Appliances and submeters:
        appliances = []
        if supervised:
            for meter in meters:
                meter_instance = meter.instance()
                update_elec_meters(meter_instance)

                for app in meter.appliances:
                    appliance = {
                        'meters': [meter_instance],
                        'type': app.identifier.type,
                        'instance': app.identifier.instance
                        # TODO this `instance` will only be correct when the
                        # model is trained on the same house as it is tested on
                        # https://github.com/nilmtk/nilmtk/issues/194
                    }
                    appliances.append(appliance)

                # Setting the name if it exists
                if meter.name:
                    if len(meter.name) > 0:
                        elec_meters[meter_instance]['name'] = meter.name
        else:  # Unsupervised
            # Submeters:
            # Starts at 2 because meter 1 is mains.
            for chan in range(2, num_meters + 2):
                update_elec_meters(meter_instance=chan)
                appliance = {
                    'meters': [chan],
                    'type': 'unknown',
                    'instance': chan - 1
                    # TODO this `instance` will only be correct when the
                    # model is trained on the same house as it is tested on
                    # https://github.com/nilmtk/nilmtk/issues/194
                }
                appliances.append(appliance)

        building_metadata = {
            'instance': building,
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)

    def _write_disaggregated_chunk_to_datastore(self, chunk, datastore):
        """ Writes disaggregated chunk to NILMTK datastore.
        Should not need to be overridden by sub-classes.

        Parameters
        ----------
        chunk : pd.DataFrame representing a single appliance
            (chunk needs to include metadata)
        datastore : nilmtk.DataStore
        """
        raise NotImplementedError()

    def import_model(self, meterlist, filename):
        """Loads learned model from file.
        Required to be overridden for learned models to persist.

        Parameters
        ----------
        filename : str path to file to load model from
        """
        # input parameter: 
        # meterlist is a list of appliances for disaggregation
        with open(filename) as infile:
            self.model = json.load(infile)

        # read the model prameters for the meterlist
        individual = OrderedDict()
        for meter in meterlist:
            if meter in self.model.keys():
                individual[meter] = self.model[meter]
                individual[meter]['numberOfStates'] = len(self.model[meter]['means'])
            else:
                print("The meter {0} is not in the trained model".format(meter))
            print("The trained meter: {0}".format(meter))
        self.individual = individual
        return self.individual
    
        raise NotImplementedError()

    def export_model(self, filename):
        """Saves learned model to file.
        Required to be overridden for learned models to persist.

        Parameters
        ----------
        filename : str path to file to save model to
        """
        raise NotImplementedError()

    def checkconstraints(self,prediction):
        """
        Check if the constraints have been satisfied or not
        """
        # Check the constraints on the state variables and relaxed variables
        stateVariableSumToOne = OrderedDict()
        relaxVariableSumRow = OrderedDict()
        relaxVariableSumColumn = OrderedDict()
        for (i, appliance) in enumerate(self.individual):
            infStates = prediction['inferred states'][appliance]
            [nosOfStates,nosOfTime] = np.shape(infStates)
            infRelaxStates = prediction['inferred relaxed states'][appliance]
            stateVariableSumToOne[appliance] = np.sum(infStates,axis=0)
            relaxVariableSumRow[appliance] = np.array(
                [np.reshape(infStates[:,0:nosOfTime-1],nosOfStates*(nosOfTime-1),1),
                 np.sum(infRelaxStates,axis=0)])
            for j in range(nosOfStates):
                if j==0:
                    relaxVariableStack = np.reshape(infRelaxStates[j,:],(nosOfTime-1,nosOfStates))
                else:
                    relaxVariableStack = np.hstack((relaxVariableStack,np.reshape(infRelaxStates[j,:],(nosOfTime-1,nosOfStates))))
            sumRelaxCol = np.sum(np.reshape(relaxVariableStack,(nosOfStates*(nosOfTime-1),nosOfStates)),axis=1)
            relaxVariableSumColumn[appliance] = np.array(
                [np.reshape(infStates[:,1:nosOfTime],nosOfStates*(nosOfTime-1),1),
                 sumRelaxCol])
        constraints = {}
        constraints['stateVariableSumToOne'] = stateVariableSumToOne
        constraints['relaxVariableSumRow'] = relaxVariableSumRow
        constraints['relaxVariableSumColumn'] = relaxVariableSumColumn        
        
        ########### Check the error function in objective####################
        nosOfTimePoints=nosOfTime
        inferred_appliance_mains_energy = prediction['inferred appliance energy']
        inferred_latent_energy = prediction['inferred latent energy']
        objLatent = {}
        for (i, appliance) in enumerate(self.individual):
            nosOfStates = int(self.individual[appliance]['numberOfStates'])
            if i == 0:
                sumAppliance = inferred_appliance_mains_energy[appliance]
            else:
                sumAppliance = sumAppliance + inferred_appliance_mains_energy[appliance]
                
            objLatent[appliance] = np.sum((inferred_latent_energy[appliance]
                                -inferred_appliance_mains_energy[appliance])**2)
            print("\n {0} latent error:{1}".format(appliance,objLatent[appliance]))
            
        # Estimate the model noise variance
        objModel = np.sum((self.mains_chunk.values.ravel().reshape(1,nosOfTimePoints)
                           - inferred_latent_energy['inferred mains'].values 
                           - inferred_latent_energy['piecewise noise'].values )**2)
        print("\n Model error:{}".format(objModel))
        
        # Estimate the noise variance for piecewise prior
        objPiecewise = np.sum(np.abs(inferred_latent_energy['piecewise noise'].iloc[1:nosOfTimePoints].values
            -inferred_latent_energy['piecewise noise'].iloc[0:nosOfTimePoints-1].values))
        print("\n Piecewise error:{}\n".format(objPiecewise))
        
        constraints['obj model'] = objModel
        constraints['obj piecewise'] = objPiecewise
        constraints['obj latent'] = objLatent
        
        return constraints