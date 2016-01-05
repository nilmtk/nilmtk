from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame

from nilmtk.utils import find_nearest
import pandas as pd
import itertools
import numpy as np
from sklearn import metrics
from hmmlearn import hmm
import pandas as pd
import numpy as np
import json
from datetime import datetime
from nilmtk.appliance import ApplianceID
from nilmtk.utils import find_nearest, container_to_string
from nilmtk.timeframe import merge_timeframes, list_of_timeframe_dicts, TimeFrame
from nilmtk.preprocessing import Apply, Clip

from copy import deepcopy
from collections import OrderedDict
import timeit

# mosek fusion modules
import mosek.fusion
from mosek.fusion import *
from mosek.array import *

######
SEED = 42

# Fix the seed for repeatibility of experiments
np.random.seed(SEED)

# set the maximum depth of the Python interpreter stack.
sys.setrecursionlimit(100000)   

class LatentBayesianMelding(object):
    """This class is derived from a common interface `disaggregator' in NILMTK.

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
    # This class implements the method called latent Bayesian melding (LBM) for
    # energy disaggregation. LBM is a Bayesian method to integrate 
    # individual model and population models, which are defined following:
    # 1) individual model: the energy disaggregation model - given a time 
    #   series of mains reading, to infer the appliance readings
    # 2) population models: the models for the summary statistics of appliances,
    #   for example, the distribution of how much energy was used in a day for
    #    a kettle; the distribution of how many times a kettle was used in a day.
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
        """ Initialize the parameters for the models and set some constant 
        values for the model.
        """
        # Import the trained models using method `import_model'. 
        # Note that the trained model was trained using the HES data.
        # For HES data, please refer to:
        #    J.-P. Zimmermann,et al.   Household electricity survey, 2012,
        #    https://www.gov.uk/government/collections/household-electricity-survey.
        self.model = {}        
        #self.predictions = pd.DataFrame()
        #self.trainMeterData = {}
        
        # initialize the noise variances defined in the latent Bayesian model         
        # initialize the model noise variance (energy diaggregation model);
        # to be more clear, Y~Normal(\sum_i^IX_i+U,\sigma^2)          
        self.varModel = 1.0
        
        # initialize the noise variance for the model for modeling X in the paper;
        # to be more clear, appliance X_i~Normal(S'\mu, \sigma_i^2)
        self.varLatentModel = 1.0
        
        # initialize the noise variance for modelling the summary statistics.
        self.varSac = 1.0
        self.varDuration = 1.0
        self.varOffOnTran = 1.0
        
        # initialize the noise variance for modelling piecewide variable U.
        self.varPieceWiseNoise = 1.0
        self.varPriorNosCycles = 1.0
        self.mains_chunk = 0
        
        # sampling seconds of the mains readings - 2 minute here.
        self.sample_seconds = 120
        
        # number of iterations for updating noise variances and model variables.
        self.NosOfIters = 3
        
        # shape and rate parameters for Gamma distributions for noise variances.
        self.alpha = 1 + 1e-6
        self.beta = 1e-6
        
        # number of states of HMM, more could be better?
        self.numberOfStates = 3
        
        # optimization objective values
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

    def disaggregate(self, mains, output_datastore, **load_kwargs):
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

        import warnings
        import datetime
        warnings.filterwarnings("ignore", category=Warning)
        MIN_CHUNK_LENGTH =100
        
        # Extract optional parameters from load_kwargs
        date_now = datetime.datetime.now().isoformat().split('.')[0]
        output_name = load_kwargs.pop('output_name', 'NILMTK_AFHMM_SAC_' + date_now)
        resample_seconds = load_kwargs.pop('resample_seconds', self.sample_seconds)
        self.sample_seconds = resample_seconds

        resample_rule = '{:d}S'.format(resample_seconds)
        timeframes = []       

        for chunk in mains.power_series(**load_kwargs):

            # Check that chunk is sensible size before resampling
            if len(chunk) < MIN_CHUNK_LENGTH:
                continue
            
            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            chunk = chunk.resample(rule=resample_rule).dropna()
            chunk = chunk*(((self.sample_seconds/60.0)/60.0)*10.0)#For uk-dale data
            
            # Check chunk size *again* after resampling
            if len(chunk) < MIN_CHUNK_LENGTH:
                continue
            
            # start doing disaggregation
            inferredVariables = self.disaggregate_chunk(chunk)
            return inferredVariables        
        
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
        '''
        perform disaggregation on a chunk data
        
        Parameters:
        ------------
        mains_chunk: pandas.core.series.Series (time series)
        return: prediction
        '''
        # define the noise variances - either treat them as variables or constant
        # constants
        computetime = []
        self.mains_chunk = mains_chunk
        self.varLatentModel = {}
        self.varSac = {}
        self.varDuration = {}
        self.varPriorNosCycles = {}
        for (i,appliance) in enumerate(self.individual):
            self.varLatentModel[appliance] = 1.0
            self.varDuration[appliance] = 1.0
            self.varSac[appliance] = 1.0
            #self.varOffOnTran[appliance] = 1.0
            self.varPriorNosCycles[appliance] = 1.0
        
        objectiveOptimized = []
        for iters in range(self.NosOfIters):
            print("*************Iteration: {}*****************".format(iters+1))
            # infer the model variables
            prediction = self.disaggregate_fixedPrecision(mains_chunk)
            
            # estimate the regulation parameters - noise variance
            self.estimate_noisevariance(mains_chunk, prediction)
            
            # compute the objective
            optimalObjective = self.objective(mains_chunk, prediction)
            objectiveOptimized.append(optimalObjective)
            
            # check results
            # self.checkconstraints(prediction)
            computetime.append(prediction['time'])
            
        print("*************Iteration: {}*****************".format(iters+2))
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
        # Mosek fusion for second-oder cone programming
        nosOfTimePoints = int(len(mains_chunk))       
        with Model("composite model") as M:

            # record computing time
            start_time = timeit.default_timer()            
            
            # define latent variable of appliance
            latentVariableOfAppliance = OrderedDict()
            
            # define the variables of number of activity cycles
            variableOfNosOfCycles = OrderedDict()
            
            # define state variables for each appliance
            stateVariableOfAppliance = OrderedDict()            

            # define relaxed variables related to state transitions
            relaxedVariableOfAppliance = OrderedDict()
            
            # define epigraph for latent variable term
            tauLatent = OrderedDict()
            
            # define epigraph for sac term
            tauSac = OrderedDict()
            
            # define epigraph for activity duration
            tauDuration = OrderedDict()
            
            # define epigraph for number of OFF to ON variable
            tauOffOn = OrderedDict()
            
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
                                            
                # declare the variable of number of cycles   
                nosOfMaxCycles = len(self.individual[appliance] \
                                    ['numberOfCyclesStats']['numberOfCycles'])
                variableOfNosOfCycles[appliance] = \
                                M.variable(NDSet(1, nosOfMaxCycles),
                                           Domain.inRange(0.0,1.0))
                                           
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
                    
                # make constraint on the variable of number of cycles
                c_nosCycleSumToOne = M.constraint(
                        Expr.sum(variableOfNosOfCycles[appliance]),
                        Domain.equalsTo(1.0))
                
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
                
                # the rotated quadratic cone constrait for the signal aggregate
                # declare the epigraph for the signal aggregate constraints
                tauSac[appliance] = M.variable(1, Domain.greaterThan(0.0))
                                   
                # employing the induced density: 
                #   the difference between the estimated and trained sac values
                jointVariance = 1/(1/self.varSac[appliance] - 
                    1/(self.individual[appliance]['induced density of sac'][1]**2))
                densityMean = Expr.dot(variableOfNosOfCycles[appliance],
                    np.array(self.individual[appliance] 
                    ['numberOfCyclesStats']['numberOfCyclesEnergy']))
                jointMean = Expr.sub(Expr.mul(1/self.varSac[appliance],densityMean),
                                     Expr.constTerm(
                    self.individual[appliance]['induced density of sac'][0]/
                    self.individual[appliance]['induced density of sac'][1]**2))
                
                diffSac = Expr.sub(Expr.sum(Expr.mul(
                  1.0*np.reshape(np.array(self.individual[appliance]['means']),
                        (1,nosOfStates)),stateVariableOfAppliance[appliance])), 
                  Expr.mul(jointVariance,jointMean)  )
                rqc_sac = M.constraint(Expr.vstack(
                    Expr.constTerm(jointVariance), 
                    tauSac[appliance].asExpr(), diffSac),
                    Domain.inRotatedQCone())
                    
                # the rotated quadratic cone constraint on activity duration
                # declare the epigraph for the activity duration
                tauDuration[appliance] = M.variable(1, Domain.greaterThan(0.0))
                
                # employing the induced density:
                #    the difference between the estimated and expected duration
                jointVarianceDur = 1/(1/self.varDuration[appliance] - 
                    1/(self.individual[appliance]['induced density of duration'][1]**2))
                densityMeanDur = Expr.dot(variableOfNosOfCycles[appliance],
                    np.array(self.individual[appliance]
                    ['numberOfCyclesStats']['numberOfCyclesDuration']))
                jointMeanDur = Expr.sub(Expr.mul(1/self.varDuration[appliance],densityMeanDur),
                                     Expr.constTerm(
                    self.individual[appliance]['induced density of duration'][0]/
                    self.individual[appliance]['induced density of duration'][1]**2))
                    
                diffDuration = Expr.sub(Expr.mul(
                    self.sample_seconds/60.0,
                    Expr.sum(stateVariableOfAppliance[appliance].\
                    slice([1,0],[nosOfStates,nosOfTimePoints]))),
                    Expr.mul(jointVarianceDur,jointMeanDur))
                rqc_duration = M.constraint(Expr.vstack(
                    Expr.constTerm(jointVarianceDur),
                    tauDuration[appliance].asExpr(), diffDuration),
                    Domain.inRotatedQCone())
                    
                # The hard constraints on the number of Off --> On                                
                # difference between the estimated and expected number of OFF->On
                diffOffOn = Expr.sub(Expr.sum(Expr.mulElm(constMatOffOn,
                    relaxedVariableOfAppliance[appliance])),
                    Expr.dot(variableOfNosOfCycles[appliance],
                            np.array(self.individual[appliance]
                            ['numberOfCyclesStats']['numberOfCycles'])))
                c_OffOn = M.constraint(diffOffOn,Domain.equalsTo(0.0))
                
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
                                                             
                # the log categorical distribution for the nos of activity cycles
                if i == 0:
                    sumLogCatProb = Expr.dot(variableOfNosOfCycles[appliance],
                        -np.log(np.maximum(1e-300,
                        np.array(self.individual[appliance]
                        ['numberOfCyclesStats']['numberOfCyclesProb']))))
                else:
                    sumLogCatProb = Expr.add(sumLogCatProb,
                        Expr.dot(variableOfNosOfCycles[appliance],
                        -np.log(np.maximum(1e-300,
                        np.array(self.individual[appliance]
                        ['numberOfCyclesStats']['numberOfCyclesProb'])))))
                
                # Summation of the latent variables for all appliances accross time
                if i == 0:
                    sumLatentVariable = latentVariableOfAppliance[appliance]
                else:
                    sumLatentVariable = Expr.add(sumLatentVariable,
                                        latentVariableOfAppliance[appliance])

                # Forming objective function by adding up the epigraphs #####
                if i == 0:
                    sumTauLatent = tauLatent[appliance]
                    sumTauSac = tauSac[appliance]
                    sumTauDuration = tauDuration[appliance]
                else:
                    sumTauLatent = Expr.add(sumTauLatent,tauLatent[appliance])
                    sumTauSac = Expr.add(sumTauSac,tauSac[appliance])
                    sumTauDuration = Expr.add(sumTauDuration,tauDuration[appliance])
                       
                ##### Counting the number of variables #######################
                nosOfVariables = nosOfVariables \
                                + stateVariableOfAppliance[appliance].size() \
                                + relaxedVariableOfAppliance[appliance].size()\
                                + variableOfNosOfCycles[appliance].size()\
                                + latentVariableOfAppliance[appliance].size() \
                                + tauLatent[appliance].size()\
                                + tauSac[appliance].size()\
                                + tauDuration[appliance].size()
                                
                ##### Counting the number of constraints #####################             
                nosOfConstrs = nosOfConstrs \
                                + c_stateSumToOne.size() \
                                + c_relaxStateSumRow.size() \
                                + c_relaxStateSumColumn.size()\
                                + c_nosCycleSumToOne.size()\
                                + rqc_latent.size()\
                                + rqc_sac.size()\
                                + rqc_duration.size()\
                                + c_OffOn.size()            
            
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
            M.objective('objectiveFunction', 
                        ObjectiveSense.Minimize, 
                        Expr.sum(Expr.vstack([sumInitialProb,
                                              sumTransProb,
                                              delta.asExpr(),
                                              sumTauLatent,
                                              sumTauSac,
                                              sumTauDuration,
                                              sumTauPiecewise,
                                              sumLogCatProb]
                                              )))
            
            # solving the problem
            print("\n Solving the problem ...")
            
            # This defines which solution status values are accepted 
            # when fetching solution values
            M.acceptedSolutionStatus(AccSolutionStatus.Anything)
            M.solve()
            
            ####### Print the optimization status ############################
            print("\n+++++++++++++++optimization status+++++++++++++++++++")
            print("Number of variables:{}".format(nosOfVariables))
            print("Number of constraints:{}".format(nosOfConstrs))
            print("Primal solution status:{}".format(M.getPrimalSolutionStatus()))
            print("Primal value:{}".format(M.primalObjValue()))
            print("Dual solution status:{}".format(M.getDualSolutionStatus()))
            print("Dual value:{}".format(M.dualObjValue()))
            print("Accepted solution status:{}".format(M.acceptedSolutionStatus()))
            print("+++++++++++++++optimization status+++++++++++++++++++")
            ###################################################################
            
            self.primalObjValue = M.primalObjValue()
            self.dualObjValue = M.dualObjValue()

            # recording the computing time
            stop_time = timeit.default_timer()
            print("Solving this problem took '{0}' seconds".format(stop_time-start_time))           

            ###### Reading the inference results ##############################
            inferred_appliance_mains_energy = pd.DataFrame(index=mains_chunk.index)
            inferred_latent_energy = pd.DataFrame(index=mains_chunk.index)
            inferred_sac = OrderedDict()
            inferred_duration = OrderedDict()
            inferred_nosOfCycle = OrderedDict()
            inferred_states = OrderedDict()
            inferred_relaxedStates = OrderedDict()
            inferred_variableOfNosOfCycles = OrderedDict()
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
                    
                inferred_sac[appliance] = \
                    np.dot(np.array(variableOfNosOfCycles[appliance].level()),
                           np.array(self.individual[appliance] 
                           ['numberOfCyclesStats']['numberOfCyclesEnergy']))
                           
                inferred_duration[appliance] = \
                    np.dot(np.array(variableOfNosOfCycles[appliance].level()),
                           np.array(self.individual[appliance]
                           ['numberOfCyclesStats']['numberOfCyclesDuration']))
                           
                inferred_nosOfCycle[appliance] = \
                    np.dot(np.array(variableOfNosOfCycles[appliance].level()),
                           np.array(self.individual[appliance]
                           ['numberOfCyclesStats']['numberOfCycles']))
                
                # the inferred variables
                inferred_states[appliance] = \
                    np.reshape(np.array(stateVariableOfAppliance[appliance].level()),
                               (nosOfStates,nosOfTimePoints))
                    
                inferred_relaxedStates[appliance] = \
                    np.reshape(np.array(relaxedVariableOfAppliance[appliance].level()),
                               constMatOffOn.shape)
                     
                inferred_variableOfNosOfCycles[appliance] = \
                    np.array(variableOfNosOfCycles[appliance].level())
                        
            inferred_appliance_mains_energy['inferred mains'] = \
                inferred_appliance_mains_energy.iloc[:,0:len(self.individual)].\
                sum(axis=1).values
            inferred_appliance_mains_energy['mains'] = mains_chunk.values
                            
            inferred_latent_energy['inferred mains'] = \
                inferred_latent_energy.iloc[:,0:len(self.individual)].sum(axis=1).values
            inferred_latent_energy['mains'] = mains_chunk.values
            inferred_latent_energy['piecewise noise'] = \
                variableOfPiecewise.level()                
            
            prediction['time'] = stop_time-start_time
            prediction['inferred appliance energy'] = inferred_appliance_mains_energy
            prediction['inferred latent energy'] = inferred_latent_energy
            prediction['inferred total energy'] = inferred_sac
            prediction['inferred activity duration'] = inferred_duration
            prediction['inferred number of cycles'] = inferred_nosOfCycle
            prediction['inferred states'] = inferred_states
            prediction['inferred relaxed states'] = inferred_relaxedStates
            prediction['inferred variable for nos of cycles'] = \
                                                inferred_variableOfNosOfCycles
            return prediction
        
    def estimate_noisevariance(self, mains_chunk, prediction):
        # estimate the noise variances
        nosOfTimePoints = int(len(mains_chunk))
        inferred_appliance_mains_energy = prediction['inferred appliance energy']
        inferred_sac = prediction['inferred total energy']
        inferred_duration = prediction['inferred activity duration']
        inferred_states = prediction['inferred states']
        inferred_nosOfCycle = prediction['inferred number of cycles']
        inferred_relaxedStates = prediction['inferred relaxed states']
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
            self.varSac[appliance] = min((self.beta+
                0.5*((sum(inferred_appliance_mains_energy[appliance])
                -inferred_sac[appliance])**2))/(0.5+self.alpha-1),
                self.individual[appliance]['induced density of sac'][1]**2-1.0)
            self.varDuration[appliance] = min((self.beta+
                0.5*(((self.sample_seconds/60.0)*
                np.sum(inferred_states[appliance][1:,:])
                -inferred_duration[appliance])**2))/(0.5+self.alpha-1),
                self.individual[appliance]['induced density of duration'][1]**2-1.0)
                
        # Estimate the model noise variance
        self.varModel = (self.beta + 
           0.5*np.sum((mains_chunk.values.ravel().reshape(1,nosOfTimePoints)
           - inferred_latent_energy['inferred mains'].values 
           - inferred_latent_energy['piecewise noise'].values )**2))\
           /(0.5*nosOfTimePoints+self.alpha-1)
        
        # Estimate the noise variance for piecewise prior
        self.varPieceWiseNoise = (self.beta +
            0.5*np.sum(np.abs(inferred_latent_energy['piecewise noise'].\
            iloc[1:nosOfTimePoints].values
            -inferred_latent_energy['piecewise noise'].\
            iloc[0:nosOfTimePoints-1].values)))\
            /(nosOfTimePoints-1+self.alpha-1)

    def objective(self, mains_chunk, prediction):
        # this method computes the log-posterior distribution
        # estimate the noise variances
        nosOfTimePoints = int(len(mains_chunk))
        inferred_appliance_mains_energy = prediction['inferred appliance energy']
        inferred_sac = prediction['inferred total energy']
        inferred_duration = prediction['inferred activity duration']
        inferred_states = prediction['inferred states']
        inferred_nosOfCycle = prediction['inferred number of cycles']
        inferred_relaxedStates = prediction['inferred relaxed states']
        inferred_latent_energy = prediction['inferred latent energy']
        inferred_variableOfNosOfCycles = prediction['inferred variable for nos of cycles']
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

            ############# signal aggregate objective ##################
            optimalObjective = optimalObjective\
                -0.5*np.log(self.varSac[appliance])\
                -0.5*(1.0/self.varSac[appliance])*\
                        ((sum(inferred_appliance_mains_energy[appliance])
                -inferred_sac[appliance])**2)\
                -(self.alpha-1.0)*np.log(self.varSac[appliance])\
                -self.beta*(1.0/self.varSac[appliance])              
                
            ############# duration objective #######################
            optimalObjective = optimalObjective\
                -0.5*np.log(self.varDuration[appliance])\
                -0.5*(1.0/self.varDuration[appliance])*\
                                    (((self.sample_seconds/60.0)*
                                    np.sum(inferred_states[appliance][1:,:])
                -inferred_duration[appliance])**2)\
                -(self.alpha-1.0)*np.log(self.varDuration[appliance])\
                -self.beta*(1.0/self.varDuration[appliance])
                
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
                                
            ########## nos of cycles prior ###############################
            optimalObjective = optimalObjective\
                + np.dot(inferred_variableOfNosOfCycles[appliance],
                        np.log(np.maximum(1e-300,
                        np.array(self.individual[appliance]
                        ['numberOfCyclesStats']['numberOfCyclesProb']))))
            
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
                 
        print("\n log(datalikelihood*prior)={}\n".format(optimalObjective))
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
        
        """
        Train the model using the HES data. Ideally, we read the HES data, and 
        train the model here. Instead we train all the appliances beforehand
        and thus we read the learnt model parameters from a file
        """
        # input parameter: 
        # meterlist is a list of appliances for disaggregation
        # filename = 'appliance_model_induced_density.json'
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
        inferred_sac = prediction['inferred total energy']
        inferred_duration = prediction['inferred activity duration']
        inferred_states = prediction['inferred states']
        inferred_nosOfCycle = prediction['inferred number of cycles']
        inferred_relaxedStates = prediction['inferred relaxed states']
        inferred_latent_energy = prediction['inferred latent energy']
        objLatent = {}
        objSac = {}
        objDuration = {}
        objNosOfCycle = {}
        for (i, appliance) in enumerate(self.individual):
            nosOfStates = int(self.individual[appliance]['numberOfStates'])
            if i == 0:
                sumAppliance = inferred_appliance_mains_energy[appliance]
            else:
                sumAppliance = sumAppliance + inferred_appliance_mains_energy[appliance]
                
            objLatent[appliance] = np.sum((inferred_latent_energy[appliance]
                                -inferred_appliance_mains_energy[appliance])**2)
            objSac[appliance] = (sum(inferred_appliance_mains_energy[appliance])
                                -inferred_sac[appliance])**2
            objDuration[appliance] = ((self.sample_seconds/60.0)*
                                np.sum(inferred_states[appliance][1:,:])
                                    -inferred_duration[appliance])**2                
            ######### necessary constant matrix for the model #############                           
            tempVector = np.zeros((nosOfStates,nosOfStates))
            tempVector[:,0][1:]=1.0
            constMatOffOn = np.kron(np.ones((1,nosOfTimePoints-1)),
                                    tempVector)
            objNosOfCycle[appliance] = (np.sum(np.multiply(constMatOffOn,inferred_relaxedStates[appliance]))
                -inferred_nosOfCycle[appliance])**2
            print("\n {0} latent error:{1}".format(appliance,objLatent[appliance]))
            print("\n {0} Sac error:{1}".format(appliance,objSac[appliance]))
            print("\n {0} Duration error:{1}".format(appliance,objDuration[appliance]))
            print("\n {0} OffOn error:{1}".format(appliance,objNosOfCycle[appliance]))
            
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
        constraints['obj sac'] = objSac
        constraints['obj duration'] = objDuration
        constraints['obj nos of cycle'] = objNosOfCycle
        
        return constraints
