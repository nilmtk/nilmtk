from __future__ import print_function, division
from copy import deepcopy

class Pipeline(object):
    """
    A data processing pipeline for processing power data.  Operates at
    the "Meter" layer.  The basic motivation is that we want to be 
    able to do a sequence of processing steps on a chunk
    while that chunk is in memory.
    
    DATASTORE -> NODE_1 -> ... -> NODE_N
    
    A pipeline consists of one datastore
    node which loads and, if necessary, splits the data into chunks;
    if there are K chunks then the pipeline runs K times; and on each
    iteration the output from the datastore is a single DataFrame
    (with metatdata such as sample_period, max_sample_period, 
    chunk_start_datetime, chunk_end_datetime, etc).
       
    Downstream of the store are an arbitrary number of "nodes"
    which process data in sequence or export the data to disk.
        
    During a single cycle of the pipeline, results from each
    stats node are stored in the `dataframe.results` dict.  At the end
    of each pipeline cycle, the contents of dataframe.results 
    are combined and the aggregate results are stored in `pipeline.results`
    
    Each processing node has a set of requirements (e.g. gaps must be
    filled) and a set of postconditions (e.g. gaps will have been
    filled).  This allows us to check that a particular pipeline is
    viable (i.e. that, for every node, the node's preconditions are
    satisfied by an upstream node or by the source).

    IDEAS FOR THE FUTURE???:
    Pipelines could be saved/loaded from disk.
    
    If the pipeline was represented by a directed acyclic
    graphical model (DAG) then:
      pipeline could fork into multiple parallel
      pipelines.  Data and metadata would be copied to each fork and
      each sub-pipeline would be run as a separate process (after
      checking requirements for each subpipeline as the start).
    
      Pipelines could be rendered
      graphically.  In the future it would be nice to have a full
      graphical UI (like Node-RED).
    
    Attributes
    ----------
    nodes : list of Node objects
    results : dict of Results objects storing aggregate stats results
    
    Examples
    --------
    
    >>> store = HDFDataStore('redd.h5')
    >>> meter = ElecMeter(store, {'data_location': '/building1/elec/meter1'})

    Calculate total energy and save the preprocessed data
    and the energy data back to disk:
    
    >>> nodes = [LocateGoodSections(), 
                 Energy(), 
                 HDFTableExport('meter1_preprocessed.h5', table_path)]
    >>> pipeline = Pipeline(nodes)
    >>> pipeline.run(meter)
    >>> energy = pipeline.results['energy'].combined
    >>> print("Active energy =", energy['active'], "kWh",
    >>>       "and reactive =", energy['reactive'], "kWh")
    
    """
    def __init__(self, nodes=None):
        self.nodes = [] if nodes is None else nodes
            
    def run(self, meter, **load_kwargs):
        """
        Parameters
        ----------
        meter : nilmtk.ElecMeter
        **load_kwargs : key word arguments for store.load() e.g.:
            - periods : list of nilmtk.TimeFrame objects
        """
        self._reset()
        meter_metadata = deepcopy(meter.metadata)
        meter_metadata.update({'device': meter.device})
        self._check_requirements(meter_metadata)

        # Run pipeline
        # TODO only load required measurements
        for chunk in meter.store.load(key=meter.key, **load_kwargs):
            processed_chunk = self._run_chunk_through_pipeline(chunk, meter_metadata)
            self._update_results(processed_chunk.results)

    def _reset(self):
        self.results = {}
        for node in self.nodes:
            node.reset()

    def _check_requirements(self, metadata):
        # Not yet fully implemented?
        state = deepcopy(metadata)
        required_measurements = set() 
        for node in self.nodes:
            node.check_requirements(state)
            required_measurements.update(node.required_measurements(state))
            state = node.update_state(state)

    def _run_chunk_through_pipeline(self, chunk, metadata):
        """
        Returns
        -------
        Processes DataFrame of data, including a `.results` dict of `Results`
        objects for this chunk.
        """
        for node in self.nodes:
            chunk = node.process(chunk, metadata)
        return chunk
    
    def _update_results(self, results_for_chunk):
        for statistic, result in results_for_chunk.iteritems():
            try:
                self.results[statistic].update(result)
            except KeyError:
                self.results[statistic] = result
