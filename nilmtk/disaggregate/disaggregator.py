
# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint

class Disaggregator(object):
    """
    Provides a common interface to all disaggregation classes.
    See https://github.com/nilmtk/nilmtk/issues/271 for discussion.

    Attributes
    ----------
    Each subclass should internally store models learned from training.
    """
    def __init__(self):
        """
        Parameters
        ----------
        """

    def train(self, metergroup):
        """
        Trains the model given a metergroup containing a appliance meters (supervised) or a site meter (unsupervised).
        Will have a default implementation in super class.
        Can be overridden for simpler in-memory training, or more complex out-of-core training.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        """

        raise NotImplementedError("NotImplementedError")
        
    def train_on_chunk(self, chunk, identifier):
        """
        Signature is fine for site meter dataframes (unsupervised learning). Would need to be called for each appliance meter along with appliance identifier for supervised learning.
        Required to be overridden to provide out-of-core disaggregation.

        Parameters
        ----------
        chunk : pd.DataFrame where each column represents a disaggregated appliance
        identifier : tuple of (nilmtk.appliance, int) representing instance of that appliance for this chunk
        """

        raise NotImplementedError("NotImplementedError")
        
    def disaggregate(self, mains, output_datastore):
        """
        Passes each chunk from mains generator to disaggregate_chunk() and passes the output to _write_disaggregated_chunk_to_datastore()
        Will have a default implementation in super class.
        Can be overridden for more simple in-memory disaggregation, or more complex out-of-core disaggregation.

        Parameters
        ----------
        mains : nilmtk.ElecMeter (single-phase) or nilmtk.MeterGroup (multi-phase)
        output_datastore : instance of nilmtk.DataStore or str of datastore location
        """
        
        raise NotImplementedError("NotImplementedError")
    
    def disaggregate_chunk(self, chunk):
        """
        Loads all of a DataFrame from disk.

        Parameters
        ----------
        chunk : pd.DataFrame (in NILMTK format)
        
        Returns
        -------
        chunk : pd.DataFrame where each column represents a disaggregated appliance
        """

        raise NotImplementedError("NotImplementedError")
    
    def _write_disaggregated_chunk_to_datastore(self, chunk, datastore):
        """
        Writes disaggregated chunk to NILMTK datastore.
        Should not need to be overridden by sub-classes.

        Parameters
        ----------
        chunk : pd.DataFrame representing a single appliance (chunk needs to include metadata)
        datastore : nilmtk.DataStore
        
        """

        raise NotImplementedError("NotImplementedError")
    
    def import_model(self, filename):
        """
        Loads learned model from file.
        Required to be overridden for learned models to persist.

        Parameters
        ----------
        filename : str path to file to load model from
        
        """

        raise NotImplementedError("NotImplementedError")
        
    def export_model(self, filename):
        """
        Saves learned model to file.
        Required to be overridden for learned models to persist.

        Parameters
        ----------
        filename : str path to file to save model to
        
        """

        raise NotImplementedError("NotImplementedError")
