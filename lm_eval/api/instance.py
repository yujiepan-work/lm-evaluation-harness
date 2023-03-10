import abc

class Instance(abc.ABC):
    """
    A class used to bind together all necessary information and metadata for 
    running forward pass of a model on a specific datapoint. 

    """

    # all Instance subclasses have an attribute which is the name of the LM() class function they call to get outputs.
    request_type = None

    def __init__(self, doc, arguments=None, id_=None, metadata=("", None, None)):

        self.doc = doc # store the document which we're using. this is a dict
        self.task_name = None
        self._arguments = arguments

        # need: task name, doc idx, num. repeats
        self.task_name, self.doc_idx, self.repeats = metadata
        # id_ = idx within a doc's requests
        self.id_ = id_

        # handle repeats internally. should be able to run K times on exact same input/output pair
        # self.repeats = repeats
        
        # lists containing: 1) the inputs and targets for each, 2) the outputs from the model, 3) the outputs from the model after applying filtering
        self.resps = None
        self.filtered_resps = None

        #TODO: add more info as needed for detailed logging

class LoglikelihoodInstance(Instance):

    request_type = "loglikelihood"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def args(self):
        """
        Returns (context,target) where `context` is the input and `target` is 
        the string to calculate loglikelihood over, conditional on `context` preceding it.
        """
        return self._arguments


class RollingLoglikelihoodInstance(Instance):

    request_type = "loglikelihood_rolling"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
    
    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return self._arguments if isinstance(self._arguments, tuple) else (self.arguments,)

class GenerationInstance(Instance):

    request_type = "greedy_until"

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        #TODO: generation/model fwd pass kwargs here and should be passed through arguments as well

    @property
    def args(self):
        """
        Returns (string, until) where `string` is the input sequence beginning generation and 
        `until` is a string or list of strings corresponding to stop sequences.
        """
        return self._arguments