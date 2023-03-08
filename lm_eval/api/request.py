from typing import List


class Instance:
    def __init__(self, inps: List[str], doc, fewshot_context="", doc_idx=None, repeats: int=1):

        self.doc = doc # store the document which we're using. this is a dict
        self.doc_idx = doc_idx # index of the doc within valid/test set
        self.fewshot_context=fewshot_context

        self.repeats = repeats
        
        # lists containing: 1) the inputs and targets for each, 2) the outputs from the model, 3) the outputs from the model after applying filtering
        self.inps = inps
        self.resps = None
        self.filtered_resps = None

        #TODO: add more info as needed for detailed logging
        

    # def __iter__(self):
    #     if REQUEST_RETURN_LENGTHS[self.request_type] is None:
    #         raise IndexError("This request type does not return multiple arguments!")
    #     for i in range(REQUEST_RETURN_LENGTHS[self.request_type]):
    #         yield Request(self.request_type, self.args, i)

    # def __getitem__(self, i):
    #     if REQUEST_RETURN_LENGTHS[self.request_type] is None:
    #         raise IndexError("This request type does not return multiple arguments!")
    #     return Request(self.request_type, self.args, i)

    # def __eq__(self, other):
    #     return (
    #         self.request_type == other.request_type
    #         and self.args == other.args
    #         and self.index == other.index
    #     )

    # def __repr__(self):
    #     return f"Req_{self.request_type}{self.args}[{self.index}]\n"


class LoglikelihoodInstance(Instance):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.output_type = "loglikelihood"


class RollingLoglikelihoodInstance(Instance):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.output_type = "loglikelihood_rolling"
    

class GenerationInstance(Instance):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # TODO: add a check that self.repeats not greater than 1
        
        self.output_type = "greedy_until"

        #TODO: add other generation/model fwd pass kwargs here to instances



