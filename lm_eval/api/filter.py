


class Filter:
    """
    Filter classes operate on a per-task level. 
    They take all model outputs (`instance.resps` for all `task.instances`)
    across all instances of a task, and perform operations.
    In a single run, one can configure any number of separate filters or lists of filters.

    """

    name = "unfiltered"

    def __init__(self):
        """
        Can define custom behavior here. 
        """

    def apply(self, resps: list):
        """This base class Filter performs a no-operation
        pass through the model responses, and returns an "unfiltered"
        version. 
        Here, the first model output is taken, in the case where repeats > 1.
        """
        # TODO: does apply() run at the Instance level or task level?
        return {"unfiltered": resps[0]}