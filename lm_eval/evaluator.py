import collections
import itertools
import numpy as np
import random
import lm_eval.api.metrics
import lm_eval.models
import lm_eval.tasks

from lm_eval.utils import run_task_tests

def cli_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    check_integrity=False,
    # decontamination_ngrams_path=None,
):

    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    # run_task_tests(task_list=tasks)

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args, {"batch_size": batch_size, "device": device}
        )
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    # if isinstance(tasks, str):
    #     task_dict = lm_eval.tasks.get_task_dict(tasks)

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    # if check_integrity:
    #     run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        # decontamination_ngrams_path=decontamination_ngrams_path,
    )

    # add info about the model and few shot config
    results["config"] = {
        "model": model,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
    }

    return results


decontaminate_suffix = "_decontaminate"


def evaluate(
    lm,
    task_dict,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    decontamination_ngrams_path=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :return
        Dictionary of results
    """

    #### PREPARE DATA FOR MODEL (maybe: asynchronous, if so then do it before model loading) #####

    #### Download dataset from HF

    #### apply transforms to dataset
    print(task_dict)
    for task_name, task in task_dict.items():
        if task.has_test_docs():
            task_doc_func = task.test_docs
        elif task.has_validation_docs():
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")
        task.build_requests(task_doc_func()) # should run construct_requests() for all docs in desired set. 
        # stick datapoints in a list by type, sort by ascending len. datapoints should "remember" their origins and store the result

    #### send the data through the model ####

    for task_name, task in task_dict.items(): 
        getattr(lm, task.requests[0].output_type)(task.requests)

    
    #### Take responses and apply filtering/"solution selection" TODO: implement this
    # for task_name, task in task_dict:
    #     task.apply_filters()
        

    #### Calculate metrics ####
    vals = collections.defaultdict(list)
    for task_name, task in task_dict.items():
        for req in task.requests:
            metrics = task.process_results(req.doc, req.resps) # TODO: this doesn't work for multiple-choice questions yet
            for metric, value in metrics.items():
                vals[(task_name, metric)].append(value)

    # aggregate results
    results = collections.defaultdict(dict)
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        results[task_name][metric] = task.aggregation()[metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.api.metrics.stderr_for_metric(
            metric=task.aggregation()[metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )

        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)


    return {"results": dict(results), "versions": {task_name: task.VERSION for task_name, task in task_dict.items()}}


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()
