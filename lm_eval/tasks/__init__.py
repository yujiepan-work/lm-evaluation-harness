from pprint import pprint
from typing import List, Union

from lm_eval.api.task import Task, ConfigurableTask

from . import lambada



########################################
# All tasks
########################################


TASK_REGISTRY = {
    "lambada_openai": lambada.LambadaOpenAI,
    "lambada_standard": lambada.LambadaStandard,
}


ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def get_task_name_from_config(task_config):
    return "configurable_{dataset_path}_{dataset_name}".format(**task_config)


def get_task_dict(task_name_list: List[Union[str, dict, Task]]):
    task_name_dict = {
        task_name: get_task(task_name)(config={"num_fewshot": 0}) # TODO: don't hardcode this and figure out a proper config system
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_config_dict = {
        get_task_name_from_config(task_config): ConfigurableTask(config=task_config)
        for task_config in task_name_list
        if isinstance(task_config, dict)
    }
    task_name_from_object_dict = {
        get_task_name_from_object(task_object): task_object
        for task_object in task_name_list
        if isinstance(task_object, Task)
    }
    assert set(task_name_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {
        **task_name_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict
    }
