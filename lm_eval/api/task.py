import abc
from dataclasses import dataclass

import datasets

import re
import random

from lm_eval.api.metrics import mean, weighted_perplexity, weighted_mean, bits_per_byte
from lm_eval.api.request import LoglikelihoodInstance, RollingLoglikelihoodInstance

from lm_eval import utils

@dataclass
class TaskConfig:
    dataset_path: str = None
    dataset_name: str = None
    should_decontaminate: bool = False
    has_training_docs: bool = None
    has_validation_docs: bool = None
    has_test_docs: bool = None
    training_split: str = None
    validation_split: str = None
    test_split: str = None
    aggregation: dict = None
    higher_is_better: dict = None

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        return setattr(self, key, value)

class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

        A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config:dict=None):
        """
        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

        self._config = config

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        """Downloads and returns the task dataset.
        Override this method to download the dataset from a custom API.

                :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    def should_decontaminate(self):
        """Whether this task supports decontamination against model training set."""
        return False

    @abc.abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass

    @abc.abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abc.abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    def training_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    @property
    def requests(self):
        return self._instances

    @property
    def request_type(self):
        """Should return the subclass of Instance that is used by task"""
        return self._req_type

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc):
        print(
            "Override doc_to_decontamination_query with document specific decontamination query."
        )
        assert False

    @abc.abstractmethod
    def doc_to_text(self, doc):
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc):
        pass

    def build_requests(self, docs):
        """Build a set of Requests for a task, and store them in task.instances.


        :param docs:
            The set of documents as returned from training_docs, validation_docs, or test_docs.
        """

        instances = []
        for idx, doc in enumerate(docs):
            # sample fewshot context (uses prompt defined in self.doc_to_text())
            fewshot_ctx = self.fewshot_context(doc, self._config["num_fewshot"], rnd=random.Random())

            # TODO: hardcoded for now: # of runs on each input to be 1. advanced users should have ability to run model multiple times on same input
            inst = self.construct_requests(doc=doc, ctx=fewshot_ctx, doc_idx=idx, repeats=1)

            # TODO: this means that e.g. the multiple calls for a given doc for multiple choice get added to this list as separate Instances 
            # (albeit with shared task_index *AND* req_id)
            if isinstance(inst, list):
                instances.extend(inst)
            else: 
                instances.append(inst)

        self._instances = instances
        assert len(self._instances) != 0, "task.build_requests() did not find any docs!"
 
    @abc.abstractmethod
    def construct_requests(self, doc, ctx):
        """
        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        pass

    @abc.abstractmethod
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pass

    @abc.abstractmethod
    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        pass

    @abc.abstractmethod
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        pass

    @utils.positional_deprecated
    def fewshot_context(
        self, doc, num_fewshot, rnd=None, description=None
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)
        return labeled_examples + example


class ConfigurableTask(Task):

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config:dict=None
    ):

        self._config = TaskConfig(**config)
        if self._config.dataset_path is not None:
            self.DATASET_PATH = self._config.dataset_path

        if self._config.dataset_name is not None:
            self.DATASET_NAME = self._config.dataset_name

        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def has_training_docs(self):
        return self._config.has_training_docs

    def has_validation_docs(self):
        return self._config.has_validation_docs

    def has_test_docs(self):
        return self._config.has_test_docs

    def training_docs(self):
        if self._config.training_split is not None:
            return self.dataset[self._config.training_split]

    def validation_docs(self):
        if self._config.validation_split is not None:
            return self.dataset[self._config.validation_split]

    def test_docs(self):
        if self._config.test_split is not None:
            return self.dataset[self._config.test_split]

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    def doc_to_text(self, doc):
        _doc_to_text_type = type(self._config.doc_to_text)
        if type(_doc_to_text_type) is str:
            return self._config.doc_to_text.format(**doc)
        elif type(_doc_to_text_type):
            return self._config.doc_to_text(doc)

    def doc_to_target(self, doc):
        _doc_to_target_type = type(self._config.doc_to_target)
        if type(_doc_to_target_type) is str:
            return self._config.doc_to_target.format(**doc)
        elif type(_doc_to_target_type):
            return self._config.doc_to_target(doc)

    def construct_requests(self, doc, ctx, **kwargs):
        return LoglikelihoodInstance(
            [ctx + self.doc_to_text(doc), self.doc_to_target(doc)],
            doc,
            fewshot_context=ctx,
            **kwargs
        )

    def process_results(self, doc, results):
        
        result_dict = {}
        for key, result in zip(self._config.aggregation, results):
            result_dict[key] = result
        
        return result_dict

    def aggregation(self):
        return self._config.aggregation

    def higher_is_better(self):
        return self._config.higher_is_better


class MultipleChoiceTask(Task):
    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx, **kwargs):
        # lls = [
        #     rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        # ]
        reqs = []
        for inp in [ctx + self.doc_to_text(ctx) + " {}".format(choice) for choice in doc["choices"]]:
             
            reqs.append(LoglikelihoodInstance(doc, fewshot_context, **kwargs))
        
        return reqs

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
        }


class PerplexityTask(Task, abc.ABC):
    def should_decontaminate(self):
        """Whether this task supports decontamination against model training set."""
        return True

    def has_training_docs(self):
        return False

    def fewshot_examples(self, k, rnd):
        assert k == 0
        return []

    def fewshot_context(
        self, doc, num_fewshot, rnd=None,
    ):
        assert (
            num_fewshot == 0
        ), "The number of fewshot examples must be 0 for perplexity tasks."
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`."

        return ""

    def higher_is_better(self):
        return {
            "word_perplexity": False,
            "byte_perplexity": False,
            "bits_per_byte": False,
        }

    def doc_to_decontamination_query(self, doc):
        return doc

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return doc

    def construct_requests(self, doc, ctx, doc_idx=None, repeats=1):
        assert not ctx
        
        return RollingLoglikelihoodInstance([self.doc_to_target(doc)], doc, fewshot_context=ctx, doc_idx=doc_idx, repeats=1)

    def process_results(self, doc, results):
        (loglikelihood,) = results
        words = self.count_words(doc)
        bytes_ = self.count_bytes(doc)
        return {
            "word_perplexity": (loglikelihood, words),
            "byte_perplexity": (loglikelihood, bytes_),
            "bits_per_byte": (loglikelihood, bytes_),
        }

    def aggregation(self):
        return {
            "word_perplexity": weighted_perplexity,
            "byte_perplexity": weighted_perplexity,
            "bits_per_byte": bits_per_byte,
        }

    @classmethod
    def count_bytes(cls, doc):
        return len(doc.encode("utf-8"))

    @classmethod
    def count_words(cls, doc):
        """Downstream tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))
