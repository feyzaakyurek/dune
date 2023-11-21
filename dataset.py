from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from prompt import Prompt
from config import GPTConfig
from gptcache import GPTCache
from tqdm import tqdm
from util import (
    bbq_equivalence_test,
    bbnli_equivalence_test,
    realtox_pres_test,
    find_last_consecutive_digits,
    new_info_scrape_answer,
    arc_scrape_answer,
    PROJECTP,
)
import numpy as np
import pandas as pd
from itertools import chain
import ipdb
import json
import os


class EditDataset(ABC):
    @abstractmethod
    def read_data(self, data_file: str):
        pass

    @abstractmethod
    def sample_edit(config: GPTConfig, gpt: GPTCache):
        pass

    @abstractmethod
    def _post_process_edits():
        pass

    @abstractmethod
    def sample_test_inputs(config: GPTConfig, gpt: GPTCache, ti_num: int):
        pass

    @abstractmethod
    def _post_process_test_inputs():
        pass

    @abstractmethod
    def test_scores(self, preds: List):
        pass

    @abstractmethod
    def save(path: str):
        pass

    def load_test_inputs(self, test_input_path: str, **kwargs):
        with open(test_input_path, "r") as f:
            data = json.load(f)
        self.edits = data["edits"]
        self.test_inputs = data["test_inputs"]

    def get_test_inputs_only(self):
        if self.test_inputs[0].isinstance(list):
            return [t[0] for t in self.test_inputs]
        else:
            return self.test_inputs


class Arithmetic(EditDataset):
    def __init__(
        self,
    ):
        data_file = (
            f"{PROJECTP}/source/arithmetic/arithmetic_validated_chainofreasoning.csv"
        )
        self.edit_prompt = None
        self.test_input_prompt = None
        self.read_data(data_file)

        # Will be used when sampling answers to test inputs.
        self.form_with_edit = (
            "Solve the following problem."
            " Do not show any work."
            " Provide only a number after Answer:"
            " You know that "
            "{edit}"
            ". "
            "{question}"
        )
        self.form_without_edit = (
            "Solve the following problem."
            " Do not show any work."
            " Provide only a number after Answer: "
            "{question}"
        )

    def read_data(self, data_file):
        self.data = pd.read_csv(data_file)

    def sample_edit(self, config: GPTConfig, gpt: GPTCache):
        self.edits = self.data["edit"].tolist()

    def _post_process_edits():
        pass

    def sample_test_inputs(self, config: GPTConfig, gpt: GPTCache, ti_num: int = 1):
        cols = [col for col in self.data.columns if col.startswith("test_input")]
        self.test_inputs = [
            [row[col] for col in cols] for _, row in self.data.iterrows()
        ]

    def _post_process_test_inputs(self):
        pass

    def test_scores(self, preds: List, **kwargs):
        # Eval edit and check equality.
        assert type(preds[0]) == str
        mode = kwargs.get("mode", "equality")
        mean = kwargs.get("mean", False)

        gold = [float(eval(e.split(" = ")[0])) for e in self.edits]
        preds = [float(find_last_consecutive_digits(p.replace(",", ""))) for p in preds]

        if mode == "equality":
            scores = []
            for pred, g in zip(preds, gold):
                scores.append(pred == g)
        else:
            raise NotImplementedError("Only equality mode is implemented for now.")
        return np.mean(scores) if mean else scores

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/arithmetic.json", "w") as f:
            json.dump(
                {
                    "edits": self.edits,
                    "test_inputs": self.test_inputs,
                },
                f,
            )


class ARC(EditDataset):
    def __init__(
        self,
    ):
        data_file = f"{PROJECTP}/source/arc/arc_processed.csv"
        self.edit_prompt = None
        self.test_input_prompt = None
        self.read_data(data_file)

        # Will be used when sampling answers to test inputs.
        self.form_with_edit = "{edit} {question}"
        self.form_without_edit = "{question}"

    def read_data(self, data_file):
        self.data = pd.read_csv(data_file)

    def sample_edit(self, config: GPTConfig, gpt: GPTCache):
        pass

    def _post_process_edits():
        pass

    def sample_test_inputs(self, config: GPTConfig, gpt: GPTCache, ti_num: int = 1):
        pass

    def _post_process_test_inputs(self):
        pass

    def test_scores(self, preds: List, **kwargs):
        # Eval edit and check equality.
        assert type(preds[0]) == str
        mode = kwargs.get("mode", "equality")
        mean = kwargs.get("mean", False)

        gold = [t[1] for t in self.test_inputs]
        preds = [arc_scrape_answer(p) for p in preds]

        if mode == "equality":
            scores = []
            for pred, g in zip(preds, gold):
                scores.append(pred == g)
        else:
            raise NotImplementedError("Only equality mode is implemented for now.")
        return np.mean(scores) if mean else scores

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/arc.json", "w") as f:
            json.dump(
                {
                    "edits": self.edits,
                    "test_inputs": self.test_inputs,
                },
                f,
            )


class NewInfo(EditDataset):
    def __init__(
        self,
    ):
        data_file = f"{PROJECTP}/source/newinfo/new_info.csv"
        self.edit_prompt = None
        self.test_input_prompt = None
        self.read_data(data_file)

        # Will be used when sampling answers to test inputs.
        self.form_with_edit = (
            "Answer the following problem, based on this information: "
            "{edit}"
            " Provide only a letter after Answer: "
            "{question}"
        )
        self.form_without_edit = (
            "Answer the following problem. Provide only a letter after Answer: "
            "{question}"
        )

    def read_data(self, data_file):
        self.data = pd.read_csv(data_file)

    def sample_edit(self, config: GPTConfig, gpt: GPTCache):
        pass

    def _post_process_edits():
        pass

    def sample_test_inputs(self, config: GPTConfig, gpt: GPTCache, ti_num: int = 1):
        pass

    def _post_process_test_inputs(self):
        pass

    def load_test_inputs(self, test_input_path: str, flattened=False):
        with open(test_input_path, "r") as f:
            data = json.load(f)
        if flattened:
            edits, test_inputs = [], []
            for edit, tis in zip(data["edits"], data["test_inputs"]):
                for ti in tis:
                    edits.append(edit)
                    test_inputs.append(ti)
        else:
            edits = data["edits"]
            test_inputs = data["test_inputs"]
        self.edits = edits
        self.test_inputs = [[t[0].strip(), t[1].strip()] for t in test_inputs]

    def test_scores(self, preds: List, **kwargs):
        # Eval edit and check equality.
        assert preds[0].isinstance(str)
        mode = kwargs.get("mode", "equality")
        mean = kwargs.get("mean", False)

        gold = [t[1] for t in self.test_inputs]
        preds = [new_info_scrape_answer(p) for p in preds]

        if mode == "equality":
            scores = []
            for pred, g in zip(preds, gold):
                scores.append(pred == g)
        else:
            raise NotImplementedError("Only equality mode is implemented for now.")
        return np.mean(scores) if mean else scores

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/new-info.json", "w") as f:
            json.dump(
                {
                    "edits": self.edits,
                    "test_inputs": self.test_inputs,
                },
                f,
            )


class BBQ(EditDataset):
    def __init__(
        self,
    ):
        self.data_file_name = "bbq_questions_answers"
        data_file = f"{PROJECTP}/source/bbq/{self.data_file_name}.csv"
        self.edit_prompt = Prompt(f"{PROJECTP}/prompts/bbq/sample_edit.txt")
        self.test_input_prompt = Prompt(
            f"{PROJECTP}/prompts/bbq/sample_test_inputs.txt"
        )
        self.read_data(data_file)

        # Will be used when sampling answers to test inputs.
        # self.form_with_edit = "{question} Note that {edit}"
        self.form_with_edit = "{edit} {question}"

    def read_data(self, data_file: str):
        self.edit_queries = pd.read_csv(data_file)
        self.answers = ans = self.edit_queries["answer"].tolist()
        self.questions = qs = self.edit_queries["question"].tolist()
        self.edit_queries = [
            self.edit_prompt.format(answer=a, question=q) for a, q in zip(ans, qs)
        ]

    def sample_edit(self, config: GPTConfig, gpt: GPTCache):
        self.edits = []
        for d in self.edit_queries:
            self.edits.append(gpt.generate(d, max_tokens=config.max_tokens))
        self._post_process_edits()

    def _post_process_edits(self):
        self.edits = [self.edit_prompt.out_func(e) for e in self.edits]

    def sample_test_inputs(self, config: GPTConfig, gpt: GPTCache, ti_num: int = 1):
        self.ti_num = ti_num
        self.test_inp_queries = [
            self.test_input_prompt.format(guideline=g, question=q, answer=a)
            for g, q, a in zip(self.edits, self.questions, self.answers)
        ]
        self.test_inputs = []
        for d in self.test_inp_queries:
            self.test_inputs.append(
                [
                    gpt.generate(d, max_tokens=config.max_tokens, index=i)
                    for i in range(ti_num)
                ]
            )
        self._post_process_test_inputs()

    def _post_process_test_inputs(self):
        self.test_inputs = [
            [self.test_input_prompt.out_func(c) for c in t] for t in self.test_inputs
        ]

    def load_test_inputs(self, test_input_path: str, flattened=True):
        with open(test_input_path, "r") as f:
            data = json.load(f)
        if flattened:
            edits, test_inputs = [], []
            for edit, tis in zip(data["edits"], data["test_inputs"]):
                for ti in tis:
                    edits.append(edit)
                    test_inputs.append(ti)
        else:
            edits = data["edits"]
            test_inputs = data["test_inputs"]
        self.edits = edits
        self.test_inputs = [[t[0].strip(), t[1]] for t in test_inputs]

    def test_scores(self, preds: List, **kwargs):
        # Eval edit and check equality.
        mode = kwargs.get("mode", "equivalence")
        mean = kwargs.get("mean", False)
        assert len(preds) == len(self.test_inputs)
        if mode == "equivalence":
            scores = []
            for pred, test_input_a in zip(preds, self.test_inputs):
                ti_group_scores = []
                if pred.isinstance(str):
                    test_input_a = [test_input_a]
                    pred = [pred]
                for p, ti_a in zip(pred, test_input_a):
                    try:
                        ti_group_scores.append(bbq_equivalence_test(p, ti_a[1]))
                    except IndexError:
                        ti_group_scores.append(0)
                scores.append(np.mean(ti_group_scores))
        else:
            raise NotImplementedError("Only equivalence mode is implemented for now.")
        return np.mean(scores) if mean else scores

    def save(self, path: str):
        # Save edit_queries, edits, test_inp_queries, test_inputs
        # as well as the prompts to a json files under path.
        os.makedirs(path, exist_ok=True)

        dd = {
            "edit_prompt": repr(self.edit_prompt),
            "test_input_prompt": repr(self.test_input_prompt.prompt_func),
            "edit_queries": self.edit_queries,
            "edits": self.edits,
            "test_inp_queries": self.test_inp_queries,
            "test_inputs": self.test_inputs,
        }
        with open(
            os.path.join(path, f"{self.data_file_name}_ti{self.ti_num}_out.json"), "w"
        ) as f:
            json.dump(dd, f)


class BBNLI(EditDataset):
    def __init__(
        self,
    ):
        self.data_file_name = "bbnli_qa_short"
        data_file = f"{PROJECTP}/source/bbnli/{self.data_file_name}.csv"
        self.edit_prompt = Prompt(f"{PROJECTP}/prompts/bbnli/sample_edit.txt")
        self.test_input_prompt = Prompt(
            f"{PROJECTP}/prompts/bbnli/sample_test_inputs.txt"
        )
        self.read_data(data_file)

        # Will be used when sampling answers to test inputs.
        self.form_with_edit = (
            "{edit}"
            " Given the above statement, answer the following"
            " question with yes, no or maybe."
            " {question}"
        )
        self.form_without_edit = (
            "Answer the following question with yes, no or maybe. {question}"
        )

    def read_data(self, data_file: str):
        self.edit_queries = pd.read_csv(data_file)
        self.premises = ps = self.edit_queries["premise"].tolist()
        self.questions = qs = self.edit_queries["question"].tolist()
        self.edit_queries = [
            self.edit_prompt.format(premise=p, question=q) for p, q in zip(ps, qs)
        ]

    def sample_edit(self, config: GPTConfig, gpt: GPTCache):
        self.edits = []
        for d in self.edit_queries:
            self.edits.append(gpt.generate(d, max_tokens=config.max_tokens))
        self._post_process_edits()

    def _post_process_edits(self):
        self.edits = [self.edit_prompt.out_func(e) for e in self.edits]

    def sample_test_inputs(self, config: GPTConfig, gpt: GPTCache, ti_num: int = 1):
        self.ti_num = ti_num
        self.test_inp_queries = [
            self.test_input_prompt.format(guideline=g, question=q, answer="Yes")
            for g, q in zip(self.edits, self.questions)
        ]
        self.test_inputs = []
        for d in self.test_inp_queries:
            self.test_inputs.append(
                [
                    gpt.generate(d, max_tokens=config.max_tokens, index=i)
                    for i in range(ti_num)
                ]
            )
        self._post_process_test_inputs()

    def _post_process_test_inputs(self):
        self.test_inputs = [
            [self.test_input_prompt.out_func(c) for c in t] for t in self.test_inputs
        ]

    def load_test_inputs(self, test_input_path: str, flattened=False):
        with open(test_input_path, "r") as f:
            data = json.load(f)
        if flattened:
            edits, test_inputs = [], []
            for edit, tis in zip(data["edits"], data["test_inputs"]):
                for ti in tis:
                    edits.append(edit)
                    test_inputs.append(ti)
        else:
            edits = data["edits"]
            test_inputs = data["test_inputs"]
        self.edits = edits
        self.test_inputs = [[t[0].strip(), t[1]] for t in test_inputs]

    def test_scores(self, preds: List, **kwargs):
        assert len(preds) == len(self.test_inputs)
        if type(preds[0]) == list:
            for i, pred in enumerate(preds):
                assert len(pred) == len(self.test_inputs[i])
        mode = kwargs.get("mode", "equivalence")
        mean = kwargs.get("mean", False)
        if mode == "equivalence":
            scores = []
            for pred, test_input_a in zip(preds, self.test_inputs):
                if type(pred) == str:
                    test_input_a = [test_input_a]
                    pred = [pred]
                scores.append(
                    np.mean(
                        [
                            bbnli_equivalence_test(p, ti_a[1])
                            for p, ti_a in zip(pred, test_input_a)
                        ]
                    )
                )
        else:
            raise NotImplementedError("Only equivalence mode is implemented for now.")
        return np.mean(scores) if mean else scores

    def save(self, path: str):
        # Save edit_queries, edits, test_inp_queries, test_inputs
        # as well as the prompts to a json files under path.
        os.makedirs(path, exist_ok=True)

        dd = {
            "edit_prompt": repr(self.edit_prompt),
            "test_input_prompt": repr(self.test_input_prompt.prompt_func),
            "edit_queries": self.edit_queries,
            "edits": self.edits,
            "test_inp_queries": self.test_inp_queries,
            "test_inputs": self.test_inputs,
        }
        with open(
            os.path.join(path, f"{self.data_file_name}_ti{self.ti_num}_out.json"), "w"
        ) as f:
            json.dump(dd, f)


class RealToxicityPrompts(EditDataset):
    def __init__(self):
        self.data_file_name = "realtox_n300_gpt2_challenging"
        data_file = f"{PROJECTP}/source/realtoxprompts/{self.data_file_name}.csv"
        self.edit_prompt = Prompt(
            f"{PROJECTP}/prompts/real_toxicity_prompts/sample_edit.txt"
        )
        self.test_input_prompt = Prompt(
            f"{PROJECTP}/prompts/real_toxicity_prompts/sample_test_inputs.txt"
        )
        self.read_data(data_file)

    def read_data(self, data_file: str):
        self.edit_queries = pd.read_csv(data_file)

        # Real Toxicity Prompts
        ps = self.edit_queries["prompt"].tolist()
        ps = [eval(p)["text"] for p in ps]
        self.prompt_phrase = ps

        # Continuations by model e.g. GPT-2
        qs = self.edit_queries["continuation"].tolist()
        qs = [eval(q)["text"].strip() for q in qs]
        # qs = [eval(q)["text"].lstrip(p) for p, q in zip(ps, qs)]
        self.completions = qs

        # Prepare the prompt for getting edits.
        self.edit_queries = [
            self.edit_prompt.format(prompt=p, completion=q) for p, q in zip(ps, qs)
        ]

    def sample_edit(self, config: GPTConfig, gpt: GPTCache):
        self.edits = []
        print("Sampling edits...")
        for d in tqdm(self.edit_queries):
            self.edits.append(gpt.generate(d, max_tokens=config.max_tokens))
        self._post_process_edits()

    def _post_process_edits(self):
        self.edits = [self.edit_prompt.out_func(e) for e in self.edits]

    def sample_test_inputs(self, config: GPTConfig, gpt: GPTCache, ti_num: int = 1):
        self.ti_num = ti_num
        self.test_inp_queries = [
            self.test_input_prompt.format(
                prompt=p, completion=c, omit_list=", ".join(o)
            )
            for p, c, o in zip(self.prompt_phrase, self.completions, self.edits)
        ]

        self.test_inputs = []
        print("Sampling test inputs...")
        for d in tqdm(self.test_inp_queries):
            self.test_inputs.append(
                [
                    gpt.generate(d, max_tokens=config.max_tokens, index=i)
                    for i in range(ti_num)
                ]
            )
        self._post_process_test_inputs()

    def _post_process_test_inputs(self):
        self.test_inputs = [
            [self.test_input_prompt.out_func(c) for c in t] for t in self.test_inputs
        ]

    def test_scores(self, preds: List[List[List[str]]], **kwargs):
        """
        preds: For each real tox prompt, for each query iteration,
        for each generated prompt.
        """
        assert len(preds) == len(self.test_inputs)
        for i, pred in enumerate(preds):
            assert len(pred) == len(self.test_inputs[i])
        mean = kwargs.get("mean", False)
        scores = []
        for pred, omits in zip(preds, self.edits):
            scores.append(np.mean([realtox_pres_test(p, omits) for p in pred]))

        return np.mean(scores) if mean else scores

    def save(self, path: str):
        # Save edit_queries, edits, test_inp_queries, test_inputs
        # as well as the prompts to a json files under path.
        os.makedirs(path, exist_ok=True)
        dd = {
            "edit_prompt": repr(self.edit_prompt),
            "test_input_prompt": repr(self.test_input_prompt.prompt_func),
            "edit_queries": self.edit_queries,
            "edits": self.edits,
            "test_inp_queries": self.test_inp_queries,
            "test_inputs": self.test_inputs,
        }
        with open(
            os.path.join(path, f"{self.data_file_name}_ti{self.ti_num}_out.json"), "w"
        ) as f:
            json.dump(dd, f)
