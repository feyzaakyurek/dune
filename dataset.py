from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from prompt import Prompt
from config import GPTConfig
from gptcache import GPTCache
from util import bbq_equivalence_test, bbnli_equivalence_test
import numpy as np
import pandas as pd
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


class Arithmetic(EditDataset):
    def __init__(
        self,
    ):
        data_file = "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_validated_chainofreasoning.csv"
        self.edit_prompt = None
        self.test_input_prompt = None
        self.read_data(data_file)

    def read_data(self, data_file):
        self.data = pd.read_csv(data_file)[:3]

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

    def test_scores(self, preds: List[List[str]], **kwargs):
        # Eval edit and check equality.
        gold = [str(eval(e.split(" = ")[0])) for e in self.edits]
        mode = kwargs.get("mode", "equality")
        mean = kwargs.get("mean", False)
        if mode == "equality":
            scores = []
            for pred, g in zip(preds, gold):
                scores.append(np.mean([p == g for p in pred]))
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


class BBQ(EditDataset):
    def __init__(
        self,
    ):
        data_file = (
            "/projectnb/llamagrp/feyzanb/dune/source/bbq/bbq_questions_answers_short.csv"
        )
        self.edit_prompt = Prompt(
            "/projectnb/llamagrp/feyzanb/dune/prompts/bbq/sample_edit.txt"
        )
        self.test_input_prompt = Prompt(
            "/projectnb/llamagrp/feyzanb/dune/prompts/bbq/sample_test_inputs.txt"
        )
        self.read_data(data_file)

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

    def test_scores(self, preds: List[List[str]], **kwargs):
        # Eval edit and check equality.
        mode = kwargs.get("mode", "equality")
        mean = kwargs.get("mean", False)
        if mode == "equavalence":
            scores = []
            for pred, test_input_a in zip(preds, self.test_inputs):
                scores.append(
                    np.mean(
                        [
                            bbq_equivalence_test(p, ti_a[1])
                            for p, ti_a in zip(pred, test_input_a)
                        ]
                    )
                )
        else:
            raise NotImplementedError("Only equavalence mode is implemented for now.")
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
        with open(os.path.join(path, "output.json"), "w") as f:
            json.dump(dd, f)


class BBNLI(EditDataset):
    def __init__(
        self,
    ):
        data_file = "/projectnb/llamagrp/feyzanb/dune/source/bbnli/bbnli_qa_short.csv"
        self.edit_prompt = Prompt(
            "/projectnb/llamagrp/feyzanb/dune/prompts/bbnli/sample_edit.txt"
        )
        self.test_input_prompt = Prompt(
            "/projectnb/llamagrp/feyzanb/dune/prompts/bbnli/sample_test_inputs.txt"
        )
        self.read_data(data_file)

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

    def test_scores(self, preds: List[List[str]], **kwargs):
        assert len(preds) == len(self.test_inputs)
        for i, pred in enumerate(preds):
            assert len(pred) == len(self.test_inputs[i])
        mode = kwargs.get("mode", "equivalence")
        mean = kwargs.get("mean", False)
        if mode == "equivalence":
            scores = []
            for pred, test_input_a in zip(preds, self.test_inputs):
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
        with open(os.path.join(path, "output.json"), "w") as f:
            json.dump(dd, f)
