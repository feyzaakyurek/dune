from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from prompt import Prompt
from config import GPTConfig
from gptcache import GPTCache
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
    def save(path: str):
        pass


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
