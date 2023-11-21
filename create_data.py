from dataset import BBNLI, BBQ, Arithmetic, RealToxicityPrompts
from config import (
    BBNLITestInputsConfig,
    BBNLIEditConfig,
    BBQTestInputsConfig,
    BBQEditConfig,
    ArithmeticEditConfig,
    ArithmeticTestInputsConfig,
    RealToxEditConfig,
    RealToxTestInputsConfig,
)
from gptcache import GPTCache
from eval import eval_openai, eval_hf_bbnli, eval_hf_bbq, eval_bard_bbq, eval_hf_wscope
from util import bm25_retriever
import ipdb
import json, os
from tqdm import tqdm
import argparse

class_dict = {
    "BBNLI": BBNLI,
    "BBQ": BBQ,
    "Arithmetic": Arithmetic,
    "RealTox": RealToxicityPrompts,
}

edit_config_dict = {
    "BBNLI": BBNLIEditConfig,
    "BBQ": BBQEditConfig,
    "Arithmetic": ArithmeticEditConfig,
    "RealTox": RealToxEditConfig,
}

ti_config_dict = {
    "BBNLI": BBNLITestInputsConfig,
    "BBQ": BBQTestInputsConfig,
    "Arithmetic": ArithmeticTestInputsConfig,
    "RealTox": RealToxTestInputsConfig,
}

project_p = "/projectnb/llamagrp/feyzanb/dune"


def create_edit_data(name: str, ti_num: int = 2):
    cc = class_dict[name]()
    edit_config = edit_config_dict[name]()
    ti_config = edit_config_dict[name]()
    gpt = GPTCache(
        cache_loc=f"{project_p}/cache/{name}/cache_{edit_config.model_name}.json",
        key_loc="openai_key.txt",
        engine=edit_config.model_name,
    )
    cc.sample_edit(edit_config, gpt)
    cc.sample_test_inputs(ti_config, gpt, ti_num)
    cc.save(f"{project_p}/outputs/{name}")


if __name__ == "__main__":
    # DATA GENERATION #

    create_edit_data("BBNLI", ti_num=8)
    create_edit_data("RealTox", ti_num=1)
