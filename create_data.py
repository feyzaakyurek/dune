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

project_p = "/projectnb/llamagrp/feyzanb/dune"

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


def create_edit_data(name: str, ti_num: int = 2):
    cc = class_dict[name]()
    edit_config = edit_config_dict[name]()
    ti_config = edit_config_dict[name]()
    cachep = f"{project_p}/cache/{name}/cache_{edit_config.model_name}.json"
    gpt = GPTCache(
        cache_loc=cachep,
        key_loc="openai_key.txt",
        engine=edit_config.model_name,
    )
    cc.sample_edit(edit_config, gpt)
    cc.sample_test_inputs(ti_config, gpt, ti_num)
    cc.save(f"{project_p}/outputs/{name}")


if __name__ == "__main__":
    create_edit_data("BBNLI", ti_num=8)
