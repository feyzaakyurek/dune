from dataset import BBNLI, BBQ
from config import (
    BBNLITestInputsConfig,
    BBNLIEditConfig,
    BBQTestInputsConfig,
    BBQEditConfig,
)
from gptcache import GPTCache

class_dict = {
    "BBNLI": BBNLI,
    "BBQ": BBQ,
}

edit_config_dict = {
    "BBNLI": BBNLIEditConfig,
    "BBQ": BBQEditConfig,
}

ti_config_dict = {
    "BBNLI": BBNLITestInputsConfig,
    "BBQ": BBQTestInputsConfig,
}

project_p = "/projectnb/llamagrp/feyzanb/dune"


def run(name: str, ti_num: int = 2):
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
    run("BBQ")
