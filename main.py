from dataset import BBNLI, BBQ, Arithmetic
from config import (
    BBNLITestInputsConfig,
    BBNLIEditConfig,
    BBQTestInputsConfig,
    BBQEditConfig,
    ArithmeticEditConfig,
    ArithmeticTestInputsConfig,
)
from gptcache import GPTCache

class_dict = {
    "BBNLI": BBNLI,
    "BBQ": BBQ,
    "Arithmetic": Arithmetic,
}

edit_config_dict = {
    "BBNLI": BBNLIEditConfig,
    "BBQ": BBQEditConfig,
    "Arithmetic": ArithmeticEditConfig,
}

ti_config_dict = {
    "BBNLI": BBNLITestInputsConfig,
    "BBQ": BBQTestInputsConfig,
    "Arithmetic": ArithmeticTestInputsConfig,
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

    preds = [["10"] * 6 + ["8"] * 2, ["7"] * 4 + ["8"] * 4]
    scores = cc.test_scores(preds, mean=True)
    print(scores)


if __name__ == "__main__":
    run("Arithmetic")
