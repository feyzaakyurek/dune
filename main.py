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
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModel,
    AutoTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from util import query_model
import ipdb
import json, os
from tqdm import tqdm

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


def eval_openai(
    name: str, model_name: str, test_inputs_pth: str, with_edit: bool = False
):
    cc = class_dict[name]()
    cc.load_test_inputs(test_inputs_pth)
    gpt = GPTCache(
        cache_loc=f"{project_p}/cache/{name}/cache_{model_name}.json",
        key_loc="openai_key.txt",
        engine=model_name,
    )
    preds = []
    for edit, test_input in tqdm(zip(cc.edits, cc.test_inputs)):
        for t in test_input:
            if with_edit:
                pred = gpt.generate(
                    "Given that " + edit[0].lower() + edit[1:] + " " + t[0].strip(),
                    max_tokens=15,
                )
            else:
                pred = gpt.generate(
                    t[0].strip(),
                    max_tokens=15,
                )
            preds.append(pred)
    scores = cc.test_scores(preds, mean=True)
    print(scores)
    return {"scores": scores, "preds": preds}


def eval_hf(name: str, model_name: str, test_inputs_pth: str, with_edit: bool = False):
    cc = class_dict[name]()
    cc.load_test_inputs(test_inputs_pth)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:0")
    # model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    preds = []
    for edit, test_input in zip(cc.edits, cc.test_inputs):
        if with_edit:
            queries = [
                t[0].strip() + " Note that " + edit[0].lower() + edit[1:]
                for t in test_input
            ]
        else:
            queries = [t[0].strip() for t in test_input]
        preds.append(
            query_model(
                queries,
                model,
                tokenizer,
                device="cuda:0",
                batch_size=1,
                do_sample=False,
                max_length=20,
            )
        )
    scores = cc.test_scores(preds, mean=True)
    print(scores)
    return {"scores": scores, "preds": preds}


if __name__ == "__main__":
    # DATA GENERATION #

    # create_edit_data("BBNLI", ti_num=8)
    # create_edit_data("RealTox", ti_num=1)

    # EVAL OPENAI #

    # filename = "bbq_questions_answers_short_ti8_out_generic_edit"
    # name = "BBQ"
    # model_name = "google/flan-t5-large"
    # model_name_ = model_name.replace("/", "_")
    # out_path = f"{project_p}/outputs/{name}"

    # for edit in [False, True]:
    #     edit_name = "edit" if edit else "noedit"
    #     dd = eval_openai(
    #         name,
    #         model_name,
    #         f"{out_path}/{filename}.json",
    #         with_edit=edit,
    #     )
    #     with open(f"{out_path}/{filename}_{model_name_}_{edit_name}.json", "w") as f:
    #         json.dump(dd, f)

    # EVAL FLAN #

    filename = "bbq_questions_answers_short_ti8_out_generic_edit"
    name = "BBQ"
    model_name = "google/flan-t5-large"
    model_name_ = model_name.replace("/", "_")
    out_path = f"{project_p}/outputs/{name}"

    for edit in [False, True]:
        edit_name = "edit" if edit else "noedit"
        dd = eval_hf(
            name,
            model_name,
            f"{out_path}/{filename}.json",
            with_edit=edit,
        )
        with open(f"{out_path}/{filename}_{model_name_}_{edit_name}.json", "w") as f:
            json.dump(dd, f)
