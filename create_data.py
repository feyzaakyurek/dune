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

    # create_edit_data("BBNLI", ti_num=8)
    # create_edit_data("RealTox", ti_num=1)

    # EVAL OPENAI #

    # filename = "bbq_questions_answers_short_ti8_out"
    # name = "BBQ"
    # model_name = "gpt-4"
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

    filename = "bbq_questions_answers_ti8_out"
    name = "BBQ"
    # filename = "bbnli_qa_200_ti8_out"
    # name = "BBNLI"
    for model_name_suffix in ["small", "base", "large", "xl", "xxl"]:
        model_name = (
            "/projectnb/llamagrp/feyzanb/dune/outputs/all_fine_tune_flant5"
            + model_name_suffix
        )
        model_name_ = model_name.replace("/", "_")
        out_path = f"{project_p}/outputs/{name}"
        retriver = None
        ret_name = "" if retriver is None else "_bm25"

        for edit in [True]:
            edit_name = "edit" if edit else "noedit"
            dd = eval_hf_bbq(
                name,
                model_name,
                f"{out_path}/{filename}.json",
                with_edit=edit,
                retriever=retriver,
                from_flax=True,
            )
            with open(
                f"{out_path}/{filename}_{model_name_}_{edit_name}{ret_name}.json", "w"
            ) as f:
                json.dump(dd, f)

    # # EVAL FLAN w SCOPE #
    # # Arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    # parser.add_argument("--name", type=str, default="BBQ")
    # parser.add_argument("--filename", type=str, default="bbq_questions_answers_ti8_out")
    # parser.add_argument("--output_dir", type=str, default=None)
    # parser.add_argument("--edit", type=bool, default=True)
    # # parser.add_argument("--retriever", type=bool, default=None)
    # parser.add_argument("--from_flax", type=bool, default=False)
    # args = parser.parse_args()

    # # filename = "bbq_questions_answers_ti8_out"
    # # name = "BBQ"
    # # filename = "bbnli_qa_200_ti8_out"
    # # name = "BBNLI"
    # # for model_name in ["google/flan-t5-small"]:

    # if args.name == "BBQ":
    #     # form_with_edit = "{question} Note that {edit}"
    #     form_with_edit = (
    #         "{edit}"
    #         " Given the above statement, answer the following"
    #         " question with yes, no or maybe."
    #         " {question}"
    #     )
    # elif args.name == "BBNLI":
    #     form_with_edit = (
    #         "{edit}"
    #         " Given the above statement, answer the following"
    #         " question with yes, no or maybe."
    #         " {question}"
    #     )
    # else:
    #     raise ValueError("Define form_with_edit for this dataset")
    # model_name_ = args.model_name.replace("/", "_")
    # classifier_name = (
    #     "/projectnb/llamagrp/feyzanb/dune/outputs/scope_classifier/distilbert-base-cased"
    # )
    # out_path = f"{project_p}/outputs/{args.name}"

    # # Read all edits
    # with open(
    #     "/projectnb/llamagrp/feyzanb/dune/source/fine-tuning-pool/all_shuffled_edits.txt",
    #     "r",
    # ) as f:
    #     all_edits = [line.strip() for line in f.readlines()]

    # def fscp(ti, edit):
    #     return "Information: " + ti + " Question: " + edit

    # dd = eval_hf_wscope(
    #     args.name,
    #     args.model_name,
    #     classifier_name,
    #     f"{out_path}/{args.filename}.json",
    #     all_edits,
    #     fscp,
    #     form_with_edit,
    # )
    # with open(f"{args.output_dir}/{args.filename}_wscope.json", "w") as f:
    #     json.dump(dd, f)

    # EVAL BARD #
    # filename = "bbq_questions_answers_ti8_out"
    # name = "BBQ"
    # filename = "bbnli_qa_200_ti8_out"
    # name = "BBNLI"
    # model_name_ = "bard"
    # out_path = f"{project_p}/outputs/{name}"
    # retriver = bm25_retriever
    # ret_name = "" if retriver is None else "_bm25"

    # for edit in [True]:
    #     edit_name = "edit" if edit else "noedit"
    #     dd = eval_bard_bbq(
    #         name,
    #         f"{out_path}/{filename}.json",
    #         with_edit=edit,
    #         retriever=retriver,
    #     )
    #     with open(
    #         f"{out_path}/{filename}_{model_name_}_{edit_name}{ret_name}.json", "w"
    #     ) as f:
    #         json.dump(dd, f)
