from dataset import BBNLI, BBQ, Arithmetic, RealToxicityPrompts, NewInfo, ARC
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

from eval_utils import eval_api, eval_hf
from util import bm25_retriever_generator, get_edits_with_scope, gpt3_retriever
import ipdb
import json
import os
import argparse

class_dict = {
    "BBNLI": BBNLI,
    "BBQ": BBQ,
    "Arithmetic": Arithmetic,
    "NewInfo": NewInfo,
    "RealTox": RealToxicityPrompts,
    "ARC": ARC,
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


def create_retriever(args):
    retriever_mechanism = args.retriever_mechanism
    if retriever_mechanism == "bm25":
        return bm25_retriever_generator(args.num_retrievals)
    elif retriever_mechanism == "gpt3":
        return gpt3_retriever
    elif retriever_mechanism == "scope":

        def fscp(ti, edit):
            return "Information: " + edit.strip() + " Question: " + ti.strip()

        def scope_retriever(edits_all, test_inputs):
            return get_edits_with_scope(
                "/projectnb/llamagrp/feyzanb/dune/outputs/scope_classifier/distilbert-base-cased",
                edits_all,
                test_inputs,
                fscp,
                args.device,
                args.scope_cache,
                args.num_retrievals,
            )

        return scope_retriever
    elif retriever_mechanism is None:
        return None
    else:
        raise ValueError("retriever_mechanism not recognized")


def get_all_edits(args):
    if args.with_edit:
        with open(
            "/projectnb/llamagrp/feyzanb/dune/source/fine-tuning-pool/all_shuffled_edits.txt",
            "r",
        ) as f:
            all_edits = [line.strip() for line in f.readlines()]
        return all_edits
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small")
    parser.add_argument("--dataset_name", type=str, default="BBQ")
    parser.add_argument(
        "--filename_queries",
        type=str,
        help="json file to read test queries from with 'test_inputs' and 'edits' keys",
        default="bbq_questions_answers_ti8_out",
    )
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--with_edit", action="store_true")
    parser.add_argument("--no_edit", dest="with_edit", action="store_false")
    parser.add_argument("--retriever_mechanism", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--from_flax", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scope_cache", type=str, default=None)
    parser.add_argument("--generations_cache", type=str, default=None)
    parser.add_argument("--api", type=str, default=None)
    parser.add_argument("--num_retrievals", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--chat_prompt_dict_path", type=str, default=None)
    args = parser.parse_args()

    if "gpt" in args.model_name:
        eval_func = eval_api
        args.api = "openai"
    elif "flan" in args.model_name:
        eval_func = eval_hf
    elif "bard" in args.model_name:
        eval_func = eval_api
        args.api = "bard"
    else:
        raise ValueError("model_name not recognized")

    # Load dataset.
    cc = class_dict[args.dataset_name]()
    cc.load_test_inputs(args.filename_queries, flattened=True)

    outputs = eval_func(
        cc,
        args,
        model_name=args.model_name,
        with_edit=args.with_edit,
        retriever=create_retriever(args),
        edits_all=get_all_edits(args),
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    # Save.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "outputs.json"), "w") as f:
        print("Saving to ", os.path.join(args.output_dir, "outputs.json"))
        outputs["args"] = vars(args)
        json.dump(outputs, f)


if __name__ == "__main__":
    main()
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

    # filename = "bbq_questions_answers_ti8_out"
    # name = "BBQ"
    # # filename = "bbnli_qa_200_ti8_out"
    # # name = "BBNLI"
    # for model_name_suffix in ["small", "base", "large", "xl", "xxl"]:
    #     model_name = (
    #         "/projectnb/llamagrp/feyzanb/dune/outputs/all_fine_tune_flant5"
    #         + model_name_suffix
    #     )
    #     model_name_ = model_name.replace("/", "_")
    #     out_path = f"{project_p}/outputs/{name}"
    #     retriver = None
    #     ret_name = "" if retriver is None else "_bm25"

    #     for edit in [True]:
    #         edit_name = "edit" if edit else "noedit"
    #         dd = eval_hf_bbq(
    #             name,
    #             model_name,
    #             f"{out_path}/{filename}.json",
    #             with_edit=edit,
    #             retriever=retriver,
    #             from_flax=True,
    #         )
    #         with open(
    #             f"{out_path}/{filename}_{model_name_}_{edit_name}{ret_name}.json", "w"
    #         ) as f:
    #             json.dump(dd, f)

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
