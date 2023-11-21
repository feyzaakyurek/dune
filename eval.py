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
from util import (
    bm25_retriever_generator,
    get_edits_with_scope,
    gpt3_retriever,
    dpr_retriever_generator,
    SCOPECLASSIFIERPATH,
    ALLEDITSPATH,
)
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


def create_retriever(args):
    retriever_mechanism = args.retriever_mechanism
    if retriever_mechanism == "bm25":
        return bm25_retriever_generator(args.num_retrievals)
    elif retriever_mechanism == "gpt3":
        return gpt3_retriever
    elif retriever_mechanism == "dpr":
        return dpr_retriever_generator(args.faiss_index_path)
    elif retriever_mechanism == "scope":

        def fscp(ti, edit):
            return "Information: " + edit.strip() + " Question: " + ti.strip()

        def scope_retriever(edits_all, test_inputs):
            return get_edits_with_scope(
                SCOPECLASSIFIERPATH,
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
        raise ValueError(f"Invalid: {retriever_mechanism} for retriever.")


def get_all_edits(args):
    if args.with_edit:
        with open(
            ALLEDITSPATH,
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
    parser.add_argument("--from_flax", action="store_true", help="If using Flax model")
    parser.add_argument("--llama", action="store_true", help="If using Flax model")
    parser.add_argument("--peft", action="store_true", help="If finetuning with PEFT")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scope_cache", type=str, default=None)
    parser.add_argument("--generations_cache", type=str, default=None)
    parser.add_argument("--api", type=str, default=None)
    parser.add_argument("--num_retrievals", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--chat_prompt_dict_path", type=str, default=None)
    parser.add_argument("--faiss_index_path", type=str, default=None)
    args = parser.parse_args()

    if "gpt" in args.model_name:
        eval_func = eval_api
        args.api = "openai"
    elif "flan" in args.model_name or "llama" in args.model_name:
        eval_func = eval_hf
    elif "bard" in args.model_name:
        eval_func = eval_api
        args.api = "bard"
    else:
        raise ValueError(f"{args.model_name} not recognized")

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
