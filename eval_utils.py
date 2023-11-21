from dataset import BBNLI, BBQ, Arithmetic, RealToxicityPrompts
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from transformers import FlaxT5ForConditionalGeneration, AutoModelForCausalLM

from typing import List, Dict
from gptcache import GPTCache
from util import query_model, Cache, PROJECTP
from tqdm import tqdm
import ipdb


class_dict = {
    "BBNLI": BBNLI,
    "BBQ": BBQ,
    "Arithmetic": Arithmetic,
    "RealTox": RealToxicityPrompts,
}


def eval_api(
    cc,
    args,
    model_name: str = "gpt-3.5-turbo",
    with_edit: bool = False,
    retriever: callable = None,
    edits_all: List[str] = None,
    max_new_tokens=20,
    device="cuda:0",
) -> Dict:
    if args.generations_cache is not None:
        print("Recreating cache...")
    # Create gpt cache.
    if args.api == "openai":
        cache = GPTCache(
            cache_loc=f"{project_p}/cache/cache_{model_name}.json",
            key_loc="openai_key.txt",
            engine=model_name,
            chat_prompt_dict_path=args.chat_prompt_dict_path,
        )
    elif args.api == "bard":
        from bardcache import BardCache

        cache = BardCache(
            cache_loc=f"{project_p}/cache/cache_bard.json", key_loc="palm_api_key.txt"
        )
    else:
        raise ValueError("api name not recognized")

    # Get edits.
    test_inputs = cc.get_test_inputs_only()
    edits = None

    # Get predictions.
    if with_edit:
        # Get edits.
        edits = cc.edits if retriever is None else retriever(edits_all, test_inputs)
        fwe = cc.form_with_edit

        # Percent of edits matching gold edits.
        pem = [e == g for e, g in zip(edits, cc.edits)]
        print("Percent of edits matching gold edits: ", sum(pem) / len(pem))

    else:
        edits = [None for _ in test_inputs]

    if hasattr(cc, "form_without_edit"):
        fwoe = cc.form_without_edit
    else:
        fwoe = "{question}"

    queries = []
    for edit, t in zip(edits, test_inputs):
        if edit is None:
            queries.append(fwoe.format(question=t.strip()))
        else:
            queries.append(fwe.format(edit=edit, question=t.strip()))

    preds = []
    for q in tqdm(queries, desc="Generating..."):
        preds.append(cache.generate(q, max_length=max_new_tokens))

    score = cc.test_scores(preds, mean=True)
    print("Mean score: ", score)
    return {
        "scores": score,
        "preds": preds,
        "edits": edits,
        "test_inputs": test_inputs,
    }


def load_model_tokenizer(args, model_name, device):
    assert not args.llama or not args.from_flax
    assert not args.from_flax or not args.peft

    # Load model and tokenizer.
    if args.from_flax:
        model = FlaxT5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif args.llama:
        if args.peft:
            from peft import PeftModel, PeftConfig

            pconfig = PeftConfig.from_pretrained(model_name)
            bmbp = pconfig.base_model_name_or_path
            model = AutoModelForCausalLM.from_pretrained(bmbp).to(device)
            model = PeftModel.from_pretrained(model, model_name)
            tokenizer = AutoTokenizer.from_pretrained(pconfig.base_model_name_or_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def eval_hf(
    cc,
    args,
    model_name: str,
    with_edit: bool = False,
    retriever: callable = None,
    edits_all: List[str] = None,
    max_new_tokens=20,
    device="cuda:0",
) -> Dict:
    model, tokenizer = load_model_tokenizer(args, model_name, device)

    test_inputs = cc.get_test_inputs_only()
    edits = None

    # Get predictions.
    if with_edit:
        # Get edits.
        edits = cc.edits if retriever is None else retriever(edits_all, test_inputs)
        fwe = cc.form_with_edit

        # Percent of edits matching gold edits.
        pem = [e == g for e, g in zip(edits, cc.edits)]
        print("Percent of edits matching gold edits: ", sum(pem) / len(pem))

    else:
        edits = [None for _ in test_inputs]

    # Identify format to use without edit.
    if hasattr(cc, "form_without_edit"):
        fwoe = cc.form_without_edit
    else:
        fwoe = "{question}"

    # Create queries.
    queries = []
    for edit, t in zip(edits, test_inputs):
        if edit is None:
            qq = fwoe.format(question=t.strip())
        else:
            qq = fwe.format(edit=edit, question=t.strip())

        # If the model is a chat model we need to add the chat prompt.
        # e.g. <s>[INST] {query} [/INST]
        if args.llama and "chat" in model_name:
            qq = f"<s>[INST] {qq} [/INST]"

        queries.append(qq)

    # Create cache.
    if args.generations_cache is not None:
        cache = Cache(args.generations_cache)
    else:
        cache = None

    preds = query_model(
        queries,
        model,
        tokenizer,
        device=device,
        batch_size=args.batch_size,
        do_sample=True if args.llama else False,
        max_length=max_new_tokens,
        flax=args.from_flax,
        cache=cache,
    )

    # Get the part after [/INST]
    if args.llama and "chat" in model_name:
        preds = [p.split("[/INST]")[1].strip() for p in preds]

    score = cc.test_scores(preds, mean=True)
    print("Mean score: ", score)
    return {
        "scores": score,
        "preds": preds,
        "edits": edits,
        "test_inputs": test_inputs,
    }
