import json
import google.generativeai as palm

# import google.ai.generativelanguage_v1beta2 as glv
import os
import ipdb
import sys
import time

# Authors: Jacob Andreas, Feyza Akyurek

# SAFETY_SETTINGS = [
#     {
#         "category": glv.types.HarmCategory.HARM_CATEGORY_DEROGATORY,
#         "threshold": glv.types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
#     },
#     {
#         "category": glv.types.HarmCategory.HARM_CATEGORY_VIOLENCE,
#         "threshold": glv.types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
#     },
#     {
#         "category": glv.types.HarmCategory.HARM_CATEGORY_SEXUAL,
#         "threshold": glv.types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
#     },
#     {
#         "category": glv.types.HarmCategory.HARM_CATEGORY_DANGEROUS,
#         "threshold": glv.types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
#     },
#     {
#         "category": glv.types.HarmCategory.HARM_CATEGORY_MEDICAL,
#         "threshold": glv.types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
#     },
#     {
#         "category": glv.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
#         "threshold": glv.types.SafetySetting.HarmBlockThreshold.BLOCK_NONE,
#     },
# ]

SAFETY_SETTINGS = [
    palm.types.SafetySettingDict(
        category=palm.types.HarmCategory.HARM_CATEGORY_DEROGATORY,
        threshold=palm.types.HarmBlockThreshold.BLOCK_NONE,
    ),
    palm.types.SafetySettingDict(
        category=palm.types.HarmCategory.HARM_CATEGORY_VIOLENCE,
        threshold=palm.types.HarmBlockThreshold.BLOCK_NONE,
    ),
    palm.types.SafetySettingDict(
        category=palm.types.HarmCategory.HARM_CATEGORY_SEXUAL,
        threshold=palm.types.HarmBlockThreshold.BLOCK_NONE,
    ),
    palm.types.SafetySettingDict(
        category=palm.types.HarmCategory.HARM_CATEGORY_DANGEROUS,
        threshold=palm.types.HarmBlockThreshold.BLOCK_NONE,
    ),
    palm.types.SafetySettingDict(
        category=palm.types.HarmCategory.HARM_CATEGORY_MEDICAL,
        threshold=palm.types.HarmBlockThreshold.BLOCK_NONE,
    ),
    palm.types.SafetySettingDict(
        category=palm.types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        threshold=palm.types.HarmBlockThreshold.BLOCK_NONE,
    ),
    palm.types.SafetySettingDict(
        category=palm.types.HarmCategory.HARM_CATEGORY_TOXICITY,
        threshold=palm.types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


class BardCache:
    def __init__(self, cache_loc, key_loc):
        self.cache_loc = cache_loc
        if os.path.exists(cache_loc):
            with open(cache_loc) as reader:
                self.cache = json.load(reader)
        else:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_loc), exist_ok=True)
            self.cache = {"scores": {}, "generations": {}}

        palm.configure(api_key=open(key_loc, "r").read().strip())

        models = [
            m
            for m in palm.list_models()
            if "generateText" in m.supported_generation_methods
        ]
        self.model = models[0].name

    def query(self, utt, temp=0, max_tokens=10):
        if utt in self.cache["scores"]:
            return self.cache["scores"][utt]
        print("calling API with", "[" + utt + "]")
        result = palm.generate_text(
            model=self.model,
            prompt=utt,
            temperature=0,
            max_output_tokens=10,
        ).result
        self.cache["scores"][utt] = result
        with open(self.cache_loc, "w") as writer:
            json.dump(self.cache, writer)
        return result

    def score(self, context, pred):
        result = self.query(context + pred)
        assert len(result["choices"]) == 1
        result = result["choices"][0]
        offset = result["logprobs"]["text_offset"].index(len(context))
        tokens = result["logprobs"]["tokens"][offset:]
        assert "".join(tokens) == pred

        logprobs = result["logprobs"]["token_logprobs"][offset:]
        if logprobs[0] is None:
            logprobs = logprobs[1:]
        return sum(logprobs)

    def generate(self, context, temp=0, max_length=100, index=0):
        if (
            context in self.cache["generations"]
            and len(self.cache["generations"][context]) > index
        ):
            return self.cache["generations"][context][index]
        # print("calling API with", "[" + context + "]")
        success = False
        retries = 1
        generation = ""
        while not success and retries < 20:
            try:
                generation = palm.generate_text(
                    model=self.model,
                    prompt=context,
                    temperature=temp,
                    max_output_tokens=max_length,
                    safety_settings=SAFETY_SETTINGS,
                ).result
                if generation is None:
                    generation = ""
                generation = generation.split("\n")[0]
                success = True
            except Exception as e:
                wait = retries * 10
                print(e)
                print(f"Error! Waiting {str(wait)} secs and re-trying...")
                sys.stdout.flush()
                time.sleep(wait)
                retries += 1
        if context not in self.cache["generations"] and generation != "":
            self.cache["generations"][context] = []
        if generation != "":
            self.cache["generations"][context].append(generation)
        with open(self.cache_loc, "w") as writer:
            json.dump(self.cache, writer)
        return generation


if __name__ == "__main__":
    bard = BardCache(cache_loc="bard_cache.json", key_loc="palm_api_key.txt")
    print(bard.generate("Hello, I am a"))
