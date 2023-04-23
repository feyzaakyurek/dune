import json
import openai
import os
import ipdb
import sys
import time

# Authors: Jacob Andreas, Feyza Akyurek


class GPTCache:
    def __init__(self, cache_loc, key_loc, engine="gpt-3.5-turbo"):
        self.cache_loc = cache_loc
        self.engine = engine
        if os.path.exists(cache_loc):
            with open(cache_loc) as reader:
                self.cache = json.load(reader)
        else:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_loc), exist_ok=True)
            self.cache = {"scores": {}, "generations": {}}

        with open(key_loc) as reader:
            openai.api_key = reader.read().strip()

    def query(self, utt):
        if utt in self.cache["scores"]:
            return self.cache["scores"][utt]
        print("calling API with", "[" + utt + "]")
        result = openai.Completion.create(
            engine=self.engine,
            prompt=utt,
            max_tokens=0,
            logprobs=0,
            echo=True,
        )
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

    def generate(self, context, max_tokens=100, index=0):
        if (
            context in self.cache["generations"]
            and len(self.cache["generations"][context]) > index
        ):
            return self.cache["generations"][context][index]
        print("calling API with", "[" + context + "]")
        success = False
        retries = 1
        while not success and retries < 20:
            try:
                if "3.5" in self.engine:
                    result = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": context}],
                    )
                    generation = result["choices"][0]["message"]["content"]
                else:
                    result = openai.Completion.create(
                        engine=self.engine,
                        prompt=context,
                        max_tokens=max_tokens,
                        temperature=1.0,
                    )
                    generation = result["choices"][0]["text"]
                success = True
            except Exception:
                wait = retries * 10
                print(
                    f"Error, rate limit reached! Waiting {str(wait)} secs and re-trying..."
                )
                sys.stdout.flush()
                time.sleep(wait)
                retries += 1
        if context not in self.cache["generations"]:
            self.cache["generations"][context] = []
        self.cache["generations"][context].append(generation)
        with open(self.cache_loc, "w") as writer:
            json.dump(self.cache, writer)
        return generation
