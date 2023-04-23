class Prompt(object):
    def __init__(self, prompt_file: str):
        assert prompt_file.endswith(".txt"), "Prompt file must be a .txt file"
        with open(prompt_file, "r") as f:
            self.prompt_func = f.read()
        self.text, self.out_func = self.prompt_func.split("|||||")

        # Lambda expression to process output
        self.out_func = eval(self.out_func)

    def format(self, *args, **kwargs):
        return self.text.format(*args, **kwargs)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text
