# models.py
from transformers import pipeline
import functools, time




# ----- Decorators -----
def log_call(func):
    def wrapper(*args, **kwargs):
        print(f"[LOG] {func.__name__} called")
        start = time.time()
        res = func(*args, **kwargs)
        print(f"[LOG] {func.__name__} finished in {time.time()-start:.2f}s")
        return res
    return wrapper

def cache_result(func):
    cache = {}
    def wrapper(*args, **kwargs):
        if args[1] in cache:
            print("[CACHE] returning cached result.......")
            return cache[args[1]]
        res = func(*args, **kwargs)
        cache[args[1]] = res
        return res
    return wrapper

# ----- Base Class -----
class ModelRunner:
    def __init__(self, model_name):
        self._model_name = model_name  # encapsulation
        self._pipeline = None

    def load(self):
        raise NotImplementedError

    def run(self, text):
        raise NotImplementedError

# ----- GPT Runner -----
class GPTModel(ModelRunner):
    def __init__(self, model_name="openai/gpt-oss-20b"):
        super().__init__(model_name)

    @log_call
    def load(self):
        self._pipeline = pipeline("text-generation", model=self._model_name)

    @cache_result
    @log_call
    def run(self, text):
        if self._pipeline is None:
            self.load()
        output = self._pipeline(text, max_new_tokens=50, do_sample=True)[0]["generated_text"]
        return output
