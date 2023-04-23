from dataclasses import dataclass, field
from typing import List
from abc import ABC


@dataclass
class GPTConfig(ABC):
    pass


@dataclass
class BBQEditConfig(GPTConfig):
    max_tokens: int = 100
    model_name: str = "gpt-3.5-turbo"


@dataclass
class BBQTestInputsConfig(GPTConfig):
    max_tokens: int = 100
    stop: List[str] = field(default_factory=lambda: ["Example"])
    model_name: str = "gpt-3.5-turbo"


@dataclass
class ArithmeticEditConfig(GPTConfig):
    max_tokens: int = 100
    model_name: str = "gpt-4"


@dataclass
class ArithmeticTestInputsConfig(GPTConfig):
    max_tokens: int = 100
    stop: List[str] = field(default_factory=lambda: ["Example"])
    model_name: str = "gpt-4"


@dataclass
class BBNLIEditConfig(GPTConfig):
    max_tokens: int = 100
    model_name: str = "gpt-3.5-turbo"


@dataclass
class BBNLITestInputsConfig(GPTConfig):
    max_tokens: int = 100
    stop: List[str] = field(default_factory=lambda: ["Example"])
    model_name: str = "gpt-3.5-turbo"
