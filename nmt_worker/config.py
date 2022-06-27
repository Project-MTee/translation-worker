import os
import yaml
from yaml.loader import SafeLoader
from typing import List, Optional, Any
from pydantic import BaseSettings, BaseModel


class MQConfig(BaseSettings):
    """
    Imports MQ configuration from environment variables
    """
    host: str = 'localhost'
    port: int = 5672
    username: str = 'guest'
    password: str = 'guest'
    exchange: str = 'translation'
    heartbeat: int = 60
    connection_name: str = 'Translation worker'

    class Config:
        env_prefix = 'mq_'


class WorkerConfig(BaseSettings):
    """
    Imports general workr configuration from environment variables
    """
    max_input_length: int = 10000

    class Config:
        env_prefix = 'worker_'


class ModelConfig(BaseModel):
    language_pairs: List[str]  # a list of hyphen-separated input/output language pairs
    domains: List[str] = ["general"]

    huggingface: Optional[str] = None
    model_root: str = ""
    modular: bool = False

    checkpoint: str = "checkpoint_best.pt"
    dict_dir: str = ""
    sentencepiece_dir: str = ""
    sentencepiece_prefix: str = "sp-model"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.checkpoint = os.path.join(self.model_root, self.checkpoint)
        self.sentencepiece_prefix = os.path.join(self.model_root, self.sentencepiece_dir, self.sentencepiece_prefix)
        self.dict_dir = os.path.join(self.model_root, self.dict_dir)

    def download(self):
        if self.huggingface is not None:
            from huggingface_hub import Repository
            Repository(clone_from=self.huggingface, local_dir=self.model_root)
        else:
            raise ValueError("Model cannot be downloaded, no HuggingFace repository specified.")


def read_model_config(file_path: str) -> ModelConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        model_config = ModelConfig(**yaml.load(f, Loader=SafeLoader))

    return model_config


mq_config = MQConfig()
worker_config = WorkerConfig()
