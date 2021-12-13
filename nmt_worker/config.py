import yaml
from yaml.loader import SafeLoader
from typing import List

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
    heartbeat: int = 30
    connection_name: str = 'Translation worker'

    class Config:
        env_prefix = 'mq_'


class ModelConfig(BaseModel):
    language_pairs: List[str]  # a list of hyphen-separated input/output language pairs
    domains: List[str] = ["general"]
    modular: bool = False
    checkpoint_path: str = "models/checkpoint_best.pt"
    dict_dir: str = "models/dicts/"
    sentencepiece_dir: str = "models/sentencepiece/"
    sentencepiece_prefix: str = "sp-model"


def read_model_config(file_path: str) -> ModelConfig:
    with open(file_path, 'r', encoding='utf-8') as f:
        model_config = ModelConfig(**yaml.load(f, Loader=SafeLoader))

    return model_config
