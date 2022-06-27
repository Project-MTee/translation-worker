import json
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder

from nmt_worker import worker_config


class InputType(Enum):
    PLAIN = 'plain'
    HTML = 'web'
    XML = 'document'
    ASR = 'asr'


class Request(BaseModel):
    """
    A class that can be used to store NMT requests
    """
    text: Union[str, list] = Field(..., max_length=worker_config.max_input_length)
    src: str
    tgt: str
    domain: str
    input_type: InputType = InputType.PLAIN


@dataclass
class Response:
    """
    A dataclass that can be used to store responses and transfer them over the message queue if needed.
    """
    translation: Optional[Union[str, list]] = None
    status_code: int = 200
    status: str = 'OK'

    def encode(self) -> bytes:
        return json.dumps(self, default=pydantic_encoder).encode()
