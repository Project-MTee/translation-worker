import json
from dataclasses import dataclass, asdict
from marshmallow import Schema, fields, validate
from typing import Optional, Union


@dataclass
class MQItem:
    """
    Parameters of a request sent via RabbitMQ.
    """
    delivery_tag: Optional[int]
    reply_to: Optional[str]
    correlation_id: Optional[str]
    request: dict


class RequestSchema(Schema):
    text = fields.Raw(required=True, validate=(
        lambda obj: type(obj) == str or (type(obj) == list and all(type(item) == str for item in obj))),
                      )
    src = fields.Str(required=True)
    tgt = fields.Str(required=True)
    domain = fields.Str(required=True)
    input_type = fields.Str(required=True, validate=validate.OneOf(['plain', 'document', 'web', 'asr']))


@dataclass
class Request:
    """
    A dataclass that can be used to store NMT requests
    """
    text: Optional[Union[str, list]]
    src: str
    tgt: str
    domain: str
    input_type: str


@dataclass
class Response:
    """
    A dataclass that can be used to store responses and transfer them over the message queue if needed.
    """
    translation: Optional[Union[str, list]] = None
    status_code: int = 200
    status: str = 'OK'

    def encode(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf8")
