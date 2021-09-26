import logging.config
from os import environ
from argparse import ArgumentParser, FileType
import yaml
from yaml.loader import SafeLoader
from pika import ConnectionParameters, credentials

_parser = ArgumentParser()
_parser.add_argument('--worker-config', type=FileType('r'), default='config/config.yaml',
                     help="The worker config YAML file to load.")
_parser.add_argument('--log-config', type=FileType('r'), default='config/logging.ini',
                     help="Path to log config file.")

_args = _parser.parse_known_args()[0]
logging.config.fileConfig(_args.log_config.name)

with open(_args.worker_config.name, 'r', encoding='utf-8') as f:
    _config = yaml.load(f, Loader=SafeLoader)

EXCHANGE_NAME = 'translation'
WORKER_PARAMETERS = _config['parameters']

ROUTING_KEYS = []
for language_pair in _config['language_pairs']:
    for domain in _config['domains']:
        for input_type, allowed in _config['input_types'].items():
            if allowed:
                # routing key format: exchange_name.src.tgt.domain.input_type
                key = f'{EXCHANGE_NAME}.{language_pair["source"]}.{language_pair["target"]}.{domain}.{input_type}'
                ROUTING_KEYS.append(key)

MQ_PARAMETERS = ConnectionParameters(
    host=environ.get('MQ_HOST', 'localhost'),
    port=int(environ.get('MQ_PORT', '5672')),
    credentials=credentials.PlainCredentials(
        username=environ.get('MQ_USERNAME', 'guest'),
        password=environ.get('MQ_PASSWORD', 'guest')
    )
)
