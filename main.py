import logging.config
from os import environ
from argparse import ArgumentParser, FileType
import yaml
from yaml.loader import SafeLoader
from pika import ConnectionParameters, credentials

from nmt_worker.mq_consumer import MQConsumer
from nmt_worker.translator import Translator

input_types = ["plain", "document", "web", "asr"]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--worker-config', type=FileType('r'), default='models/config.yaml',
                        help="The worker config YAML file to load.")
    parser.add_argument('--log-config', type=FileType('r'), default='logging/logging.ini',
                        help="Path to log config file.")
    args = parser.parse_known_args()[0]
    logging.config.fileConfig(args.log_config.name)

    with open(args.worker_config.name, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    exchange_name = 'translation'

    routing_keys = []  # routing key format: exchange_name.src.tgt.domain.input_type
    for language_pair in config['language_pairs']:
        source, target = language_pair.split('-')
        for domain in config['domains']:
            for input_type in input_types:
                key = f'{exchange_name}.{source}.{target}.{domain}.{input_type}'
                routing_keys.append(key)

    mq_parameters = ConnectionParameters(host=environ.get('MQ_HOST', 'localhost'),
                                         port=int(environ.get('MQ_PORT', '5672')),
                                         credentials=credentials.PlainCredentials(
                                             username=environ.get('MQ_USERNAME', 'guest'),
                                             password=environ.get('MQ_PASSWORD', 'guest')))

    translator = Translator(
        config['modular'],
        config['checkpoint'],
        config['dict_dir'],
        config['sentencepiece_dir'],
        config['sentencepiece_prefix'],
    )
    worker = MQConsumer(translator=translator,
                        connection_parameters=mq_parameters,
                        exchange_name=exchange_name,
                        routing_keys=routing_keys)

    worker.start()
