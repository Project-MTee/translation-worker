import logging.config
from os import environ
from argparse import ArgumentParser, FileType
import yaml
from yaml.loader import SafeLoader
from pika import ConnectionParameters, credentials

from nmt_worker.mq_consumer import MQConsumer
from nmt_worker.translator import Translator

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--worker-config', type=FileType('r'), default='config/config.yaml',
                        help="The worker config YAML file to load.")
    parser.add_argument('--log-config', type=FileType('r'), default='config/logging.ini',
                        help="Path to log config file.")
    args = parser.parse_known_args()[0]
    logging.config.fileConfig(args.log_config.name)

    with open(args.worker_config.name, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=SafeLoader)

    exchange_name = 'translation'
    WORKER_PARAMETERS = config['parameters']

    ROUTING_KEYS = []  # routing key format: exchange_name.src.tgt.domain.input_type
    for language_pair in config['language_pairs']:
        for domain in config['domains']:
            for input_type, allowed in config['input_types'].items():
                if allowed:
                    key = f'{exchange_name}.{language_pair["source"]}.{language_pair["target"]}.{domain}.{input_type}'
                    ROUTING_KEYS.append(key)

    MQ_PARAMETERS = ConnectionParameters(host=environ.get('MQ_HOST', 'localhost'),
                                         port=int(environ.get('MQ_PORT', '5672')),
                                         credentials=credentials.PlainCredentials(
                                             username=environ.get('MQ_USERNAME', 'guest'),
                                             password=environ.get('MQ_PASSWORD', 'guest')))

    translator = Translator(**config['parameters'])
    worker = MQConsumer(translator=translator,
                        connection_parameters=MQ_PARAMETERS,
                        exchange_name=exchange_name,
                        routing_keys=ROUTING_KEYS)

    worker.start()
