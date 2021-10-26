# Translation Worker

A component that runs a machine translation engine to process incoming translation requests.

TODO: model description & reference to training code.

## Setup

The worker can be used by running the prebuilt [docker image](ghcr.io/project-mtee/translation-worker). The `latest` 
tag contains only the code, images with included models have a suffix with the pattern `<language-pair>.<domain>`, 
for example `ru-et.general`, or `multilingual.<domain>` for multilingual models.

The container is designed to run in a CPU environment. For a manual setup, please refer to the included Dockerfile and
the Conda environment specification described in `config/environment.yml`.

The worker depends on the following components:
- [RabbitMQ message broker](https://www.rabbitmq.com/)

The following environment variables should be specified when running the container:
- `MQ_USERNAME` - RabbitMQ username
- `MQ_PASSWORD` - RabbitMQ user password
- `MQ_HOST` - RabbitMQ host
- `MQ_PORT` (optional) - RabbitMQ port (`5672` by default)

### Performance and Hardware Requirements

TODO

### Request Format

The worker consumes translation requests from a RabbitMQ message broker and responds with the translated text. 
The following format is compatible with the [text translation API](ghcr.io/project-mtee/text-translation-api).

Requests should be published with the following parameters:
- Exchange name: `translation` (exchange type is `direct`)
- Routing key: `translation.<src>.<tgt>.<domain>.<input_type>` where `<src>` refers to 2-letter ISO language code of 
  the input text, `<tgt>` is the 2-letter code of the target language, `<domain>` is the text domain and 
  `<input_type>` refers to the origin of the text and its format. For example `translation.et.en.legal.web`.
- Message properties:
    - Correlation ID - a UID for each request that can be used to correlate requests and responses.
    - Reply To - name of the callback queue where the response should be posted.
    - Content Type - `application/json`
    - Headers:
        - `RequestId`
        - `ReturnMessageType`
- JSON-formatted message content with the following keys:
    - `text` – input text, either a string or a list of strings which are allowed to contain multiple sentences or
      paragraphs.
    - `src` – 2-letter ISO language code
    - `tgt` – 2-letter ISO language code
    - `domain` – the text domain, either `general`, `legal`, `military`, `crisis`.
    - `input_type` – input type category that refers to the origin format, either `plain`, `document`, `web` or `asr`

The worker will return a response with the following parameters:
- Exchange name: (empty string)
- Routing key: the Reply To property value from the request
- Message properties:
    - Correlation ID - the Correlation ID value of the request
    - Content Type - `application/json`
    - Headers:
        - `RequestId` - the `RequestId` value of the request
        - `MT-MessageType` - the `ReturnMessageType` value of the request
- JSON-formatted message content with the following keys:
    - `status` - a human-readable status message, `OK` by default
    - `status_code` – (integer) a HTTP status code, `200` by default
    - `translation` - string or a list of strings (depending on the input text format) with the translation. May be 
      `null` in case `status_code!=200`

Known non-OK responses can occur in case the request format was incorrect. Example request and response:
```
{
    "text": 1,
    "src": "et",
    "tgt": "en",
    "domain": "general",
    "input_type": "plain"
}
```
```
{
    "status": "Error parsing input: {'text': ['Invalid value.']}",
    "status_code": 400,
    "translation": null
}
```
The JSON-formatted part of the `status` field is the
[ValidationError](https://marshmallow.readthedocs.io/en/stable/_modules/marshmallow/exceptions.html) message from 
Marshmallow Schema validation.