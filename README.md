# Translation Worker

A component that runs neural machine translation models to process incoming translation requests via RabbitMQ. This
application is based on a custom version of FairSeq - a PyTorch-based library for sequence modeling - and is compatible
with standard single-direction baseline models as well as modular multilingual models.

For more information, please refer to the [model training scripts](https://github.com/Project-MTee/model_training) and
the [custom FairSeq fork](https://github.com/TartuNLP/fairseq/releases/tag/mtee-0.1.0) used in this project.

## Setup

The easiest and recommended way to set up the translation worker, is by using the docker images published alongside this
repository, and it is designed to be used in a CPU environment. Starting from version 1.2.0, we
publish [`translation-worker`](https://ghcr.io/project-mtee/translation-worker) images with and without models. For more
information about configuring custom models can be found
in [`models/README.md`](https://github.com/Project-MTee/translation-worker/tree/main/models)).

- `HF_MODEL` (optional) - a HuggingFace model
  identifyer ([`tartuNLP/mtee-domain-detection`](https://huggingface.co/tartuNLP/mtee-domain-detection) by
  default). The model is automatically downloaded from HuggingFace during the build phase.

The running container should be configured using the following parameters:

- Environment variables:
    - Variables that configure the connection to a [RabbitMQ message broker](https://www.rabbitmq.com/):
        - `MQ_USERNAME` - RabbitMQ username
        - `MQ_PASSWORD` - RabbitMQ user password
        - `MQ_HOST` - RabbitMQ host
        - `MQ_PORT` (optional) - RabbitMQ port (`5672` by default)
        - `MQ_EXCHANGE` (optional) - RabbitMQ exchange name (`translation` by default)
        - `MQ_CONNECTION_NAME` (optional) - friendly connection name (`Translation worker` by default)
        - `MQ_HEARTBEAT` (optional) - heartbeat value (`60` seconds by default)
    - PyTorch-related variables:
        - `MKL_NUM_THREADS` (optional) - number of threads used for intra-op parallelism by PyTorch. This defaults to
          the number of CPU cores which may cause computational overhead when deployed on larger nodes. Alternatively,
          the `docker run` flag `--cpuset-cpus` can be used to control this. For more details, refer to
          the [performance and hardware requirements](#performance-and-hardware-requirements) section below.
    - Translation-related variables:
        - `WORKER_MAX_INPUT_LENGTH` (optional) - the number of characters allowed per request (`10000` by default).
          Longer requests will return validation errors with status code `400`.

- Optional runtime flags (the `COMMAND` option):
    - `--model-config` - path to the model config file (`models/config.yaml` by default). The default file is included
      in images that already include models. Compatible sample files are included in the `models/` directory and the
      format is described in [`models/README.md`](https://github.com/Project-MTee/translation-worker/tree/main/models)).
    - `--log-config` - path to logging config files (`logging/logging.ini` by default), `logging/debug.ini` can be used
      for debug-level logging
    - `--port` - port of the healthcheck probes (`8000` by default):

- Endpoints for healthcheck probes:
    - `/health/startup`
    - `/health/readiness`
    - `/health/liveness`

### Building new images

When building the image, the model can be built with different targets. BuildKit should be enabled to skip any unused
stages of the build.

- `worker-base` - the worker code without any models.
- `worker-model` - a worker with an included model. Requires **one** of the following build-time arguments:
    - `MODEL_IMAGE` - the image name where the model is copied from. For example any of
      the [`translation-model`](https://ghcr.io/project-mtee/translation-model) images.
    - `MODEL_CONFIG_FILE` - path to the model configuration file, for example `models/general.yaml`. The file must
      contain the otherwise optional key `huggingface` to download the model or the build will fail.

- `env` - an intermediate build stage with all packages installed, but no code.
- `model-dl` - images that only contain model files and configuration. The separate stage is used to cache this step and
  speed up builds because HuggingFace downloads can be very slow compared to copying model files from a build stage.
  Published at [`translation-model`](https://ghcr.io/project-mtee/translation-model). Alternatively, these can be used
  as init containers to copy models over during startup, but this is quite slow and not recommended.
- `model` - an alias for the model image, the value of `MODEL_IMAGE` or `model-dl` by default. 

### Performance and hardware requirements

The worker loads the NMT model into memory. The exact RAM usage depends on the model and should always be tested, but a
conservative estimate is to have **8 GB of memory** available (tested with a modular model with 4 input and 4 output
languages).

The performance depends on the available CPU resources, however, this should be finetuned for the deployment
infrastructure. By default, PyTorch will try to utilize all CPU cores to 100% and run as many threads as there are
cores. This can cause major computational overhead if the worker is deployed on large nodes. The **number of threads
used should be limited** using the `MKL_NUM_THREADS` environment variable or the `docker run` flag `--cpuset-cpus`.

Limiting CPU usage by docker configuration which only limits CPU shares is not sufficient (e.g. `docker run` flag
`--cpus` or the CPU limit in K8s, unless the non-default
[static CPU Manager policy](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/) is used). For
example, on a node with 128 cores, setting the CPU limit at `16.0` results in 128 parallel threads running with each one
utilizing only 1/8 of each core's computational potential. This amplifies the effect of multithreading overhead and can
result in inference speeds up to 20x slower than expected.

Although the optimal number of threads depends on the exact model and infrastructure used, a good starting point is
around `16`. With optimal configuration and modern hardware, the worker should be able to process ~7 sentences per
second. For more information, please refer to
[PyTorch documentation](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).

### Manual / development setup

For a manual setup, please refer to the included Dockerfile and the environment specification described in
`requirements/requirements.txt`.
Additionally, [`models/README.md`](https://github.com/project-mtee/translation-worker/tree/main/models) describes how
models should be set up correctly.

To initialize the sentence splitting functionality, the following command should be run before starting the application:

```python -c "import nltk; nltk.download(\"punkt\")"```

RabbitMQ and PyTorch parameters should be configured with environment variables as described above. The worker can be
started with:

```python main.py [--model-config models/config.yaml] [--log-config logging/logging.ini] [--port 8000]```

## Request format

The worker consumes translation requests from a RabbitMQ message broker and responds with the translated text. The
following format is compatible with the [translation API](https://ghcr.io/project-mtee/translation-api-service).

Requests should be published with the following parameters:

- Exchange name: `translation` (exchange type is `direct`)
- Routing key: `translation.<src>.<tgt>.<domain>.<input_type>` where `<src>` refers to 2-letter ISO language code of the
  input text, `<tgt>` is the 2-letter code of the target language, `<domain>` is the text domain and
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
    "src": "et",
    "tgt": "en",
    "domain": "general",
    "input_type": "plain"
}
```

```
{
    "status": "1 validation error for Request\ntext\n  field required (type=value_error.missing)",
    "status_code": 400,
    "translation": null
}
```

The JSON-formatted part of the `status` field is the
[ValidationError](https://pydantic-docs.helpmanual.io/usage/models/#error-handling) message from Pydantic validation.