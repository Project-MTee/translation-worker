# Translation models

Models are included in some [translation-worker](https://github.com/project-mtee/translation-worker) images, or they can
be attached to the base image by mounting a volume at `/app/models/`. If a model is not found, the worker will try to
download it from HuggingFace upon startup. Official translation models can be found
in [HuggingFace](https://huggingface.co/models?other=MTee&modularNMT) with `MTee` and `ModularNMT` tags.

## Model configuration

By default, the `translation-worker` looks for a `config.yaml` file on the `/app/models` volume (the `models/` directory
of the repository). This file should contain the following keys:

- `language_pairs` - a list of hyphen-separated language pairs (using 2-letter ISO language codes)
- `domains` - a list of supported domains
- `huggingface` (optional) - a HuggingFace model ID. This is used during model build to download the model or upon
  startup if a model does not exist.
- `model_root` - a path where the model is stored and loaded from. This path should be absolute or relative to the root
  directory of this repository. All paths described below are relative to this path.
- `modular` - `True`or `False` depending on whether the model is a modular multilingual model
- `checkpoint` - name of the model checkpoint file (usually named `checkpoint_best.pt` (default) or `modular_model.pt`),
  relative to `model_root`
- `dict_dir` (optional) - the directory path that contains the model dictionary files (name pattern: `dict.{lang}.txt`),
  by default, the worker assumes that `model_root` is used.
- `sentencepiece_dir` - the directory that contains sentencepiece models, by default, the worker assumes
  that `model_root` is used.
- `sentencepiece_prefix` - the prefix used on all sentencepiece model files, `sp-model` by default.

More info on where to find the correct files is documented with our
[model training workflow](https://github.com/Project-MTee/model_training).

### Configuration samples

Sample configuration for a general domain Estonian-English single direction model:

```
language_pairs:
  - et-en
domains:
  - general

huggingface: 
model_root: models/et-en-general
modular: False

checkpoint: checkpoint_best.pt
dict_dir: dicts/
sentencepiece_dir: sentencepiece/
sentencepiece_prefix: sp-model
```

The configuration above matches the following folder structure:

```
models/et-en-general/
├── checkpoint_best.pt
├── dicts
│   ├── dict.en.txt
│   └── dict.et.txt
└── sentencepiece
    ├── sp-model.model
    └── sp-model.vocab
```

---

Sample configuration for a general modular Estonian-centric model:

```
language_pairs:
  - de-et
  - en-et
  - ru-et
  - et-de
  - et-en
  - et-ru
domains:
  - general

huggingface: tartuNLP/mtee-general
model_root: models/tartuNLP/mtee-general
modular: True

checkpoint: modular_model.pt
sentencepiece_prefix: sp-model
```

And a matching folder structure:

```
models/tartuNLP/mtee-general/
├── modular_model.pt
├── dict.de.txt
├── dict.en.txt
├── dict.et.txt
├── dict.ru.txt
├── sp-model.de.model
├── sp-model.de.vocab
├── sp-model.en.model
├── sp-model.en.vocab
├── sp-model.et.model
├── sp-model.et.vocab
├── sp-model.ru.model
└── sp-model.ru.vocab
```