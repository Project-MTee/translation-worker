# Translation models

Models can be attached to the main [translation-worker](https://github.com/project-mtee/translation-worker) container by
mounting a volume at `/app/models/`. Official translation models can be downloaded from the 
[releases](https://github.com/project-mtee/translation-worker/releases) section of this repository. Due to GitHub's 
file size limitations, these may be uploaded as multipart zip files which have to unpacked first.

Alternatively, models are built into the [`translation-model`](https://ghcr.io/project-mtee/translation-model) images 
published alongside this repository. These are `busybox` images that simply contain all model files in the 
`/app/models/` directory. They can be used as init containers to populate the `/app/models/` volume of the 
[`translation-worker`](https://ghcr.io/project-mtee/translation-worker) instance. 

Each model is published as a separate image and corresponds to a specific release. Compatibility between 
[`translation-worker`](https://ghcr.io/project-mtee/translation-worker) and 
[`translation-model`](https://ghcr.io/project-mtee/translation-model) versions will be specified in the release notes.

## Model configuration

By default, the `translation-worker` looks for a `config.yaml` file on the `/app/models` volume (the `models/` directory
of the repository). This file should contain the following keys:

- `modular` - `True`or `False` depending on whether the model is a modular multilingual model
- `language_pairs` - a list of hyphen-separated language pairs
- `domains` - a list of supported domains
- `checkpoint` - path of the model checkpoint file (usually named `checkpoint_best.pt`)
- `dict_dir` - the directory path that contains the model dictionary files (name pattern: `dict.{lang}.txt`)
- `sentencepiece_dir` - the directory that contains sentencepiece models
- `sentencepiece_prefix` - the prefix used on all sentencepiece model files

All file and directory paths must relative to the root directory of this repository (for example 
`models/checkpoint_best.pt`). More info on where to find the correct files is documented with our 
[model training workflow](https://github.com/Project-MTee/model_training).

The included Dockerfile can be used to publish new model versions.

### Configuration samples

Sample configuration for a general domain Estonian-English single direction model:

```
modular: False
language_pairs:
  - et-en
domains:
  - general
checkpoint: models/checkpoint_best.pt
dict_dir: models/dicts/
sentencepiece_dir: models/sentencepiece/
sentencepiece_prefix: sp-model
```

The configuration above matches the following folder structure:

```
models/
├── checkpoint_best.pt
├── config.yaml
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
modular: True
language_pairs:
  - de-et
  - en-et
  - ru-et
  - et-de
  - et-en
  - et-ru
domains:
  - general
checkpoint: models/checkpoint_best.pt
dict_dir: models/dicts/
sentencepiece_dir: models/sentencepiece/
sentencepiece_prefix: sp-model
```

And a matching folder structure:

```
models/
├── checkpoint_best.pt
├── config.yaml
├── dicts
│   ├── dict.de.txt
│   ├── dict.en.txt
│   ├── dict.et.txt
│   └── dict.ru.txt
└── sentencepiece
    ├── sp-model.de.model
    ├── sp-model.de.vocab
    ├── sp-model.en.model
    ├── sp-model.en.vocab
    ├── sp-model.et.model
    ├── sp-model.et.vocab
    ├── sp-model.ru.model
    └── sp-model.ru.vocab
```