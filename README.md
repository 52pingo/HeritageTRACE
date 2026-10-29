# HeritageTRACE

A configurable research workflow connecting online discourse and documentary or material evidence for heritage research.

HeritageTRACE separates source collection, question-defined decisions, value-oriented analysis and object-level interpretation. The implementation supports a local Laya decision engine; conventional classifiers can be used as alternatives. This repository provides software and domain definitions, not the unpublished manuscript or private research data.

## Quick start

Python 3.10 or newer is recommended (the publication smoke tests use Python 3.12).

```sh
python -m venv .venv
# Activate .venv using your operating system's activation command.
python -m pip install -r requirements.txt
python decide.py --questions domains/third_line.json --validate-only
python collector/main.py --help
python collector/main.py --platforms wikipedia --keywords heritage --mode quick --mock --non-interactive --export none --run-id offline-demo
```

Mock mode uses deterministic artificial content and makes no platform requests. It exercises collection, screening and output management. Live sources have different access requirements; use authorised credentials in collector/.env and respect platform conditions. Core sources work without the optional text/OCR/model extras in collector/requirements.txt.

## Question-defined decisions

```sh
python -m pip install -r requirements-model.txt
python decide.py --questions domains/third_line.json --input examples/records.jsonl --output scored.jsonl --model /path/to/checkpoint --subfolder multilingual
```

The input contract is a JSONL record with title and summary. Decisions retain named option probabilities and predicted classes. Model execution needs suitable Laya assets and hardware; weights are not bundled. The model-loading interface follows the project's Laya 0.4.11 implementation. New-domain definitions can be entered through domains/new_domain_template.json. Validation-only checks the structure of questions, not their predictive accuracy. Additional domain examples, adaptation and held-out evaluation may be needed; no zero-training accuracy guarantee is made.

## Training

training/train_ddp.py provides parameterised single/multi-GPU adaptation based on the Laya training implementation. training/train_grouped.py records the state-disjoint fold-training procedure and expects its documented fold assets. These scripts require model dependencies and permitted local labelled data. They do not download or redistribute the private reference set.

## Reproducibility and scope

Public checks cover Python parsing, question configuration, collector startup and offline mock collection. GPU training, newly packaged model inference and real platform collection are not repeated in the release smoke tests. The recorded Third-Line study used manually labelled project reference data. Configuration reuse is distinct from demonstrated cross-domain accuracy or consumer-device usability.

No passwords, cookies, browser profiles, raw posts, field imagery or model weights belong in this repository. A fictional format example is supplied under examples/.

## Licence and attribution

Project code is distributed under Apache-2.0; see LICENSE and NOTICE. Training code adapts Laya routines and retains its upstream attribution. Laya and underlying encoders have their own licences and terms; verify the exact model revision before redistribution. The project licence does not grant rights to third-party platform content or private research data.

## Value analysis

`python analysis/measure.py --input scored.jsonl --rules analysis/measurement_rules.json --output profile.json`

The bundled rules define the Third-Line application lexicons, acquisition-term removal and dimension mapping. The calculation expects the tier/dimension schema from domains/third_line.json; adapt both definitions and rules for a different domain. Lexicon coverage and probability allocation use distinct denominators.

## Offline checks

`python -m unittest discover -s tests -v`

The checks include fictional mock records and live-adapter construction without network access. They do not establish predictive accuracy or conservation effects.
