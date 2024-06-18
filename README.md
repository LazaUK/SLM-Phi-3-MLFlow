# Building a wrapper and using Phi-3 as an MLFlow model

MLflow is an open-source platform designed to streamline the entire machine learning (ML) lifecycle. It helps data scientists track experiments, manage their ML models and deploy them into production, ensuring reproducibility and efficient collaboration.

In this repo, I'll show 2 different approaches on building a wrapper around Phi-3 small language model (SLM), and then running it as an MLFlow model either locally or in a cloud, e.g. in Azure Machine Learning workspace.

## Table of contents:
- [Option 1: Transformer pipeline](https://github.com/LazaUK/SLM-Phi-3-MLFlow#option-1-transformer-pipeline)
- [Option 2: Custom Python wrapper]()
- [MLFlow model's signatures]()
- [Inference of Phi-3 on MLFlow runtime]()

## Option 1: Transformer pipeline
This is the easiest option, if you want to use HuggingFace model with MLFlow's _experimental_ **transformers** flavour.
1. You would require relevant Python packages from MLFlow and HuggingFace.
``` Python
import mlflow
import transformers
```
2. You can initiate then a transformer pipeline, by referring to a target Phi-3 model on HuggingFace. As can be verified on the _Phi-3-mini-4k-instruct_'s model card, its task is of "Text Generation" type.
``` Python
pipeline = transformers.pipeline(
    task = "text-generation",
    model = "microsoft/Phi-3-mini-4k-instruct"
)
```
3. 

## Option 2: Custom Python wrapper

## MLFlow model's signatures

## Inference of Phi-3 on MLFlow runtime
