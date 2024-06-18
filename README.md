# Building a wrapper and using Phi-3 as an MLFlow model

MLflow is an open-source platform designed to streamline the entire machine learning (ML) lifecycle. It helps data scientists track experiments, manage their ML models and deploy them into production, ensuring reproducibility and efficient collaboration.

In this repo, I'll demonstrate 2 different approaches to building a wrapper around Phi-3 small language model (SLM), and then running it as an MLFlow model either locally or in a cloud, e.g. in Azure Machine Learning workspace.

## Table of contents:
- [Option 1: Transformer pipeline](https://github.com/LazaUK/SLM-Phi-3-MLFlow#option-1-transformer-pipeline)
- [Option 2: Custom Python wrapper](https://github.com/LazaUK/SLM-Phi-3-MLFlow#option-2-custom-python-wrapper)
- [Signatures of generated MLFlow models]()
- [Inference of Phi-3 with MLFlow runtime]()

## Option 1: Transformer pipeline
This is the easiest option, if you want to use HuggingFace model with MLFlow's _experimental_ **transformers** flavour.
1. You would require relevant Python packages from MLFlow and HuggingFace.
``` Python
import mlflow
import transformers
```
2. You can initiate then a transformer pipeline, by referring to a target Phi-3 model in the HuggingFace registry. As can be verified from the _Phi-3-mini-4k-instruct_'s model card, its task is of a "Text Generation" type.
``` Python
pipeline = transformers.pipeline(
    task = "text-generation",
    model = "microsoft/Phi-3-mini-4k-instruct"
)
```
3. You can now save your Phi-3 model from a transformer pipeline into MLFlow format, and provide additional details such as the target artifacts path, model configuration settings and inference API type.
``` Python
model_info = mlflow.transformers.log_model(
    transformers_model = pipeline,
    artifact_path = "phi3-mlflow-model",
    model_config = model_config,
    task = "llm/v1/chat"
)
```

## Option 2: Custom Python wrapper

## Signatures of generated MLFlow models
1. In the Step 3 of the Option 1 above, we have set the MLFlow's model task to "_llm/v1/chat_". This instruction generates an API wrapper, compatible with OpenAI's Chat API.
2. 
``` JSON
{
inputs: 
  ['messages': Array({content: string (required), name: string (optional), role: string (required)}) (required), 'temperature': double (optional), 'max_tokens': long (optional), 'stop': Array(string) (optional), 'n': long (optional), 'stream': boolean (optional)],
outputs: 
  ['id': string (required), 'object': string (required), 'created': long (required), 'model': string (required), 'choices': Array({finish_reason: string (required), index: long (required), message: {content: string (required), name: string (optional), role: string (required)} (required)}) (required), 'usage': {completion_tokens: long (required), prompt_tokens: long (required), total_tokens: long (required)} (required)],
params: 
  None
}
```

## Inference of Phi-3 with MLFlow runtime
