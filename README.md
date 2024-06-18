# Building a wrapper and using Phi-3 as an MLFlow model

MLflow is an open-source platform designed to streamline the entire machine learning (ML) lifecycle. It helps data scientists track experiments, manage their ML models and deploy them into production, ensuring reproducibility and efficient collaboration.

In this repo, I'll demonstrate 2 different approaches to building a wrapper around Phi-3 small language model (SLM) and then running it as an MLFlow model either locally or in a cloud, e.g. in Azure Machine Learning workspace. You can use attached Jupyter notebooks to jump-start your development process.

## Table of contents:
- [Option 1: Transformer pipeline](https://github.com/LazaUK/SLM-Phi-3-MLFlow/tree/main#option-1-transformer-pipeline)
- [Option 2: Custom Python wrapper](https://github.com/LazaUK/SLM-Phi-3-MLFlow/tree/main#option-2-custom-python-wrapper)
- [Signatures of generated MLFlow models](https://github.com/LazaUK/SLM-Phi-3-MLFlow/tree/main#signatures-of-generated-mlflow-models)
- [Inference of Phi-3 with MLFlow runtime](https://github.com/LazaUK/SLM-Phi-3-MLFlow/tree/main#inference-of-phi-3-with-mlflow-runtime)

## Option 1: Transformer pipeline
This is the easiest option to build a wrapper, if you want to use a HuggingFace model with the MLFlow's _experimental_ **transformers** flavour.
1. You would require relevant Python packages from MLFlow and HuggingFace.
``` Python
import mlflow
import transformers
```
2. Next, you should initiate a transformer pipeline, by referring target Phi-3 model in the HuggingFace registry. As can be seen from the _Phi-3-mini-4k-instruct_'s model card, its task is of a "Text Generation" type.
``` Python
pipeline = transformers.pipeline(
    task = "text-generation",
    model = "microsoft/Phi-3-mini-4k-instruct"
)
```
3. You can now save your Phi-3 model's transformer pipeline into MLFlow format, and provide additional details such as the target artifacts path, model configuration settings and inference API type.
``` Python
model_info = mlflow.transformers.log_model(
    transformers_model = pipeline,
    artifact_path = "phi3-mlflow-model",
    model_config = model_config,
    task = "llm/v1/chat"
)
```

## Option 2: Custom Python wrapper
1. At the time of writing, transformer pipeline didn't support MLFlow wrapper generation for the HuggingFace models in ONNX format, even with the experimental _optimum_ Python package. As a workaround, you can build your custom Python wrapper for MLFlow model.
2. To build a wrapper, I'l utilise Microsoft's [ONNX Runtime generate() API](https://github.com/microsoft/onnxruntime-genai) for the original ONNX model's inference, and input / output tokens encoding / decoding. You would require onnxruntime_genai Python package for your target compute along with the MLFlow Python packages.
``` Python
import mlflow
from mlflow.models import infer_signature
import onnxruntime_genai as og
```
3. 

## Signatures of generated MLFlow models
1. In the Step 3 of the Option 1 above, we have set the MLFlow's model task to "_llm/v1/chat_". Such instruction generates model's API wrapper, compatible with OpenAI's Chat API as shown below. 
``` Python
{inputs: 
  ['messages': Array({content: string (required), name: string (optional), role: string (required)}) (required), 'temperature': double (optional), 'max_tokens': long (optional), 'stop': Array(string) (optional), 'n': long (optional), 'stream': boolean (optional)],
outputs: 
  ['id': string (required), 'object': string (required), 'created': long (required), 'model': string (required), 'choices': Array({finish_reason: string (required), index: long (required), message: {content: string (required), name: string (optional), role: string (required)} (required)}) (required), 'usage': {completion_tokens: long (required), prompt_tokens: long (required), total_tokens: long (required)} (required)],
params: 
  None}
```
2. As a result, you can submit your prompt in the following format.
``` Python
messages = [{"role": "user", "content": "What is the capital of Spain?"}]
```
3. Then use OpenAI API-compatible post-processing, e.g. _response[0]['choices'][0]['message']['content']_ to beautify your output to something like this.
``` JSON
Question: What is the capital of Spain?

Answer: The capital of Spain is Madrid. It is the largest city in Spain and serves as the political, economic, and cultural center of the country. Madrid is located in the center of the Iberian Peninsula and is known for its rich history, art, and architecture, including the Royal Palace, the Prado Museum, and the Plaza Mayor.

Usage: {'prompt_tokens': 11, 'completion_tokens': 73, 'total_tokens': 84}
```

## Inference of Phi-3 with MLFlow runtime
