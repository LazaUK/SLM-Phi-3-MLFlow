# Building a wrapper and using Phi-3 as an MLFlow model

MLflow is an open-source platform designed to streamline the entire machine learning (ML) lifecycle. It helps data scientists track experiments, manage their ML models and deploy them into production, ensuring reproducibility and efficient collaboration.

In this repo, I’ll demonstrate two different approaches to building a wrapper around Phi-3 small language model (SLM) and then running it as an MLFlow model either locally or in the cloud, e.g., in Azure Machine Learning workspace. You can use attached Jupyter notebooks to jump-start your development process.

## Table of contents:
- [Option 1: Transformer pipeline](https://github.com/LazaUK/SLM-Phi-3-MLFlow/tree/main#option-1-transformer-pipeline)
- [Option 2: Custom Python wrapper](https://github.com/LazaUK/SLM-Phi-3-MLFlow/tree/main#option-2-custom-python-wrapper)
- [Signatures of generated MLFlow models](https://github.com/LazaUK/SLM-Phi-3-MLFlow/tree/main#signatures-of-generated-mlflow-models)
- [Inference of Phi-3 with MLFlow runtime](https://github.com/LazaUK/SLM-Phi-3-MLFlow/tree/main#inference-of-phi-3-with-mlflow-runtime)

## Option 1: Transformer pipeline
This is the easiest option to build a wrapper if you want to use a HuggingFace model with MLFlow’s _experimental_ **transformers** flavour.
1. You would require relevant Python packages from MLFlow and HuggingFace:
``` Python
import mlflow
import transformers
```
2. Next, you should initiate a transformer pipeline by referring to the target Phi-3 model in the HuggingFace registry. As can be seen from the _Phi-3-mini-4k-instruct_’s model card, its task is of a “Text Generation” type:
``` Python
pipeline = transformers.pipeline(
    task = "text-generation",
    model = "microsoft/Phi-3-mini-4k-instruct"
)
```
3. You can now save your Phi-3 model’s transformer pipeline into MLFlow format and provide additional details such as the target artifacts path, specific model configuration settings and inference API type:
``` Python
model_info = mlflow.transformers.log_model(
    transformers_model = pipeline,
    artifact_path = "phi3-mlflow-model",
    model_config = model_config,
    task = "llm/v1/chat"
)
```

## Option 2: Custom Python wrapper
At the time of writing, the transformer pipeline did not support MLFlow wrapper generation for HuggingFace models in ONNX format, even with the experimental _optimum_ Python package. For the cases like this, you can build your custom Python wrapper for MLFlow model.
1. I'll utilise here Microsoft's [ONNX Runtime generate() API](https://github.com/microsoft/onnxruntime-genai) for the ONNX model's inference and tokens encoding / decoding. You have to choose _onnxruntime_genai_ package for your target compute, with the below example targeting CPU:
``` Python
import mlflow
from mlflow.models import infer_signature
import onnxruntime_genai as og
```
2. Our custom class implements two methods: _load_context()_ to initialise the **ONNX model** of Phi-3 Mini 4K Instruct, **generator parameters** and **tokenizer**; and _predict()_ to generate output tokens for the provided prompt:
``` Python
class Phi3Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Retrieving model from the artifacts
        model_path = context.artifacts["phi3-mini-onnx"]
        model_options = {
             "max_length": 300,
             "temperature": 0.2,         
        }
    
        # Defining the model
        self.phi3_model = og.Model(model_path)
        self.params = og.GeneratorParams(self.phi3_model)
        self.params.set_search_options(**model_options)
        
        # Defining the tokenizer
        self.tokenizer = og.Tokenizer(self.phi3_model)

    def predict(self, context, model_input):
        # Retrieving prompt from the input
        prompt = model_input["prompt"][0]
        self.params.input_ids = self.tokenizer.encode(prompt)

        # Generating the model's response
        response = self.phi3_model.generate(self.params)

        return self.tokenizer.decode(response[0][len(self.params.input_ids):])
```
3. The last step is to generate a custom Python wrapper (in pickle format) for the Phi-3 model, along with the original ONNX model and required dependencies:
``` Python
model_info = mlflow.pyfunc.log_model(
    artifact_path = artifact_path,
    python_model = Phi3Model(),
    artifacts = {
        "phi3-mini-onnx": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
    },
    input_example = input_example,
    signature = infer_signature(input_example, ["Run"]),
    extra_pip_requirements = ["torch", "onnxruntime_genai", "numpy"],
)
```

## Signatures of generated MLFlow models
1. In Step 3 of Option 1 above, we set the MLFlow model’s task to “_llm/v1/chat_”. Such instruction generates a model’s API wrapper, compatible with OpenAI’s Chat API as shown below:
``` Python
{inputs: 
  ['messages': Array({content: string (required), name: string (optional), role: string (required)}) (required), 'temperature': double (optional), 'max_tokens': long (optional), 'stop': Array(string) (optional), 'n': long (optional), 'stream': boolean (optional)],
outputs: 
  ['id': string (required), 'object': string (required), 'created': long (required), 'model': string (required), 'choices': Array({finish_reason: string (required), index: long (required), message: {content: string (required), name: string (optional), role: string (required)} (required)}) (required), 'usage': {completion_tokens: long (required), prompt_tokens: long (required), total_tokens: long (required)} (required)],
params: 
  None}
```
2. As a result, you can submit your prompt in the following format:
``` Python
messages = [{"role": "user", "content": "What is the capital of Spain?"}]
```
3. Then, use OpenAI API-compatible post-processing, e.g., _response[0][‘choices’][0][‘message’][‘content’]_, to beautify your output to something like this:
``` JSON
Question: What is the capital of Spain?

Answer: The capital of Spain is Madrid. It is the largest city in Spain and serves as the political, economic, and cultural center of the country. Madrid is located in the center of the Iberian Peninsula and is known for its rich history, art, and architecture, including the Royal Palace, the Prado Museum, and the Plaza Mayor.

Usage: {'prompt_tokens': 11, 'completion_tokens': 73, 'total_tokens': 84}
```
4.  In Step 3 of Option 2 above, we allow the MLFlow package to generate the model’s signature from a given input example. Our MLFlow wrapper's signature will look like this:
``` Python
{inputs: 
  ['prompt': string (required)],
outputs: 
  [string (required)],
params: 
  None}
```
5. So, our prompt would need to contain "prompt" dictionary key, similar to this:
``` Python
{"prompt": "<|system|>You are a stand-up comedian.<|end|><|user|>Tell me a joke about atom<|end|><|assistant|>",}
```
6. The model's output will be provided then in string format:
``` JSON
Alright, here's a little atom-related joke for you!

Why don't electrons ever play hide and seek with protons?

Because good luck finding them when they're always "sharing" their electrons!

Remember, this is all in good fun, and we're just having a little atomic-level humor!
```

## Inference of Phi-3 with MLFlow runtime
1. To run the generated MLFlow model locally, you can load it with _mlflow.pyfunc.load_model()_ from the model’s directory and then call its _predict()_ method. You can load the model as follows:
``` Python
loaded_model = mlflow.pyfunc.load_model(
    model_uri = model_info.model_uri
)
```
2. To run in a cloud environment like an Azure Machine Learning workspace, you can register your MLFlow model with a custom Python wrapper in workspace's model registry:
![phi3_mlflow_registration](/images/phi3_aml_registry.png)
3. Then, deploy it to a managed real-time endpoint:
![phi3_mlflow_deploy](/images/phi3_aml_deploy.png)
4. Once the deployment succeeds, you can immediately start using it with code samples provided in **JavaScript**, **Python**, **C#** or **R**:
![phi3_mlflow_endpoint](/images/phi3_aml_endpoint.png)
