import io
from typing import List
import random
import json
import time
import requests

import ray
import torch
import transformers
from transformers import AutoTokenizer
import numpy as np
from instill.configuration import CORE_RAY_ADDRESS
from instill.helpers.ray_helper import (
    InstillRayModelConfig,
    DataType,
    serialize_byte_tensor,
    deserialize_bytes_tensor,
    entry,
)

from ray_pb2 import (
    ModelReadyRequest,
    ModelReadyResponse,
    ModelMetadataRequest,
    ModelMetadataResponse,
    ModelInferRequest,
    ModelInferResponse,
    InferTensor,
)

ray.init(address=CORE_RAY_ADDRESS)
# this import must come after `ray.init()`
from ray import serve


@serve.deployment()
class Llama2:
    def __init__(self, model_path: str):
        self.application_name = "_".join(model_path.split("/")[3:5])
        self.deployement_name = model_path.split("/")[4]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float32,  # Needs 26G Memory
            device="cpu",
        )

    def ModelMetadata(self, req: ModelMetadataRequest) -> ModelMetadataResponse:
        resp = ModelMetadataResponse(
            name=req.name,
            versions=req.version,
            framework="python",
            inputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="prompt_image",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="max_new_tokens",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="stop_words",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="temperature",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="top_k",
                    datatype=str(DataType.TYPE_INT32.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="random_seed",
                    datatype=str(DataType.TYPE_UINT64.name),
                    shape=[1],
                ),
                ModelMetadataResponse.TensorMetadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                ModelMetadataResponse.TensorMetadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                ),
            ],
        )
        return resp

    def ModelReady(self, req: ModelReadyRequest) -> ModelReadyResponse:
        resp = ModelReadyResponse(ready=True)
        return resp

    async def ModelInfer(self, request: ModelInferRequest) -> ModelInferResponse:
        resp = ModelInferResponse(
            model_name=request.model_name,
            model_version=request.model_version,
            outputs=[],
            raw_output_contents=[],
        )

        for i, b_input_tensor in zip(request.inputs, request.raw_input_contents):
            input_name = i.name
            input_shape = i.shape
            input_datatype = i.datatype
            input_tensor = deserialize_bytes_tensor(b_input_tensor)

            prompt = ""
            if input_name == "prompt":
                prompt = str(input_tensor[0][0].decode("utf-8"))
                print(f"[DEBUG] input `prompt` type({type(prompt)}): {prompt}")

            max_new_tokens = 100
            if input_name == "max_new_tokens":
                max_new_tokens = int(input_tensor[0])
                print(
                    f"[DEBUG] input `max_new_tokens` type({type(max_new_tokens)}): {max_new_tokens}"
                )

            top_k = 1
            if input_name == "top_k":
                top_k = int(input_tensor[0])
                print(f"[DEBUG] input `top_k` type({type(top_k)}): {top_k}")

            temperature = 0.8
            if input_name == "temperature":
                temperature = float(input_tensor[0])
                print(
                    f"[DEBUG] input `temperature` type({type(temperature)}): {temperature}"
                )

            random_seed = 0
            if input_name == "random_seed":
                random_seed = int(input_tensor[0])
                print(
                    f"[DEBUG] input `random_seed` type({type(random_seed)}): {random_seed}"
                )
                if random_seed > 0:
                    random.seed(random_seed)
                    np.random.seed(random_seed)
                    torch.manual_seed(random_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(random_seed)

            stop_words = ""
            if input_name == "stop_words":
                stop_words = input_tensor[0]
                print(
                    f"[DEBUG] input `stop_words` type({type(stop_words)}): {stop_words}"
                )
                if len(stop_words) == 0:
                    stop_words = None
                elif stop_words.shape[0] > 1:
                    # TODO: Check wether shoule we decode this words
                    stop_words = list(stop_words)
                else:
                    stop_words = [str(stop_words[0])]
                print(
                    f"[DEBUG] parsed input `stop_words` type({type(stop_words)}): {stop_words}"
                )

            extra_params = {}
            if input_name == "extra_params":
                extra_params_str = str(input_tensor[0][0].decode("utf-8"))
                print(
                    f"[DEBUG] input `extra_params` type({type(extra_params_str)}): {extra_params_str}"
                )

                try:
                    extra_params = json.loads(extra_params_str)
                except json.decoder.JSONDecodeError:
                    print("[DEBUG] WARNING `extra_params` parsing faield!")
                    continue

        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            **extra_params,
        )

        text_outputs = [seq["generated_text"].encode("utf-8") for seq in sequences]

        resp.outputs.append(
            InferTensor(
                name="text",
                shape=[1, len(sequences)],
                datatype=str(DataType.TYPE_STRING),
            )
        )

        resp.raw_output_contents.append(serialize_byte_tensor(np.asarray(text_outputs)))

        return resp


def deploy_model(model_config: InstillRayModelConfig):
    c_app = Llama2.options(
        name=model_config.application_name,
        ray_actor_options=model_config.ray_actor_options,
        max_concurrent_queries=model_config.max_concurrent_queries,
        autoscaling_config=model_config.ray_autoscaling_options,
    ).bind(model_config.model_path)

    serve.run(
        c_app, name=model_config.model_name, route_prefix=model_config.route_prefix
    )


def undeploy_model(model_name: str):
    serve.delete(model_name)


if __name__ == "__main__":
    func, model_config = entry("Llama-2-7b-hf/")

    if func == "deploy":
        deploy_model(model_config=model_config)
    elif func == "undeploy":
        undeploy_model(model_name=model_config.model_name)
