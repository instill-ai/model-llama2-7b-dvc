import random

import ray
import torch
import transformers
from transformers import AutoTokenizer
import numpy as np

from instill.helpers.const import DataType, TextGenerationInput
from instill.helpers.ray_io import StandardTaskIO
from instill.helpers.ray_config import (
    instill_deployment,
    get_compose_ray_address,
    InstillDeployable,
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

ray.init(address=get_compose_ray_address(10001))
# this import must come after `ray.init()`
from ray import serve


@instill_deployment
class Llama2:
    def __init__(self, model_path: str):
        # self.application_name = "_".join(model_path.split("/")[3:5])
        # self.deployement_name = model_path.split("/")[4]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float16,  # Needs 26G Memory
            device="cuda",
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
                    datatype=str(DataType.TYPE_UINT32.name),
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

        task_text_generation_input: TextGenerationInput = (
            StandardTaskIO.parse_task_text_generation_input(request=request)
        )

        if task_text_generation_input.random_seed > 0:
            random.seed(task_text_generation_input.random_seed)
            np.random.seed(task_text_generation_input.random_seed)
            torch.manual_seed(task_text_generation_input.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(task_text_generation_input.random_seed)

        sequences = self.pipeline(
            task_text_generation_input.prompt,
            do_sample=True,
            top_k=task_text_generation_input.top_k,
            temperature=task_text_generation_input.temperature,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=task_text_generation_input.max_new_tokens,
            **task_text_generation_input.extra_params,
        )

        task_text_generation_output = StandardTaskIO.parse_task_text_generation_output(
            sequences=sequences
        )

        resp.outputs.append(
            InferTensor(
                name="text",
                shape=[1, len(sequences)],
                datatype=str(DataType.TYPE_STRING),
            )
        )

        resp.raw_output_contents.append(task_text_generation_output)

        return resp


# def deploy_model(model_config: InstillRayModelConfig):
#     c_app = Llama2.options(
#         name=model_config.application_name,
#         ray_actor_options=model_config.ray_actor_options,
#         max_concurrent_queries=model_config.max_concurrent_queries,
#         autoscaling_config=model_config.ray_autoscaling_options,
#     ).bind(model_config.model_path)

#     serve.run(
#         c_app, name=model_config.model_name, route_prefix=model_config.route_prefix
#     )


# def undeploy_model(model_name: str):
#     serve.delete(model_name)


# if __name__ == "__main__":
#     func, model_config = entry("Llama-2-7b-hf/")

#     model_config.ray_actor_options["num_cpus"] = 6  # TOOD: Check wether this needed in GPU mode
#     model_config.ray_actor_options["num_gpus"] = 1  # Specify the number of GPUs

#     if func == "deploy":
#         deploy_model(model_config=model_config)
#     elif func == "undeploy":
#         undeploy_model(model_name=model_config.model_name)

deployable = InstillDeployable(Llama2, model_weight_or_folder_name="Llama-2-7b-hf")

# you can also have a fine-grained control of the cpu and gpu resources allocation
deployable.update_num_cpus(4)
deployable.update_num_gpus(1)
