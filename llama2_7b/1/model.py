# pylint: skip-file
import os

# Enable following code for gpu mode only
# TORCH_GPU_DEVICE_ID = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{TORCH_GPU_DEVICE_ID}"


import io
import time
import requests
import random
import base64
import ray
import torch
import transformers
from transformers import AutoTokenizer
from PIL import Image

import numpy as np

from instill.helpers.const import DataType, TextGenerationChatInput
from instill.helpers.ray_io import (
    serialize_byte_tensor,
    deserialize_bytes_tensor,
    StandardTaskIO,
)

from instill.helpers.ray_config import instill_deployment, InstillDeployable
from instill.helpers import (
    construct_infer_response,
    construct_metadata_response,
    Metadata,
)


# from conversation import Conversation, conv_templates, SeparatorStyle

# torch.cuda.set_per_process_memory_fraction(
#     TORCH_GPU_MEMORY_FRACTION, 0  # it count of number of device instead of device index
# )


@instill_deployment
class Llama2:
    def __init__(self, model_path: str):
        self.application_name = "_".join(model_path.split("/")[3:5])
        self.deployement_name = model_path.split("/")[4]
        print(f"application_name: {self.application_name}")
        print(f"deployement_name: {self.deployement_name}")
        print(f"torch version: {torch.__version__}")

        print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count() : {torch.cuda.device_count()}")
        # print(f"torch.cuda.current_device() : {torch.cuda.current_device()}")
        # print(f"torch.cuda.device(0) : {torch.cuda.device(0)}")
        # print(f"torch.cuda.get_device_name(0) : {torch.cuda.get_device_name(0)}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float32,  # Needs 26G Memory
            device="cpu",
        )

    def ModelMetadata(self, req):
        resp = construct_metadata_response(
            req=req,
            inputs=[
                Metadata(
                    name="prompt",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="prompt_images",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="chat_history",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="system_message",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
                Metadata(
                    name="max_new_tokens",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="temperature",
                    datatype=str(DataType.TYPE_FP32.name),
                    shape=[1],
                ),
                Metadata(
                    name="top_k",
                    datatype=str(DataType.TYPE_UINT32.name),
                    shape=[1],
                ),
                Metadata(
                    name="random_seed",
                    datatype=str(DataType.TYPE_UINT64.name),
                    shape=[1],
                ),
                Metadata(
                    name="extra_params",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[1],
                ),
            ],
            outputs=[
                Metadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[-1, -1],
                ),
            ],
        )
        return resp

    async def __call__(self, req):
        task_text_generation_chat_input: TextGenerationChatInput = (
            StandardTaskIO.parse_task_text_generation_chat_input(request=req)
        )
        print("----------------________")
        print(task_text_generation_chat_input)
        print("----------------________")

        print("print(task_text_generation_chat.prompt")
        print(task_text_generation_chat_input.prompt)
        print("-------\n")

        print("print(task_text_generation_chat.prompt_images")
        print(task_text_generation_chat_input.prompt_images)
        print("-------\n")

        print("print(task_text_generation_chat.chat_history")
        print(task_text_generation_chat_input.chat_history)
        print("-------\n")

        print("print(task_text_generation_chat.system_message")
        print(task_text_generation_chat_input.system_message)
        if len(task_text_generation_chat_input.system_message) is not None:
            if len(task_text_generation_chat_input.system_message) == 0:
                task_text_generation_chat_input.system_message = None
        print("-------\n")

        print("print(task_text_generation_chat.max_new_tokens")
        print(task_text_generation_chat_input.max_new_tokens)
        print("-------\n")

        print("print(task_text_generation_chat.temperature")
        print(task_text_generation_chat_input.temperature)
        print("-------\n")

        print("print(task_text_generation_chat.top_k")
        print(task_text_generation_chat_input.top_k)
        print("-------\n")

        print("print(task_text_generation_chat.random_seed")
        print(task_text_generation_chat_input.random_seed)
        print("-------\n")

        print("print(task_text_generation_chat.stop_words")
        print(task_text_generation_chat_input.stop_words)
        print("-------\n")

        print("print(task_text_generation_chat.extra_params")
        print(task_text_generation_chat_input.extra_params)
        print("-------\n")

        if task_text_generation_chat_input.temperature <= 0.0:
            task_text_generation_chat_input.temperature = 0.8

        if task_text_generation_chat_input.random_seed > 0:
            random.seed(task_text_generation_chat_input.random_seed)
            np.random.seed(task_text_generation_chat_input.random_seed)

        # No chat_history Needed
        # No system message Needed

        t0 = time.time()

        sequences = self.pipeline(
            task_text_generation_chat_input.prompt,
            do_sample=True,
            top_k=task_text_generation_chat_input.top_k,
            temperature=task_text_generation_chat_input.temperature,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=task_text_generation_chat_input.max_new_tokens,
            **task_text_generation_chat_input.extra_params,
        )
        print(f"Inference time cost {time.time()-t0}s")

        max_output_len = 0

        text_outputs = []
        for seq in sequences:
            print("Output No Clean ----")
            print(seq["generated_text"])
            # print("Output Clean ----")
            # print(seq["generated_text"][len(task_text_generation_chat_input.prompt) :])
            print("---")
            generated_text = seq["generated_text"].strip().encode("utf-8")
            text_outputs.append(generated_text)
            max_output_len = max(max_output_len, len(generated_text))
        text_outputs_len = len(text_outputs)
        task_output = serialize_byte_tensor(np.asarray(text_outputs))
        # task_output = StandardTaskIO.parse_task_text_generation_output(sequences)

        print("Output:")
        print(task_output)
        print(type(task_output))

        return construct_infer_response(
            req=req,
            outputs=[
                Metadata(
                    name="text",
                    datatype=str(DataType.TYPE_STRING.name),
                    shape=[text_outputs_len, max_output_len],
                )
            ],
            raw_outputs=[task_output],
        )


deployable = InstillDeployable(
    Llama2, model_weight_or_folder_name="Llama-2-7b-hf/", use_gpu=False
)

# # Optional
# deployable.update_max_replicas(2)
# deployable.update_min_replicas(0)
