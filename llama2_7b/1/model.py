# pylint: skip-file
import os
import io
import base64
from json.decoder import JSONDecodeError
from typing import List

from PIL import Image

import random

import time
import json
from pathlib import Path

import traceback

import numpy as np
from typing import Any, Dict, List, Union

from transformers import AutoTokenizer
import transformers
import torch

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TextGenerationInput:
    prompt = ""
    prompt_images: Union[List[np.ndarray], None] = None
    chat_history: Union[List[str], None] = None
    system_message: Union[str, None] = None
    max_new_tokens = 100
    temperature = 0.8
    top_k = 1
    random_seed = 0
    stop_words: Any = ""  # Optional
    extra_params: Dict[str, str] = {}


class TritonPythonModel:
    # Reference: https://docs.nvidia.com/launchpad/data-science/sentiment/latest/sentiment-triton-overview.html
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
        Both keys and values are strings. The dictionary keys and values are:
        * model_config: A JSON string containing the model configuration
        * model_instance_kind: A string containing model instance kind
        * model_instance_device_id: A string containing model instance device ID
        * model_repository: Model repository path
        * model_version: Model version
        * model_name: Model name"""
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])

        # Load the model
        model_path = str(
            Path(__file__).parent.absolute().joinpath("Llama-2-7b-hf/")
        )  # .with_name('llama-2-7b-hf/') # fix invalid error

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float32,  # Needs 26G Memory
            device="cpu",
        )

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            text_generation_input = TextGenerationInput()

            if pb_utils.get_input_tensor_by_name(request, "prompt") is not None:
                text_generation_input.prompt = str(
                    pb_utils.get_input_tensor_by_name(request, "prompt")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
            else:
                raise ValueError("Prompt must be non-empty")

            if pb_utils.get_input_tensor_by_name(request, "prompt_images") is not None:
                input_tensors = pb_utils.get_input_tensor_by_name(
                    request, "prompt_images"
                ).as_numpy()
                images = []
                for enc in input_tensors:
                    if len(enc) == 0:
                        continue
                    try:
                        enc_json = json.loads(str(enc.decode("utf-8")))
                        if len(enc_json) == 0:
                            continue
                        decoded_enc = enc_json[0]
                    except JSONDecodeError:
                        print("[DEBUG] WARNING `enc_json` parsing faield!")
                    # pil_img = Image.open(io.BytesIO(enc.astype(bytes)))
                    pil_img = Image.open(io.BytesIO(base64.b64decode(decoded_enc)))
                    image = np.array(pil_img)
                    if len(image.shape) == 2:  # gray image
                        raise ValueError(
                            f"The image shape with {image.shape} is "
                            f"not in acceptable"
                        )
                    images.append(image)
                text_generation_input.prompt_images = images

            # Chat history not supported in Chat Completion Model
            # if pb_utils.get_input_tensor_by_name(request, "chat_history") is not None:
            #     chat_history_str = str(
            #         pb_utils.get_input_tensor_by_name(request, "chat_history")
            #         .as_numpy()[0]
            #         .decode("utf-8")
            #     )
            #     try:
            #         text_generation_input.chat_history = json.loads(chat_history_str)
            #     except json.decoder.JSONDecodeError:
            #         pass

            if pb_utils.get_input_tensor_by_name(request, "system_message") is not None:
                text_generation_input.system_message = str(
                    pb_utils.get_input_tensor_by_name(request, "system_message")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                if len(text_generation_input.system_message) == 0:
                    text_generation_input.system_message = None

            if pb_utils.get_input_tensor_by_name(request, "max_new_tokens") is not None:
                text_generation_input.max_new_tokens = int(
                    pb_utils.get_input_tensor_by_name(
                        request, "max_new_tokens"
                    ).as_numpy()[0]
                )

            if pb_utils.get_input_tensor_by_name(request, "top_k") is not None:
                text_generation_input.top_k = int(
                    pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()[0]
                )

            if pb_utils.get_input_tensor_by_name(request, "temperature") is not None:
                text_generation_input.temperature = round(
                    float(
                        pb_utils.get_input_tensor_by_name(
                            request, "temperature"
                        ).as_numpy()[0]
                    ),
                    2,
                )

            if pb_utils.get_input_tensor_by_name(request, "random_seed") is not None:
                text_generation_input.random_seed = int(
                    pb_utils.get_input_tensor_by_name(
                        request, "random_seed"
                    ).as_numpy()[0]
                )

            if pb_utils.get_input_tensor_by_name(request, "extra_params") is not None:
                extra_params_str = str(
                    pb_utils.get_input_tensor_by_name(request, "extra_params")
                    .as_numpy()[0]
                    .decode("utf-8")
                )
                try:
                    text_generation_input.extra_params = json.loads(extra_params_str)
                except json.decoder.JSONDecodeError:
                    pass

            print(
                f"Before Preprocessing `prompt`        : {type(text_generation_input.prompt)}. {text_generation_input.prompt}"
            )
            print(
                f"Before Preprocessing `prompt_images` : {type(text_generation_input.prompt_images)}. {text_generation_input.prompt_images}"
            )
            print(
                f"Before Preprocessing `chat_history`  : {type(text_generation_input.chat_history)}. {text_generation_input.chat_history}"
            )
            print(
                f"Before Preprocessing `system_message`: {type(text_generation_input.system_message)}. {text_generation_input.system_message}"
            )
            print(
                f"Before Preprocessing `max_new_tokens`: {type(text_generation_input.max_new_tokens)}. {text_generation_input.max_new_tokens}"
            )
            print(
                f"Before Preprocessing `temperature`   : {type(text_generation_input.temperature)}. {text_generation_input.temperature}"
            )
            print(
                f"Before Preprocessing `top_k`         : {type(text_generation_input.top_k)}. {text_generation_input.top_k}"
            )
            print(
                f"Before Preprocessing `random_seed`   : {type(text_generation_input.random_seed)}. {text_generation_input.random_seed}"
            )
            print(
                f"Before Preprocessing `stop_words`    : {type(text_generation_input.stop_words)}. {text_generation_input.stop_words}"
            )
            print(
                f"Before Preprocessing `extra_params`  : {type(text_generation_input.extra_params)}. {text_generation_input.extra_params}"
            )

            # Preprocessing
            if text_generation_input.random_seed > 0:
                random.seed(text_generation_input.random_seed)
                np.random.seed(text_generation_input.random_seed)
                torch.manual_seed(text_generation_input.random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(text_generation_input.random_seed)

            # Inference
            # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
            # https://huggingface.co/docs/transformers/v4.30.1/en/main_classes/text_generation#transformers.GenerationConfig
            t0 = time.time()
            print("----------------")
            print(f"[DEBUG] Conversation Prompt: \n{text_generation_input.prompt}")
            print("----------------")

            sequences = self.pipeline(
                text_generation_input.prompt,
                do_sample=True,
                top_k=text_generation_input.top_k,
                temperature=text_generation_input.temperature,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=text_generation_input.max_new_tokens,
                **text_generation_input.extra_params,
            )

            self.logger.log_info(
                f"Inference time cost {time.time()-t0}s with input lenth {len(text_generation_input.prompt)}"
            )

            text_outputs = [seq["generated_text"].encode("utf-8") for seq in sequences]
            triton_output_tensor = pb_utils.Tensor(
                "text", np.asarray(text_outputs, dtype=self.output0_dtype)
            )
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])
            )

        return responses

    def finalize(self):
        self.logger.log_info("Cleaning up ...")
