<<<<<<< HEAD
# pylint: skip-file
import time
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
import transformers
import torch

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


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
* model_name: Model name
"""
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])

        # Load the model
        model_path = str(Path(__file__).parent.absolute().joinpath('Llama-2-7b-hf/')) # .with_name('llama-2-7b-hf/') # fix invalid error

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            torch_dtype=torch.float32, # Needs 26G Memory
            device="cpu"
        )

        # Get output configurations
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                prompt = pb_utils.get_input_tensor_by_name(request, "query").as_numpy()[0]
                output_len = pb_utils.get_input_tensor_by_name(request, "len").as_numpy()[0]
                topk = pb_utils.get_input_tensor_by_name(request, "runtime_top_k").as_numpy()[0]

                # bad_words_list, stop_words_list, random_seed are not used in this model
                decoded_input = str(prompt.decode("utf-8"))

                # calculate time cost in following function call
                t0 = time.time()
                sequences = self.pipeline(
                    decoded_input,
                    do_sample=True,
                    top_k=int(topk),
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=int(output_len),
                )

                self.logger.log_info(f'Inference time cost {time.time()-t0}s with input lenth {len(decoded_input)}')

                text_outputs = [
                    seq['generated_text'].encode('utf-8')
                    for seq in sequences
                ]
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(text_outputs, dtype=self.output0_dtype)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[triton_output_tensor]))

            except Exception as e:
                self.logger.log_info(f"Error generating stream: {e}")
                error = pb_utils.TritonError(f"Error generating stream: {e}")
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(["N/A"], dtype=self.output0_dtype)
                )
                response = pb_utils.InferenceResponse(
                    output_tensors=[triton_output_tensor], error=error
                )
                responses.append(response)
                self.logger.log_info("The model did not receive the expected inputs")
                raise e
            return responses

    def finalize(self):
        self.logger.log_info("Cleaning up ...")
=======
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
import json
import os
import threading
from typing import AsyncGenerator

import numpy as np
import triton_python_backend_utils as pb_utils
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        # assert are in decoupled mode. Currently, Triton needs to use
        # decoupled policy for asynchronously forwarding requests to
        # vLLM engine.
        self.using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self.model_config
        )
        assert (
            self.using_decoupled
        ), "vLLM Triton backend must be configured to use decoupled model transaction policy"

        model_path = os.path.join(os.getcwd(), 'llama-2-7b-f16-hf')

        # Create an AsyncLLMEngine from the config from JSON
        # Reference for AsyncLLMEngine Configs:
        # https://github.com/vllm-project/vllm/blob/main/vllvlm/engine/arg_utils.py
        self.llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model = model_path,
                disable_log_requests = "true",
                gpu_memory_utilization = 0.9
            )
        )

        output_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

        # Counter to keep track of ongoing request counts
        self.ongoing_request_count = 0

        # Starting asyncio event loop to process the received requests asynchronously.
        self._loop = asyncio.get_event_loop()
        self._loop_thread = threading.Thread(
            target=self.engine_loop, args=(self._loop,)
        )
        self._shutdown_event = asyncio.Event()
        self._loop_thread.start()

    def create_task(self, coro):
        """
        Creates a task on the engine's event loop which is running on a separate thread.
        """
        assert (
            self._shutdown_event.is_set() is False
        ), "Cannot create tasks after shutdown has been requested"

        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def engine_loop(self, loop):
        """
        Runs the engine's event loop on a separate thread.
        """
        asyncio.set_event_loop(loop)
        self._loop.run_until_complete(self.await_shutdown())

    async def await_shutdown(self):
        """
        Primary coroutine running on the engine event loop. This coroutine is responsible for
        keeping the engine alive until a shutdown is requested.
        """
        # first await the shutdown signal
        while self._shutdown_event.is_set() is False:
            await asyncio.sleep(5)

        # Wait for the ongoing_requests
        while self.ongoing_request_count > 0:
            self.logger.log_info(
                "Awaiting remaining {} requests".format(self.ongoing_request_count)
            )
            await asyncio.sleep(5)

        self.logger.log_info("Shutdown complete")

    def get_sampling_params_dict(self, params_json):
        """
        This functions parses the dictionary values into their
        expected format.
        """

        params_dict = json.loads(params_json)

        # Special parsing for the supported sampling parameters
        # TODO: Add more parameters if needed
        float_keys = ["temperature", "top_p"]
        for k in float_keys:
            if k in params_dict:
                params_dict[k] = float(params_dict[k])

        return params_dict

    def create_response(self, vllm_output):
        """
        Parses the output from the vLLM engine into Triton
        response.
        """
        prompt = vllm_output.prompt
        text_outputs = [
            (prompt + output.text).encode("utf-8") for output in vllm_output.outputs
        ]
        triton_output_tensor = pb_utils.Tensor(
            "text", np.asarray(text_outputs, dtype=self.output_dtype)
        )
        return pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])

    async def generate(self, request):
        """
        Forwards single request to LLM engine and returns responses.
        """
        response_sender = request.get_response_sender()
        self.ongoing_request_count += 1
        try:
            request_id = random_uuid()
            prompt = pb_utils.get_input_tensor_by_name(request, "query").as_numpy()[0]
            output_len = pb_utils.get_input_tensor_by_name(request, "len").as_numpy()[0]
            
            # bad_words_list = pb_utils.get_input_tensor_by_name(request, "bad").as_numpy()[0].decode("utf-8")
            # bad_words_list is not used in the vllm engine

            stop_words_list = pb_utils.get_input_tensor_by_name(request, "stop").as_numpy()[0].decode("utf-8")
            topk = pb_utils.get_input_tensor_by_name(request, "runtime_top_k").as_numpy()[0]
            
            # random_seed = pb_utils.get_input_tensor_by_name(request, "random_seed").as_numpy()[0]
            # bad_words_list is not used in the vllm engine
            
            stream = False
            # TODO: Add `stream` to Input Parameters
            
            # Reference for Sampling Parameters
            # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
            sampling_params = SamplingParams(
                temperature = 0.8, # TODO: Add to Input Parameters
                max_tokens = int(output_len),
                top_k = int(topk),
                stop = stop_words_list,
                early_stopping = True,
                ignore_eos = False
            )
            last_output = None
            async for output in self.llm_engine.generate(
                str(prompt), sampling_params, request_id
            ):
                if stream:
                    response_sender.send(self.create_response(output))
                else:
                    last_output = output
            if not stream:
                response_sender.send(self.create_response(last_output))

        except Exception as e:
            self.logger.log_info(f"Error generating stream: {e}")
            error = pb_utils.TritonError(f"Error generating stream: {e}")
            triton_output_tensor = pb_utils.Tensor(
                "text", np.asarray(["N/A"], dtype=self.output_dtype)
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[triton_output_tensor], error=error
            )
            response_sender.send(response)
            raise e
        finally:
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            self.ongoing_request_count -= 1

    def execute(self, requests):
        """
        Triton core issues requests to the backend via this method.

        When this method returns, new requests can be issued to the backend. Blocking
        this function would prevent the backend from pulling additional requests from
        Triton into the vLLM engine. This can be done if the kv cache within vLLM engine
        is too loaded.
        We are pushing all the requests on vllm and let it handle the full traffic.
        """
        for request in requests:
            self.create_task(self.generate(request))
        return None

    def finalize(self):
        """
        Triton virtual method; called when the model is unloaded.
        """
        self.logger.log_info("Issuing finalize to vllm backend")
        self._shutdown_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join()
            self._loop_thread = None
>>>>>>> 11208b4... feat: support vllm
