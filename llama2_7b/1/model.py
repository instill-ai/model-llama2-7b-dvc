# pylint: skip-file
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import json
import time
from pathlib import Path

import traceback

import numpy as np
import triton_python_backend_utils as pb_utils
from vllm import SamplingParams, LLM

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        model_path = str(Path(__file__).parent.absolute().joinpath('llama-2-7b-f16-hf/'))

        # https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py
        self.llm_engine = LLM(
            model = model_path,
            gpu_memory_utilization = 0.21,
            tensor_parallel_size = 2 # for A100 40G Memory
        )

        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # request_id = random_uuid()
                prompt = str(pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode("utf-8"))
                print(f'[DEBUG] input `prompt` type({type(prompt)}): {prompt}')

                extra_params_str = ""
                if pb_utils.get_input_tensor_by_name(request, "extra_params") is not None:
                    extra_params_str = str(pb_utils.get_input_tensor_by_name(request, "extra_params").as_numpy()[0].decode("utf-8"))
                print(f'[DEBUG] input `extra_params` type({type(extra_params_str)}): {extra_params_str}')

                extra_params = {}
                # TODO: Add a function handle penalty
                try:
                    extra_params = json.loads(extra_params_str)
                except json.decoder.JSONDecodeError:
                    pass

                max_new_tokens = 100
                if pb_utils.get_input_tensor_by_name(request, "max_new_tokens") is not None:
                    max_new_tokens = int(pb_utils.get_input_tensor_by_name(request, "max_new_tokens").as_numpy()[0])
                print(f'[DEBUG] input `max_new_tokens` type({type(max_new_tokens)}): {max_new_tokens}')

                top_k = 1
                if pb_utils.get_input_tensor_by_name(request, "top_k") is not None:
                    top_k = int(pb_utils.get_input_tensor_by_name(request, "top_k").as_numpy()[0])
                print(f'[DEBUG] input `top_k` type({type(top_k)}): {top_k}')

                temperature = 0.8
                if pb_utils.get_input_tensor_by_name(request, "temperature") is not None:
                    temperature = float(pb_utils.get_input_tensor_by_name(request, "temperature").as_numpy()[0])
                temperature = round(temperature, 2)
                print(f'[DEBUG] input `temperature` type({type(temperature)}): {temperature}')

                random_seed = 0
                if pb_utils.get_input_tensor_by_name(request, "random_seed") is not None:
                    random_seed = int(pb_utils.get_input_tensor_by_name(request, "random_seed").as_numpy()[0])
                print(f'[DEBUG] input `random_seed` type({type(random_seed)}): {random_seed}')

                if random_seed > 0:
                   random.seed(random_seed)
                   np.random.seed(random_seed)
                #    torch.manual_seed(random_seed)
                #    if torch.cuda.is_available():
                #        torch.cuda.manual_seed_all(random_seed)

                stop_words = ""
                if pb_utils.get_input_tensor_by_name(request, "stop_words") is not None:
                    stop_words = pb_utils.get_input_tensor_by_name(request, "stop_words").as_numpy()
                print(f'[DEBUG] input `stop_words` type({type(stop_words)}): {stop_words}')
                if len(stop_words) == 0:
                    stop_words = None
                elif stop_words.shape[0] > 1:
                    # TODO: Check wether shoule we decode this words
                    stop_words = list(stop_words)
                else:
                    stop_words = [str(stop_words[0])]

                if stop_words is not None:
                    extra_params['stop'] = stop_words
                print(f'[DEBUG] parsed input `stop_words` type({type(stop_words)}): {stop_words}')

                # TODO: Add a extra_params checker
                # Reference for Sampling Parameters
                # https://github.com/vllm-project/vllm/blob/v0.2.0/vllm/sampling_params.py
                sampling_params = SamplingParams(
                    temperature = temperature,
                    max_tokens = max_new_tokens,
                    top_k = top_k,
                    **extra_params
                )

                # calculate time cost in following function call
                t0 = time.time()
                vllm_outputs = self.llm_engine.generate(prompt, sampling_params)
                self.logger.log_info(f'Inference time cost {time.time()-t0}s with input lenth {len(prompt)}')

                text_outputs = []
                for vllm_output in vllm_outputs:
                    concated_complete_output = prompt + "".join([
                        str(complete_output.text)
                        for complete_output in vllm_output.outputs
                    ])
                    text_outputs.append(concated_complete_output.encode("utf-8"))
                triton_output_tensor = pb_utils.Tensor(
                    "text", np.asarray(text_outputs, dtype=self.output0_dtype)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[triton_output_tensor]))


            except Exception as e:
                print("DEBUG\n", traceback.format_exc())
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

        return responses

    def finalize(self):
        self.logger.log_info("Issuing finalize to vllm backend")