import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
from pathlib import Path

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
            gpu_memory_utilization = 0.42,
            tensor_parallel_size = 1 # for A100 40G Memory
        )
        
        output0_config = pb_utils.get_output_config_by_name(self.model_config, "text")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # request_id = random_uuid()
                prompt = str(pb_utils.get_input_tensor_by_name(request, "query").as_numpy()[0].decode("utf-8"))
                output_len = pb_utils.get_input_tensor_by_name(request, "len").as_numpy()[0]
                
                # bad_words_list, stop_words_list, random_seed are not used in this model
                stop_words_list = str(pb_utils.get_input_tensor_by_name(request, "stop").as_numpy()[0].decode("utf-8"))
                topk = pb_utils.get_input_tensor_by_name(request, "runtime_top_k").as_numpy()[0]
                
                # Reference for Sampling Parameters
                # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
                sampling_params = SamplingParams(
                    temperature = 0.8,
                    max_tokens = int(output_len),
                    top_k = int(topk),
                    # stop = stop_words_list,
                    # ignore_eos = False
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