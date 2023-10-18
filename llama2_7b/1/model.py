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
