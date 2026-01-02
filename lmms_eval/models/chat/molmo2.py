from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
from PIL import Image

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

@register_model("molmo2")
class Molmo2(lmms):
    is_simple = False

    def __init__(self, pretrained: str, 
                 device: str = "cuda", 
                 batch_size: int = 4,
                 **kwargs):
        super().__init__()

        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            trust_remote_code=True,
            dtype="auto",
            device_map=device
        )

        # load the model
        self.model = AutoModelForImageTextToText.from_pretrained(
             pretrained,
             trust_remote_code=True,
             dtype="auto",
             device_map=device
        )

        self.batch_size = int(batch_size)

    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []

        for request in tqdm(requests, desc="Generating answers"):
            question, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            
            messages = doc_to_messages(doc)

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # generate output
            gen_kwargs.pop("until") # Not used by Molmo2
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **gen_kwargs)

            # only get generated tokens; decode them to text
            generated_tokens = generated_ids[0, inputs['input_ids'].size(1):]
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # print the generated text
            results.append(generated_text)

        return results


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        ...
        
    def generate_until_multi_round(self, requests) -> List[str]:
        return super().generate_until_multi_round(requests)

