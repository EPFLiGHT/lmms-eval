import logging
from typing import List, Tuple, Optional, Dict, Any
from multimeditron.model.model import MultiModalModelForCausalLM
from multimeditron.model.data_loader import DataCollatorForMultimodal
import torch
from transformers import AutoTokenizer

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

@register_model("multimeditron")
class MultiMeditron(lmms):
    is_simple = False

    def __init__(self, pretrained: str, device: str = "cuda", 
                 attachment_token: str = "<|reserved_special_token_0|>", 
                 batch_size: int = 4,
                 tokenizer_type: str = "llama",
                 **kwargs):
        super().__init__()

        self.device = device
        self.model = MultiModalModelForCausalLM.from_pretrained(pretrained, dtype=torch.bfloat16)

        self.model.to(self.device)
        self.attachment_token = attachment_token
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        except:
            default_llm = kwargs.pop("default_llm", None)
            if default_llm is None:
                raise ValueError("Default LLM must be specified if tokenizer loading fails.")

            eval_logger.warning(f"Loading tokenizer from {default_llm}")
            self.tokenizer = AutoTokenizer.from_pretrained(default_llm)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        attachment_token_idx = self.tokenizer.convert_tokens_to_ids(attachment_token) 
        self.collator = DataCollatorForMultimodal(
                tokenizer=self.tokenizer,
                tokenizer_type=tokenizer_type,
                modality_processors=self.model.processors(), 
                attachment_token_idx=attachment_token_idx
        )

        self.batch_size = int(batch_size)

    
    def generate_until(self, requests: List[Instance]) -> List[str]:
        results = []
        all_messages = []

        for request in requests:
            question, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = self.map_messages_to_multimeditron_format(doc_to_messages(doc))
            all_messages.append(messages)

        for i in range(0, len(all_messages), self.batch_size):
            batch_messages = all_messages[i:i + self.batch_size]
            batch = self.collator(batch_messages)

            outputs = self.model.generate(
                input_ids=batch["input_ids"],
                processed_multimodal_inputs=batch["processed_multimodal_inputs"],
            )

            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            eval_logger.info(f"Sample outputs: {decoded_outputs[0]}")
            results.extend(decoded_outputs)

        return results


    def map_messages_to_multimeditron_format(self, messages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        mapped_messages = []
        modalities = []
        for message in messages:
            mapped_text = ""
            for content_part in message["content"]:
                match content_part["type"]:
                    case "text":
                        mapped_text += content_part["text"]
                    case "image":
                        mapped_text += self.attachment_token
                        modalities.append({
                            "type" : "image",
                            "value" : content_part["url"]
                        })
                    case _:
                        eval_logger.warning(f"Skipping unknown content type {content_part['type']}")

            mapped_messages.append({
                "role" : message["role"],
                "content" : mapped_text
            })

        
        return {
            "conversations": mapped_messages,
            "modalities": modalities
        }


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        return super().loglikelihood(requests)

    def generate_until_multi_round(self, requests) -> List[str]:
        return super().generate_until_multi_round(requests)

