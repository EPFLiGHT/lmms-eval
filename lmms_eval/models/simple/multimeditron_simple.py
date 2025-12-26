import logging
from typing import List, Tuple, Optional, Dict, Any
from multimeditron.dataset.loader import FileSystemImageLoader, RawImageLoader
from multimeditron.model.model import ChatTemplate, MultiModalModelForCausalLM
from multimeditron.model.data_loader import DataCollatorForMultimodal
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

from loguru import logger as eval_logger

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

@register_model("multimeditron_simple")
class MultiMeditronSimple(lmms):
    is_simple = True

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
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left", use_fast=True)
        except:
            default_llm = kwargs.pop("default_llm", None)
            if default_llm is None:
                raise ValueError("Default LLM must be specified if tokenizer loading fails.")

            eval_logger.warning(f"Loading tokenizer from {default_llm}")
            self.tokenizer = AutoTokenizer.from_pretrained(default_llm, padding_side="left", use_fast=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        loader = RawImageLoader()

        self.collator = DataCollatorForMultimodal(
                tokenizer=self.tokenizer,
                attachment_token=self.attachment_token,
                chat_template=ChatTemplate.from_name(tokenizer_type),
                modality_processors=self.model.processors(), 
                modality_loaders={"image" : loader},
                add_generation_prompt=False,
        )

        self.prompt_collator = DataCollatorForMultimodal(
                tokenizer=self.tokenizer,
                attachment_token=self.attachment_token,
                chat_template=ChatTemplate.from_name(tokenizer_type),
                modality_processors=self.model.processors(), 
                modality_loaders={"image" : loader},
                add_generation_prompt=True,
        )

        self.batch_size = int(batch_size)

        if self.tokenizer.eos_token_id is not None:
            self.eos_length = 1
        else:
            self.eos_length = 0


    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        for request in tqdm(requests, desc="Processing requests"):
            messages = self._build_messages(request)

            batch = self.collator([messages])

            batch["input_ids"] = batch["input_ids"].to(self.device)
            batch["position_ids"] = batch["position_ids"].to(self.device)
            batch["attention_mask"] = batch["attention_mask"].to(self.device)
            batch["labels"] = batch["labels"].to(self.device)
            
            prompt_ids = self.compute_prompt_ids(messages).to(self.device)
            continuation_ids = batch["input_ids"][0, len(prompt_ids) : -self.eos_length].to(self.device)

            outputs = self.model(**batch)

            loss = outputs["loss"]
            logits = outputs["logits"][0][:-1]
            greedy_tokens = logits.argmax(dim=-1).to(self.device)

            greedy_tokens = greedy_tokens[prompt_ids.shape[0] - 1: -self.eos_length]

            # print("Decoded", self.tokenizer.decode(greedy_tokens), "continuation_ids", self.tokenizer.decode(continuation_ids)) 

            max_equal = (greedy_tokens == continuation_ids).all()

            res.append((float(loss.item()), bool(max_equal)))

        return res

    def compute_prompt_ids(self, messages: Dict[str, Any]) -> torch.Tensor:
        prompt_only_conv = messages["conversations"][:-1]
        sample = {
                "conversations" : prompt_only_conv,
                "modalities" : messages["modalities"]
        }
        batch = self.prompt_collator([sample])

        prompt_ids = batch["input_ids"][0]

        return prompt_ids

    def _build_messages(self, request: Instance) -> Dict[str, Any]:
        contexts, doc_to_target, doc_to_visual, doc_id, task, split = request.args
        doc = self.task_dict[task][split][doc_id]

        images = doc_to_visual(doc)
        attachments = "".join([self.attachment_token for _ in range(len(images))])

        if not isinstance(doc_to_target, str):
            label = doc_to_target(doc)
        else:
            label = doc_to_target

        conversations = [
                {
                    "role" : "user",
                    "content" : f"{attachments} {contexts}",
                },
                {
                    "role" : "assistant",
                    "content" : label,
                }
        ]

        messages = {
            "conversations" : conversations,
            "modalities" : [
                {"type" : "image", "value" : img} 
                for img in images
            ]
        }

        return messages 

    def generate_until(self, requests: List[Instance]):
        raise NotImplementedError()

    def generate_until_multi_round(self, requests: List[Instance]):
        raise NotImplementedError()
