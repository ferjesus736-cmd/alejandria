import os
from functools import lru_cache
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from providers.base import BaseGenerator


class HuggingFaceProvider(BaseGenerator):
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("HF_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

    @lru_cache(maxsize=1)
    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model, use_fast=True)

    @lru_cache(maxsize=1)
    def _load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype=torch.float32,
        )

    def _generate_text(self, prompt: str, max_new_tokens: int = 96) -> str:
        tokenizer = self._load_tokenizer()
        model = self._load_model()
        model.to("cpu")
        model.eval()

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = prompt

        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = output_tokens[0][prompt_length:]
        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated.strip()

    def generate(self, prompt: str) -> str:
        try:
            return self._generate_text(prompt)
        except Exception as exc:
            return f"Error conectando con Hugging Face local: {exc}"

    def stream(self, prompt: str) -> Iterator[str]:
        try:
            text = self._generate_text(prompt)
            for token in text.split():
                yield token + " "
        except Exception as exc:
            yield f"\n[Error de Streaming Hugging Face local: {exc}]"

    def is_available(self) -> tuple[bool, str]:
        try:
            self._load_tokenizer()
            self._load_model()
            return (True, f"Hugging Face local model ready: {self.model}")
        except Exception as exc:
            return (False, f"Failed to load Hugging Face local model: {exc}")
