"""LLM-based query rewriter with LoRA fine-tuning via PEFT.

Supports two training paradigms:
  - **SFT** (Supervised Fine-Tuning): train on (prompt, completion) pairs
  - **DPO** (Direct Preference Optimization): train on (prompt, chosen, rejected)

The base model is loaded in a memory-efficient configuration suitable for
consumer hardware (e.g. Apple Silicon with 16 GB unified memory).

Architecture::

    ┌────────────────────────────┐
    │  Base Causal LM            │  (frozen weights)
    │  e.g. Qwen/Qwen2.5-0.5B   │
    ├────────────────────────────┤
    │  LoRA Adapters             │  (trainable, rank=16)
    │  target: q_proj, v_proj,   │
    │          k_proj, o_proj     │
    └────────────────────────────┘
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from models.prompting import format_sft_prompt


class QueryRewriter(nn.Module):
    """Wraps a causal LM and applies LoRA adapters for query rewriting.

    Parameters
    ----------
    base_model_name : str
        HuggingFace model identifier (e.g. ``Qwen/Qwen2.5-0.5B``).
    lora_rank : int
        Rank for LoRA decomposition.
    lora_alpha : int
        LoRA scaling factor (typically ``2 * rank``).
    lora_dropout : float
        Dropout probability inside LoRA layers.
    target_modules : list[str] | None
        Attention projection layers to adapt. If None, uses a sensible
        default covering all attention projections.
    load_in_8bit : bool
        Use 8-bit quantization for the base model (requires bitsandbytes).
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        load_in_8bit: bool = False,
        torch_dtype: str = "auto",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.base_model_name = base_model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        model_source = self._resolve_model_source(base_model_name)

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Load tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        resolved_dtype = self._resolve_torch_dtype(torch_dtype)

        # Load base model
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": resolved_dtype,
            # "torch_type": torch.bfloat16
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "sdpa"
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        base_model = AutoModelForCausalLM.from_pretrained(
            model_source, **model_kwargs
        )
        if gradient_checkpointing:
            base_model.config.use_cache = False
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)
        if gradient_checkpointing and hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        self._print_trainable_params()

    @staticmethod
    def _resolve_model_source(base_model_name: str) -> str:
        try:
            cached_snapshot = snapshot_download(base_model_name, local_files_only=True)
        except Exception:
            return base_model_name

        print(f"  Using cached model snapshot: {cached_snapshot}")
        return cached_snapshot

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: str) -> torch.dtype:
        if torch_dtype == "auto":
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if torch_dtype not in dtype_map:
            raise ValueError(
                f"Unsupported torch_dtype={torch_dtype!r}. "
                "Use one of: auto, float32, float16, bfloat16."
            )
        return dtype_map[torch_dtype]

    def _print_trainable_params(self) -> None:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = 100.0 * trainable / total if total > 0 else 0.0
        print(f"  Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the causal LM forward pass.

        Parameters
        ----------
        input_ids : (batch, seq_len)
        attention_mask : (batch, seq_len)
        labels : (batch, seq_len), optional — Target token IDs for LM loss.

        Returns
        -------
        dict with ``logits`` and optionally ``loss``.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        result: Dict[str, torch.Tensor] = {"logits": outputs.logits}
        if outputs.loss is not None:
            result["loss"] = outputs.loss
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 64,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Auto-regressively generate rewritten queries."""
        self.model.eval()
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            **generate_kwargs,
        )

    @torch.no_grad()
    def rewrite_queries(
        self,
        queries: List[str],
        max_new_tokens: int = 64,
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> List[str]:
        """High-level API: rewrite a list of raw query strings.

        Parameters
        ----------
        queries : list[str] — Raw user queries.
        max_new_tokens : int — Max generation length.
        batch_size : int — Inference batch size.
        device : str, optional — Device override.

        Returns
        -------
        list[str] — Rewritten queries.
        """
        if device is None:
            device = next(self.model.parameters()).device.type

        self.model.eval()
        results: List[str] = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            prompts = [format_sft_prompt(q) for q in batch]

            encoded = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            prompt_len = encoded["input_ids"].shape[1]

            output_ids = self.generate(
                encoded["input_ids"],
                encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
            )

            for j in range(output_ids.shape[0]):
                gen_ids = output_ids[j, prompt_len:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                results.append(text)

        return results

    def save_adapter(self, path: str) -> None:
        """Persist only the LoRA adapter weights."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"  LoRA adapter saved to {path}")

    def load_adapter(self, path: str) -> None:
        """Load LoRA adapter weights from disk."""
        self.model = PeftModel.from_pretrained(
            self.model.base_model.model,
            path,
            is_trainable=True,
        )
        print(f"  LoRA adapter loaded from {path}")

    def get_ref_model(self) -> nn.Module:
        """Create a frozen copy of the current model for DPO reference.

        Returns a model with the same architecture but detached from
        gradient computation. Used as π_ref in DPO.
        """
        ref = copy.deepcopy(self.model)
        for param in ref.parameters():
            param.requires_grad = False
        ref.eval()
        return ref
