"""
Speculative Decoding for LMFast.

Accelerates inference by using a small draft model to predict tokens,
then verifying with the target model in a single forward pass.

Achieves 1.5-4x speedup without changing output quality.

Research:
- "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
- "Fast Inference from Transformers via Speculative Decoding" (2023)

Example:
    >>> from lmfast.inference import SpeculativeDecoder
    >>> decoder = SpeculativeDecoder("./my_1b_model", draft_model="./my_135m_model")
    >>> output = decoder.generate("Hello, how are you?", max_tokens=100)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeDecodingConfig:
    """Configuration for speculative decoding."""
    
    # Speculation depth (K tokens to draft)
    speculation_depth: int = 5
    
    # Draft model settings
    draft_temperature: float = 1.0
    
    # Target model settings
    target_temperature: float = 1.0
    
    # Acceptance settings
    acceptance_threshold: float = 0.0  # 0 = always accept if match
    
    # Performance
    max_batch_size: int = 1
    
    # Memory optimization
    share_kv_cache: bool = True


class SpeculativeDecoder:
    """
    Speculative decoding for accelerated LLM inference.
    
    Uses a small draft model to predict K tokens ahead,
    then verifies all K tokens with the target model in one forward pass.
    
    If draft predictions match target's distribution, we accept them.
    This achieves 1.5-4x speedup depending on draft quality.
    
    Example:
        >>> decoder = SpeculativeDecoder(
        ...     target_model="./my_1b_model",
        ...     draft_model="./my_135m_model",  # Optional: auto-selects if None
        ...     speculation_depth=5
        ... )
        >>> 
        >>> output = decoder.generate(
        ...     "Write a poem about AI:",
        ...     max_tokens=100
        ... )
    """
    
    # Auto-mapping of target models to recommended draft models
    DRAFT_MODEL_MAP = {
        "SmolLM-360M": "SmolLM-135M",
        "SmolLM-1.7B": "SmolLM-360M",
        "Qwen2.5-0.5B": "SmolLM-135M",
        "Qwen2.5-1.5B": "Qwen2.5-0.5B",
        "Qwen2.5-3B": "Qwen2.5-0.5B",
        "Llama-3.2-1B": "SmolLM-360M",
        "Llama-3.2-3B": "Llama-3.2-1B",
        "Phi-3.5-mini": "SmolLM-360M",
        "TinyLlama-1.1B": "SmolLM-360M",
    }
    
    def __init__(
        self,
        target_model: str,
        draft_model: Optional[str] = None,
        config: Optional[SpeculativeDecodingConfig] = None,
        device: str = "auto"
    ):
        """
        Initialize speculative decoder.
        
        Args:
            target_model: Path to target (large) model
            draft_model: Path to draft (small) model. Auto-selects if None.
            config: Decoding configuration
            device: Device to use
        """
        self.config = config or SpeculativeDecodingConfig()
        self.device = self._get_device(device)
        
        # Load models
        self.target_model_path = target_model
        self.draft_model_path = draft_model or self._select_draft_model(target_model)
        
        self.target = None
        self.draft = None
        self.tokenizer = None
        
        # Statistics
        self.stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "draft_calls": 0,
            "target_calls": 0,
            "time_draft": 0.0,
            "time_target": 0.0,
        }
        
        logger.info(f"SpeculativeDecoder: target={target_model}, draft={self.draft_model_path}")
    
    def _get_device(self, device: str) -> str:
        """Determine device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _select_draft_model(self, target_model: str) -> Optional[str]:
        """Auto-select a suitable draft model."""
        for key, value in self.DRAFT_MODEL_MAP.items():
            if key.lower() in target_model.lower():
                logger.info(f"Auto-selected draft model: {value}")
                return value
        return None
    
    def _load_models(self):
        """Load target and draft models."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Loading models for speculative decoding...")
        
        # Load target model
        self.target = AutoModelForCausalLM.from_pretrained(
            self.target_model_path,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.target.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load draft model
        if self.draft_model_path:
            self.draft = AutoModelForCausalLM.from_pretrained(
                self.draft_model_path,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            self.draft.eval()
        else:
            logger.warning("No draft model available. Using standard decoding.")
            self.draft = None
        
        logger.info("Models loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        return_stats: bool = False
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Generate text with speculative decoding.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop_sequences: Sequences that stop generation
            return_stats: Whether to return generation statistics
            
        Returns:
            Generated text, optionally with statistics
        """
        # Lazy load models
        if self.target is None:
            self._load_models()
        
        # Use standard decoding if no draft model
        if self.draft is None:
            return self._generate_standard(prompt, max_tokens, temperature)
        
        # Reset stats
        self.stats = {k: 0 for k in self.stats}
        start_time = time.time()
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()
        
        tokens_generated = 0
        k = self.config.speculation_depth
        
        while tokens_generated < max_tokens:
            # Step 1: Draft K tokens
            t0 = time.time()
            draft_tokens, draft_probs = self._draft_tokens(generated_ids, k)
            self.stats["time_draft"] += time.time() - t0
            self.stats["draft_calls"] += 1
            
            # Step 2: Verify with target model
            t0 = time.time()
            accepted_tokens, new_token = self._verify_tokens(
                generated_ids, 
                draft_tokens, 
                draft_probs,
                temperature
            )
            self.stats["time_target"] += time.time() - t0
            self.stats["target_calls"] += 1
            
            # Step 3: Accept tokens and add new token
            if len(accepted_tokens) > 0:
                generated_ids = torch.cat([generated_ids, accepted_tokens], dim=1)
                self.stats["accepted_tokens"] += len(accepted_tokens[0])
            
            # Always add at least one new token from target
            if new_token is not None:
                generated_ids = torch.cat([
                    generated_ids, 
                    new_token.unsqueeze(0).unsqueeze(0)
                ], dim=1)
            
            tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
            self.stats["total_tokens"] = tokens_generated
            
            # Check for EOS
            if self.tokenizer.eos_token_id in generated_ids[0][-k:]:
                break
            
            # Check stop sequences
            if stop_sequences:
                current_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                if any(seq in current_text for seq in stop_sequences):
                    break
        
        # Decode output
        output = self.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        total_time = time.time() - start_time
        
        # Calculate acceptance rate
        if self.stats["draft_calls"] > 0:
            acceptance_rate = (
                self.stats["accepted_tokens"] / 
                (self.stats["draft_calls"] * k)
            )
        else:
            acceptance_rate = 0
        
        self.stats["acceptance_rate"] = acceptance_rate
        self.stats["total_time"] = total_time
        self.stats["tokens_per_second"] = tokens_generated / total_time if total_time > 0 else 0
        
        logger.info(
            f"Generated {tokens_generated} tokens in {total_time:.2f}s "
            f"({self.stats['tokens_per_second']:.1f} tok/s, "
            f"acceptance: {acceptance_rate:.1%})"
        )
        
        if return_stats:
            return output, self.stats
        return output
    
    def _draft_tokens(
        self,
        input_ids: torch.Tensor,
        k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate K draft tokens using the draft model.
        
        Returns:
            (draft_tokens, draft_probabilities)
        """
        draft_tokens = []
        draft_probs = []
        
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(k):
                outputs = self.draft(current_ids)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                probs = F.softmax(logits / self.config.draft_temperature, dim=-1)
                
                # Sample
                next_token = torch.multinomial(probs, num_samples=1)
                token_prob = probs[0, next_token[0, 0]]
                
                draft_tokens.append(next_token)
                draft_probs.append(token_prob)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        draft_tokens = torch.cat(draft_tokens, dim=1)
        draft_probs = torch.stack(draft_probs)
        
        return draft_tokens, draft_probs
    
    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Verify draft tokens using the target model.
        
        Uses rejection sampling to accept tokens that match
        the target distribution.
        
        Returns:
            (accepted_tokens, new_token_from_target)
        """
        # Concatenate input with draft tokens
        full_ids = torch.cat([input_ids, draft_tokens], dim=1)
        
        with torch.no_grad():
            outputs = self.target(full_ids)
        
        # Get probabilities for each position
        logits = outputs.logits
        
        accepted = []
        new_token = None
        
        for i, draft_token in enumerate(draft_tokens[0]):
            pos = input_ids.shape[1] + i - 1
            if pos < 0:
                pos = 0
            
            # Get target probability for draft token
            target_probs = F.softmax(logits[0, pos] / temperature, dim=-1)
            target_prob = target_probs[draft_token].item()
            draft_prob = draft_probs[i].item()
            
            # Rejection sampling
            # Accept if target_prob >= draft_prob, or randomly with p = target_prob/draft_prob
            if draft_prob > 0:
                accept_prob = min(1.0, target_prob / draft_prob)
            else:
                accept_prob = 0.0
            
            if torch.rand(1).item() < accept_prob:
                accepted.append(draft_token)
            else:
                # Rejection - sample new token from adjusted distribution
                # p'(x) = max(0, p_target(x) - p_draft(x)) / Z
                adjusted_probs = torch.clamp(target_probs - F.softmax(logits[0, pos] / self.config.draft_temperature, dim=-1), min=0)
                if adjusted_probs.sum() > 0:
                    adjusted_probs = adjusted_probs / adjusted_probs.sum()
                    new_token = torch.multinomial(adjusted_probs, num_samples=1)[0]
                else:
                    new_token = torch.multinomial(target_probs, num_samples=1)[0]
                break
        
        if accepted:
            accepted_tokens = torch.tensor([accepted], device=self.device)
        else:
            accepted_tokens = torch.tensor([[]], device=self.device)
        
        # If all were accepted, sample one more from target
        if new_token is None and len(accepted) == draft_tokens.shape[1]:
            last_pos = input_ids.shape[1] + len(accepted) - 1
            target_probs = F.softmax(logits[0, last_pos] / temperature, dim=-1)
            new_token = torch.multinomial(target_probs, num_samples=1)[0]
        
        return accepted_tokens, new_token
    
    def _generate_standard(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Standard autoregressive generation (fallback)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.target.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    def benchmark(
        self,
        prompts: List[str],
        max_tokens: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark speculative vs standard decoding.
        
        Returns speedup and statistics.
        """
        if self.target is None:
            self._load_models()
        
        # Speculative decoding
        spec_times = []
        spec_tokens = []
        for prompt in prompts:
            _, stats = self.generate(prompt, max_tokens, return_stats=True)
            spec_times.append(stats["total_time"])
            spec_tokens.append(stats["total_tokens"])
        
        # Standard decoding
        std_times = []
        for prompt in prompts:
            start = time.time()
            self._generate_standard(prompt, max_tokens, 1.0)
            std_times.append(time.time() - start)
        
        avg_spec = sum(spec_times) / len(spec_times)
        avg_std = sum(std_times) / len(std_times)
        speedup = avg_std / avg_spec if avg_spec > 0 else 1.0
        
        return {
            "speculative_time": avg_spec,
            "standard_time": avg_std,
            "speedup": speedup,
            "average_acceptance_rate": sum(spec_tokens) / (len(prompts) * self.config.speculation_depth * len(prompts)),
        }


def generate_fast(
    model_path: str,
    prompt: str,
    draft_model: Optional[str] = None,
    max_tokens: int = 256,
    **kwargs
) -> str:
    """
    One-line fast generation with speculative decoding.
    
    Args:
        model_path: Path to target model
        prompt: Input prompt
        draft_model: Optional draft model path
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
        
    Example:
        >>> from lmfast.inference import generate_fast
        >>> output = generate_fast("./my_model", "Hello, how are you?")
    """
    decoder = SpeculativeDecoder(model_path, draft_model)
    return decoder.generate(prompt, max_tokens, **kwargs)
