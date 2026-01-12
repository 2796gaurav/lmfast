"""
Model explainability utilities.

Provides tools to understand model behavior and decisions.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualize attention patterns in transformer models.

    Example:
        >>> viz = AttentionVisualizer(model, tokenizer)
        >>> viz.visualize("The quick brown fox")
    """

    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialize attention visualizer.

        Args:
            model: The language model
            tokenizer: The tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def get_attention(
        self,
        text: str,
        *,
        layer: int = -1,
        head: int | None = None,
    ) -> dict:
        """
        Get attention weights for text.

        Args:
            text: Input text
            layer: Which layer (-1 for last)
            head: Which attention head (None for all)

        Returns:
            Dictionary with tokens and attention weights
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Forward with attention output
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
            )

        # Get attention from specified layer
        attentions = outputs.attentions[layer]  # (batch, heads, seq, seq)

        if head is not None:
            attentions = attentions[:, head : head + 1, :, :]

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return {
            "tokens": tokens,
            "attention": attentions[0].cpu().numpy(),  # (heads, seq, seq)
            "num_heads": attentions.shape[1],
            "seq_length": len(tokens),
        }

    def visualize(
        self,
        text: str,
        *,
        layer: int = -1,
        head: int = 0,
        save_path: str | None = None,
    ):
        """
        Visualize attention as a heatmap.

        Args:
            text: Input text
            layer: Which layer
            head: Which attention head
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning(
                "matplotlib/seaborn not installed. "
                "Install with: pip install slmflow[observability]"
            )
            return

        data = self.get_attention(text, layer=layer, head=head)
        tokens = data["tokens"]
        attention = data["attention"][0]  # First head

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Blues",
            annot=False,
        )

        plt.title(f"Attention Pattern (Layer {layer}, Head {head})")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved to {save_path}")

        plt.show()

    def attention_rollout(
        self,
        text: str,
    ) -> dict:
        """
        Compute attention rollout across all layers.

        This shows which input tokens the model attends to across
        all layers, useful for understanding information flow.

        Args:
            text: Input text

        Returns:
            Dictionary with tokens and rollout attention
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
            )

        # Start with identity matrix
        attentions = outputs.attentions
        num_tokens = attentions[0].shape[-1]
        rollout = torch.eye(num_tokens).to(attentions[0].device)

        for attention in attentions:
            # Average over heads
            attention_heads_avg = attention.mean(dim=1)[0]  # (seq, seq)

            # Add residual connection
            attention_heads_avg = attention_heads_avg + torch.eye(num_tokens).to(attention.device)

            # Normalize
            attention_heads_avg = attention_heads_avg / attention_heads_avg.sum(
                dim=-1, keepdim=True
            )

            # Accumulate
            rollout = torch.matmul(attention_heads_avg, rollout)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return {
            "tokens": tokens,
            "rollout": rollout.cpu().numpy(),
        }


class TokenImportance:
    """
    Analyze token importance using gradient-based methods.
    """

    def __init__(self, model: Any, tokenizer: Any):
        """
        Initialize token importance analyzer.

        Args:
            model: The language model
            tokenizer: The tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer

    def compute_importance(
        self,
        text: str,
        target_position: int = -1,
    ) -> dict:
        """
        Compute importance of each input token for a prediction.

        Uses gradient-based attribution.

        Args:
            text: Input text
            target_position: Position to analyze (-1 for last)

        Returns:
            Dictionary with tokens and importance scores
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        # Get embeddings with gradient
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()

        # Forward pass
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=inputs["attention_mask"],
        )

        # Get logits at target position
        logits = outputs.logits[0, target_position, :]

        # Get predicted token
        predicted_idx = logits.argmax().item()

        # Backward pass
        logits[predicted_idx].backward()

        # Compute importance as gradient norm
        gradients = embeddings.grad[0]  # (seq, hidden)
        importance = gradients.norm(dim=-1).cpu().numpy()

        # Normalize
        importance = importance / importance.max()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        predicted_token = self.tokenizer.decode([predicted_idx])

        return {
            "tokens": tokens,
            "importance": importance.tolist(),
            "predicted_token": predicted_token,
            "target_position": target_position,
        }

    def visualize_importance(
        self,
        text: str,
        target_position: int = -1,
        save_path: str | None = None,
    ):
        """
        Visualize token importance as a bar chart.

        Args:
            text: Input text
            target_position: Position to analyze
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed")
            return

        data = self.compute_importance(text, target_position)

        plt.figure(figsize=(12, 4))
        colors = plt.cm.Reds(data["importance"])
        plt.bar(range(len(data["tokens"])), data["importance"], color=colors)
        plt.xticks(range(len(data["tokens"])), data["tokens"], rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title(f"Token Importance for Predicting: '{data['predicted_token']}'")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()
