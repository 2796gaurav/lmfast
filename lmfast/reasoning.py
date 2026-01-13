"""
Advanced Reasoning Module for LMFast.

Implements Test-Time Compute Scaling techniques:
- Best-of-N Sampling: Generate multiple solutions, pick best
- Chain-of-Thought (CoT): Enforce step-by-step reasoning
- Self-Verification: Model checks its own answers
- Majority Voting: Consensus from multiple generations
- Adaptive Compute: Allocate compute based on difficulty

Research shows that smaller models with optimized test-time compute
can match models **14x larger** on reasoning tasks.

Example:
    >>> from lmfast.reasoning import ThinkingAgent, reason
    >>>
    >>> # Quick usage
    >>> answer = reason(
    ...     model_fn=generate,
    ...     problem="What is 15% of 80?",
    ...     method="self_verify",
    ...     n=3
    ... )
"""

import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from lmfast.agents.core import Agent

logger = logging.getLogger(__name__)


@dataclass
class ReasoningConfig:
    """Configuration for reasoning strategies."""

    # Sampling
    n: int = 5  # Number of candidate solutions
    temperature: float = 0.7  # Sampling temperature

    # Self-verification
    max_verification_rounds: int = 3
    verification_threshold: float = 0.7

    # Adaptive compute
    min_n: int = 1
    max_n: int = 10
    difficulty_threshold: float = 0.5

    # CoT settings
    cot_prompt: str = "Let's think step by step."
    require_final_answer: bool = True


class ThinkingAgent(Agent):
    """
    Advanced reasoning agent implementing Test-Time Compute Scaling.

    Extends base Agent with sophisticated reasoning strategies that
    leverage additional inference compute to improve answer quality.

    Methods:
    - best_of_n: Generate N solutions, score, pick best
    - cot: Chain-of-Thought reasoning enforcement
    - self_verify: Generate, verify, correct loop
    - majority_vote: Consensus from multiple generations
    - adaptive: Dynamically adjust compute based on difficulty

    Example:
        >>> thinker = ThinkingAgent(generate_fn, n=5)
        >>> answer = thinker.reason(
        ...     "If 3x + 5 = 20, what is x?",
        ...     method="self_verify"
        ... )
    """

    def __init__(
        self,
        model_generate_fn: Callable[[str], str],
        n: int = 5,
        config: ReasoningConfig | None = None,
        tools: list | None = None,
    ):
        """
        Initialize ThinkingAgent.

        Args:
            model_generate_fn: Function that takes prompt, returns response
            n: Default number of candidate solutions
            config: Reasoning configuration
            tools: Optional tools for the agent
        """
        super().__init__(model_generate_fn, tools or [])
        self.n = n
        self.config = config or ReasoningConfig(n=n)

    def reason(
        self,
        problem: str,
        method: Literal[
            "best_of_n", "cot", "self_verify", "majority_vote", "adaptive"
        ] = "best_of_n",
        n: int | None = None,
        **kwargs,
    ) -> str:
        """
        Solve a problem using enhanced reasoning.

        Args:
            problem: The problem or question to solve
            method: Reasoning strategy to use
            n: Override default N for sampling methods
            **kwargs: Method-specific arguments

        Returns:
            Best answer based on the chosen method
        """
        effective_n = n or self.n
        logger.info(f"Reasoning with method: {method} (N={effective_n})")

        method_map = {
            "best_of_n": self._best_of_n,
            "cot": self._chain_of_thought,
            "self_verify": self._self_verify,
            "majority_vote": self._majority_vote,
            "adaptive": self._adaptive_reason,
        }

        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Available: {list(method_map.keys())}")

        return method_map[method](problem, effective_n, **kwargs)

    def _best_of_n(
        self, problem: str, n: int, scorer: Callable[[str], float] | None = None, **kwargs
    ) -> str:
        """
        Generate N solutions and pick the best one.

        Uses a scoring function to evaluate candidates.
        Default scorer uses a heuristic based on:
        - Answer length (longer reasoning often better)
        - Presence of numerical answers
        - Structured reasoning indicators
        """
        candidates = []
        prompt = f"Question: {problem}\n\n{self.config.cot_prompt}\n\nAnswer:"

        for i in range(n):
            try:
                sol = self.generate_fn(prompt)
                candidates.append(sol)
                logger.debug(f"Candidate {i+1}: {sol[:100]}...")
            except Exception as e:
                logger.warning(f"Generation {i+1} failed: {e}")

        if not candidates:
            return self.generate_fn(prompt)

        # Score candidates
        if scorer is None:
            scorer = self._default_scorer

        scored = [(sol, scorer(sol)) for sol in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Best score: {scored[0][1]:.2f}, Worst: {scored[-1][1]:.2f}")
        return scored[0][0]

    def _default_scorer(self, solution: str) -> float:
        """
        Default heuristic scorer for solutions.

        Higher scores for:
        - Longer reasoning chains
        - Presence of numerical answers
        - Step-by-step structure
        - Confidence indicators
        """
        score = 0.0

        # Length bonus (normalized)
        score += min(len(solution) / 500, 1.0) * 0.3

        # Numerical answer bonus
        if re.search(r"\d+\.?\d*", solution):
            score += 0.2

        # Step indicators bonus
        step_indicators = ["step", "first", "then", "next", "finally", "therefore", "so"]
        for indicator in step_indicators:
            if indicator.lower() in solution.lower():
                score += 0.05

        # Structure bonus (numbered steps, bullet points)
        if re.search(r"^\d+[\.\)]\s", solution, re.MULTILINE):
            score += 0.15

        # Conclusion bonus
        if any(c in solution.lower() for c in ["answer is", "result is", "equals", "therefore"]):
            score += 0.1

        return min(score, 1.0)

    def _chain_of_thought(self, problem: str, n: int = 1, **kwargs) -> str:
        """
        Enforce Chain-of-Thought reasoning.

        Prompts the model to reason explicitly before answering.
        """
        prompt = f"""Question: {problem}

{self.config.cot_prompt}

Work through this problem step by step:
1. First, identify what we're asked to find.
2. Then, break down the problem into smaller parts.
3. Solve each part carefully.
4. Finally, combine to get the answer.

Solution:"""

        if n > 1:
            # Generate multiple and pick best
            return self._best_of_n(problem, n, **kwargs)

        response = self.generate_fn(prompt)

        # Try to extract final answer if configured
        if self.config.require_final_answer:
            response = self._ensure_final_answer(response, problem)

        return response

    def _ensure_final_answer(self, response: str, problem: str) -> str:
        """Ensure response ends with a clear final answer."""
        # Check if there's already a clear answer
        if any(
            indicator in response.lower()
            for indicator in ["the answer is", "answer:", "result:", "therefore,"]
        ):
            return response

        # Ask for clarification
        followup = f"""Based on your reasoning:
{response}

What is the final answer to: {problem}

Final Answer:"""

        final = self.generate_fn(followup)
        return f"{response}\n\nFinal Answer: {final}"

    def _self_verify(self, problem: str, n: int, **kwargs) -> str:
        """
        Self-verification loop.

        1. Generate initial solution
        2. Ask model to verify correctness
        3. If incorrect, ask for correction
        4. Repeat until confident or max rounds
        """
        solution = self._chain_of_thought(problem, n=1)

        for round_num in range(self.config.max_verification_rounds):
            # Verification prompt
            verify_prompt = f"""Problem: {problem}

Proposed Solution:
{solution}

Please verify if this solution is correct. Check each step carefully.

If the solution is CORRECT, respond with: "VERIFIED: The solution is correct."
If the solution is INCORRECT, respond with: "ERROR: [explain the error]" followed by "CORRECTED SOLUTION: [your corrected solution]"

Verification:"""

            verification = self.generate_fn(verify_prompt)

            # Parse verification
            if "VERIFIED" in verification.upper():
                logger.info(f"Solution verified after {round_num + 1} rounds")
                return solution

            elif "CORRECTED" in verification.upper():
                # Extract corrected solution
                if "CORRECTED SOLUTION:" in verification.upper():
                    parts = verification.upper().split("CORRECTED SOLUTION:")
                    if len(parts) > 1:
                        # Get the actual text (original case)
                        idx = verification.upper().find("CORRECTED SOLUTION:")
                        solution = verification[idx + len("CORRECTED SOLUTION:") :].strip()
                        logger.info(f"Solution corrected in round {round_num + 1}")
                else:
                    # Just use the verification as new solution
                    solution = verification
            else:
                # No clear signal, try best-of-n
                logger.info("Unclear verification, generating new candidates")
                return self._best_of_n(problem, n)

        logger.warning(f"Max verification rounds ({self.config.max_verification_rounds}) reached")
        return solution

    def _majority_vote(self, problem: str, n: int, **kwargs) -> str:
        """
        Majority voting across multiple generations.

        Generates N solutions, extracts final answers, returns most common.
        """
        answers = []
        full_solutions = []

        prompt = f"Question: {problem}\n\n{self.config.cot_prompt}\n\nAnswer:"

        for _ in range(n):
            solution = self.generate_fn(prompt)
            full_solutions.append(solution)

            # Extract final answer (numbers or last sentence)
            answer = self._extract_answer(solution)
            answers.append(answer)

        # Find majority
        if answers:
            counter = Counter(answers)
            majority_answer, count = counter.most_common(1)[0]
            logger.info(f"Majority vote: '{majority_answer}' ({count}/{n} votes)")

            # Return full solution that matches majority
            for sol, ans in zip(full_solutions, answers):
                if ans == majority_answer:
                    return sol

        # Fallback to first solution
        return full_solutions[0] if full_solutions else self.generate_fn(prompt)

    def _extract_answer(self, solution: str) -> str:
        """Extract the final answer from a solution."""
        # Try to find explicit answer markers
        patterns = [
            r"the answer is[:\s]+(.+?)(?:\.|$)",
            r"answer[:\s]+(.+?)(?:\.|$)",
            r"result[:\s]+(.+?)(?:\.|$)",
            r"=\s*(.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, solution.lower())
            if match:
                return match.group(1).strip()

        # Fall back to last number found
        numbers = re.findall(r"\d+\.?\d*", solution)
        if numbers:
            return numbers[-1]

        # Fall back to last sentence
        sentences = solution.split(".")
        return sentences[-1].strip() if sentences else solution

    def _adaptive_reason(self, problem: str, n: int, **kwargs) -> str:
        """
        Adaptive compute allocation.

        Estimates problem difficulty and allocates compute accordingly:
        - Easy problems → single CoT generation
        - Medium problems → best-of-N
        - Hard problems → self-verify with high N
        """
        # Estimate difficulty
        difficulty = self._estimate_difficulty(problem)
        logger.info(f"Estimated difficulty: {difficulty:.2f}")

        # Adapt compute
        if difficulty < 0.3:
            # Easy - simple CoT
            return self._chain_of_thought(problem, n=1)
        elif difficulty < 0.6:
            # Medium - best of N
            adaptive_n = max(self.config.min_n, int(n * difficulty * 2))
            return self._best_of_n(problem, adaptive_n)
        else:
            # Hard - full self-verification
            adaptive_n = min(self.config.max_n, int(n * (1 + difficulty)))
            return self._self_verify(problem, adaptive_n)

    def _estimate_difficulty(self, problem: str) -> float:
        """
        Estimate problem difficulty using heuristics.

        Returns 0.0 (easy) to 1.0 (hard).
        """
        difficulty = 0.3  # Base difficulty

        # Length indicator
        word_count = len(problem.split())
        difficulty += min(word_count / 100, 0.2)

        # Math indicators
        if re.search(r"[+\-*/=]", problem):
            difficulty += 0.1
        if re.search(r"\d{3,}", problem):  # Large numbers
            difficulty += 0.1

        # Complexity keywords
        hard_keywords = [
            "optimize",
            "prove",
            "derive",
            "analyze",
            "complex",
            "multiple",
            "constraint",
            "maximize",
            "minimize",
        ]
        for keyword in hard_keywords:
            if keyword in problem.lower():
                difficulty += 0.05

        # Multi-step indicators
        if re.search(r"(first|then|after|before|finally)", problem.lower()):
            difficulty += 0.1

        return min(difficulty, 1.0)


def reason(
    model_fn: Callable[[str], str], problem: str, method: str = "best_of_n", n: int = 5, **kwargs
) -> str:
    """
    One-line reasoning with test-time compute scaling.

    Args:
        model_fn: Generation function (prompt -> response)
        problem: Problem to solve
        method: Reasoning method (best_of_n, cot, self_verify, majority_vote, adaptive)
        n: Number of candidates for sampling methods
        **kwargs: Additional method arguments

    Returns:
        Best answer

    Example:
        >>> from lmfast import reason
        >>> answer = reason(
        ...     model_fn=my_model.generate,
        ...     problem="What is 25% of 80?",
        ...     method="self_verify",
        ...     n=3
        ... )
    """
    agent = ThinkingAgent(model_fn, n=n)
    return agent.reason(problem, method=method, **kwargs)


# Specialized reasoning functions
def solve_math(model_fn: Callable[[str], str], problem: str, n: int = 5) -> str:
    """Solve math problems with self-verification."""
    return reason(model_fn, problem, method="self_verify", n=n)


def solve_with_consensus(model_fn: Callable[[str], str], problem: str, n: int = 7) -> str:
    """Solve problems using majority voting."""
    return reason(model_fn, problem, method="majority_vote", n=n)


def solve_adaptive(model_fn: Callable[[str], str], problem: str) -> str:
    """Solve problems with adaptive compute allocation."""
    return reason(model_fn, problem, method="adaptive", n=5)
