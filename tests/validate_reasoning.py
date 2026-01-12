
import logging
import sys
import unittest
from unittest.mock import MagicMock

sys.path.append("/home/gaurav/small-idea")
from lmfast.reasoning import ThinkingAgent, reason

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestReasoning(unittest.TestCase):
    def setUp(self):
        # Mock generator that acts intelligent enough for testing logic
        self.mock_responses = {
            "default": "The answer is 42.",
            "cot": "Let's think step by step. First we add 2 and 2. The result is 4. Answer: 4.",
            "verify_correct": "VERIFIED: The solution is correct.",
            "verify_incorrect": "ERROR: calculation wrong. CORRECTED SOLUTION: 42",
        }
        
    def mock_generate(self, prompt, **kwargs):
        # tailored responses based on prompt content
        if "step by step" in prompt.lower() and "verify" not in prompt.lower():
            return "Step 1: Analyze problem. Step 2: Calculate. Answer: 42 " + prompt[-10:]
        elif "verify" in prompt.lower():
            if "Solution" in prompt: 
                return "VERIFIED: The solution is correct."
            return "VERIFIED"
        return "The answer is 42."

    def test_best_of_n_scorer(self):
        logger.info("Testing Best of N Scorer...")
        agent = ThinkingAgent(self.mock_generate, n=3)
        
        # Candidate 1: Short, no structure
        c1 = "42" 
        # Candidate 2: Long, structured
        c2 = "Step 1: 20+20 = 40. Step 2: 40+2 = 42. Therefore, result is 42."
        
        score1 = agent._default_scorer(c1)
        score2 = agent._default_scorer(c2)
        
        logger.info(f"Score 1: {score1}, Score 2: {score2}")
        self.assertGreater(score2, score1, "Structured reasoning should score higher")

    def test_thinking_agent_flow(self):
        logger.info("Testing ThinkingAgent Flow...")
        agent = ThinkingAgent(self.mock_generate, n=2)
        
        # Test default (best_of_n)
        ans = agent.reason("What is life?", method="best_of_n")
        self.assertTrue("42" in ans or "Answer" in ans)
        
        # Test CoT
        ans_cot = agent.reason("Math problem", method="cot")
        self.assertTrue("Step" in ans_cot or "42" in ans_cot)

    def test_self_verify(self):
        logger.info("Testing Self-Verification...")
        agent = ThinkingAgent(self.mock_generate, n=1)
        
        # Our mock generator always verifies as correct
        ans = agent.reason("Complex problem", method="self_verify")
        self.assertIsNotNone(ans)

if __name__ == "__main__":
    unittest.main()
