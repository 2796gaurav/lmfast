
import pytest
from unittest.mock import MagicMock, patch
from lmfast.reasoning import ThinkingAgent

def test_thinking_agent():
    """Test thinking agent dynamics."""
    mock_gen = MagicMock(side_effect=["Sol A (short)", "Solution B (longer reasoning)"])
    agent = ThinkingAgent(mock_gen, n=2)
    
    # best_of_n should pick the longer one by our heuristic
    result = agent.reason("solve x", method="best_of_n")
    
    assert result == "Solution B (longer reasoning)"
    assert mock_gen.call_count == 2 # Called twice

def test_alignment_import():
    """Test that we can import alignment via lazy loading."""
    import lmfast
    try:
        # Should not raise AttributeError
        _ = lmfast.align
    except Exception as e:
        pytest.fail(f"Could not import align: {e}")
