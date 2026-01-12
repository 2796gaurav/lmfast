
import pytest
from unittest.mock import MagicMock
from lmfast.agents.core import Agent, Tool
from lmfast.agents.specialized import CodeAgent

def test_tool_creation():
    """Test creating a tool from function."""
    def sample_tool(a: int, b: str) -> str:
        """Sample description."""
        return f"{a}-{b}"
        
    tool = Tool.from_function(sample_tool)
    assert tool.name == "sample_tool"
    assert tool.description == "Sample description."
    assert tool.parameters['a'] == "<class 'int'>"
    assert tool.parameters['b'] == "<class 'str'>"
    
    result = tool.run(a=1, b="test")
    assert result == "1-test"

def test_agent_loop_no_tool():
    """Test basic agent loop without tools."""
    mock_gen = MagicMock(return_value="Hello world")
    agent = Agent(model_generate_fn=mock_gen)
    
    response = agent.run("Hi")
    assert response == "Hello world"
    mock_gen.assert_called_once()
    assert "Hi" in mock_gen.call_args[0][0]

def test_agent_loop_with_tool():
    """Test agent loop identifying and calling a tool."""
    def calculator(x: int, y: int) -> int:
        return x + y
        
    # Mock LLM behavior: 
    # 1st call: Returns JSON to call tool
    # 2nd call: Returns final answer after seeing tool output
    mock_gen = MagicMock(side_effect=[
        '{"tool": "calculator", "args": {"x": 5, "y": 3}}',
        "The answer is 8."
    ])
    
    agent = Agent(model_generate_fn=mock_gen, tools=[calculator])
    
    response = agent.run("Calculate 5 + 3")
    
    assert response == "The answer is 8."
    assert mock_gen.call_count == 2
    
def test_code_agent():
    """Test CodeAgent initialization."""
    agent = CodeAgent(model_generate_fn=MagicMock())
    assert "execute_python" in agent.tools
    assert "You are an expert Python coding assistant" in agent.system_prompt
