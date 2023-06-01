import pytest

from autogpt.agent.agent_manager import AgentManager
from autogpt.llm.chat import create_chat_completion


@pytest.fixture
def agent_manager():
    # Hack, real gross. Singletons are not good times.
    yield AgentManager()
    del AgentManager._instances[AgentManager]


@pytest.fixture
def task():
    return "translate English to French"


@pytest.fixture
def prompt():
    return "Translate the following English text to French: 'Hello, how are you?'"


@pytest.fixture
def model():
    return "gpt-3.5-turbo"


@pytest.fixture(autouse=True)
def mock_create_chat_completion(mocker):
    mock_create_chat_completion = mocker.patch(
        "autogpt.agent.agent_manager.create_chat_completion",
        wraps=create_chat_completion,
    )
    mock_create_chat_completion.return_value = "irrelevant"
    return mock_create_chat_completion


def test_create_agent(agent_manager: AgentManager, task, prompt, model):
    key, agent_reply = agent_manager.create_agent(task, prompt, model)
    assert isinstance(key, int)
    assert isinstance(agent_reply, str)
    assert key in agent_manager.agents


def test_message_agent(agent_manager: AgentManager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    user_message = "Please translate 'Good morning' to French."
    agent_reply = agent_manager.message_agent(key, user_message)
    assert isinstance(agent_reply, str)


def test_list_agents(agent_manager: AgentManager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    agents_list = agent_manager.list_agents()
    assert isinstance(agents_list, list)
    assert (key, task) in agents_list


def test_delete_agent(agent_manager: AgentManager, task, prompt, model):
    key, _ = agent_manager.create_agent(task, prompt, model)
    success = agent_manager.delete_agent(key)
    assert success
    assert key not in agent_manager.agents

def test_invalid_key_message_agent(agent_manager):
    # Attempt to send a message to an invalid key
    invalid_key = 999
    user_message = "Test message"
    with pytest.raises(KeyError):
        agent_reply = agent_manager.message_agent(invalid_key, user_message)

def test_multiple_agents(agent_manager, task, prompt, model):
    # Test create multiple agents and verify they are in the list_agents() output
    key1, _ = agent_manager.create_agent(task, prompt, model)
    key2, _ = agent_manager.create_agent(task, prompt, model)
    agents_list = agent_manager.list_agents()
    assert (key1, task) in agents_list
    assert (key2, task) in agents_list
    assert len(agents_list) == 2

    # Test delete one agent and assert that it is removed from the list_agents() output
    agent_manager.delete_agent(key1)
    agents_list = agent_manager.list_agents()
    assert (key1, task) not in agents_list
    assert (key2, task) in agents_list
    assert len(agents_list) == 1

def test_delete_nonexistent_agent(agent_manager):
    # Test attempting to delete a nonexistent agent returns False
    nonexistent_key = 999
    success = agent_manager.delete_agent(nonexistent_key)
    assert not success   
