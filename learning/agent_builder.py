import yaml


from pg.pg_agent import PGAgent   # adjust import based on where you store pg_agent.py

def build_agent(agent_file, env, device):
    agent_config = load_agent_file(agent_file)
    agent_name = agent_config["agent_name"]
    if agent_name != PGAgent.NAME:
        raise AssertionError(f"Unsupported agent: {agent_name}")
    return PGAgent(config=agent_config, env=env, device=device)

def load_agent_file(file):
    with open(file, "r") as stream:
        return yaml.safe_load(stream)
