import yaml
from my_hello_agents.utils.path_tool import get_abs_path

def load_prompts_config(config_path: str = get_abs_path("config/prompts.yaml"), encoding="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

prompts_config = load_prompts_config()

if __name__ == '__main__':
    print(prompts_config["summary_prompt_path"])
