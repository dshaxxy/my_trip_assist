from utils.config_handler import prompts_config
from utils.path_tool import get_abs_path


def load_summary_prompt():
    try:
        summary_prompt_path = get_abs_path(prompts_config["summary_prompt_path"])
    except KeyError:
        raise KeyError
    try:
        return open(summary_prompt_path, "r", encoding="utf-8").read()
    except Exception:
        raise Exception

def load_preference_prompt():
    try:
        preference_prompt_path = get_abs_path(prompts_config["preference_prompt_path"])
    except KeyError:
        raise KeyError
    try:
        return open(preference_prompt_path, "r", encoding="utf-8").read()
    except Exception:
        raise Exception

def load_extract_prompt():
    try:
        extract_prompt_path = get_abs_path(prompts_config["extract_prompt_path"])
    except KeyError:
        raise KeyError
    try:
        return open(extract_prompt_path, "r", encoding="utf-8").read()
    except Exception:
        raise Exception

def load_system_prompt():
    try:
        system_prompt_path = get_abs_path(prompts_config["system_prompt_path"])
    except KeyError:
        raise KeyError
    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception:
        raise Exception

if __name__ == '__main__':
    print(load_summary_prompt())