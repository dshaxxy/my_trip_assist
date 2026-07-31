import os

def get_project_root():
    """
    获取项目根目录
    :return:
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(current_dir)
    return project_root


def get_abs_path(relative_path: str):
    """
    获取绝对路径
    :param relative_path: 相对路径
    :return: 绝对路径
    """
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path

if __name__ == '__main__':
    print(get_project_root())
    print(get_abs_path("rag/knowledge_base/202308641.pdf"))
