
import torch.utils.data
import os
def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


def get_workspace():
    """
    get the workspace path, i.e., the root directory of the project
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    return file
ws =  get_workspace()



def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

