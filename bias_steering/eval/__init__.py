from .winogenerated import Winogenerated
from .occupation import OccupationStereotypes

def load_evaluation_task(task):
    if task == "winogenerated":
        return Winogenerated()
    elif task == "occupational_stereotypes":
        return OccupationStereotypes()
    else:
        raise Exception("Requested dataset does not exist.")