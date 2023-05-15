from tianshou.trainer import offline_trainer,OfflineTrainer
from typing import Any, Callable, DefaultDict, Dict, Optional, Tuple, Union

class OfflineTrainersave(OfflineTrainer):
    def __init__(self,*args, **kwargs):
        super(OfflineTrainersave,self).__init__(*args, **kwargs)


    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        super().__next__()
        if self.epoch % 10 == 0:
            if self.save_best_fn:
                self.save_best_fn(self.policy,'epoch'+str(self.epoch))
def offline_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for offline_trainer run method.

    It is identical to ``OfflineTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return OfflineTrainersave(*args, **kwargs).run()


offline_trainer_iter = OfflineTrainersave