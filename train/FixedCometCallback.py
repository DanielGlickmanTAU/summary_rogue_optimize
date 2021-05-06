from experiments import experiment as exp
import comet_ml
from transformers.integrations import CometCallback


class FixedCometCallback(CometCallback):
    def __init__(self, config):
        super(FixedCometCallback, self).__init__()
        self.config = config

    def setup(self, args, state, model):

        self._initialized = True
        if state.is_world_process_zero:
            experiment = comet_ml.config.get_global_experiment()
            if experiment is None:
                experiment = exp.start_experiment(hyperparams=self.config)

            experiment._set_model_graph(model, framework="transformers")
            experiment._log_parameters(args, prefix="args/", framework="transformers")
            if hasattr(model, "config"):
                experiment._log_parameters(model.config, prefix="config/", framework="transformers")
