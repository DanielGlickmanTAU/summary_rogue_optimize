from dataclasses import dataclass


@dataclass
class RankingDatasetConfig:
    num_examples: int = 1
    num_skip: int = 2
    num_summaries_per_text: int = 2
    max_seq_len = None


@dataclass
class RankerConfig(RankingDatasetConfig):
    learning_rate: float = 5e-3
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 25000
    # half_percision = compute.get_torch().cuda.is_available()
    half_percision: bool = False
    do_evaluation: bool = True
    use_dropout: bool = True  # use dropout in training
    print_logits: bool = False
    evaluate_every_steps: int = None

    def get_evaluation_strategy(self):
        if self.evaluate_every_steps is None and self.do_evaluation is None:
            return 'no'

        if self.evaluate_every_steps is not None:
            return 'steps'
        return 'epoch'
