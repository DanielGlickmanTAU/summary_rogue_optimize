from dataclasses import dataclass


@dataclass
class RankingDatasetConfig:
    validation_mapped_saved_path: str = 'sshleifer_distilbart-xsum-12-3/processed_dataset__validation_xsum10000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
    train_mapped_saved_path: str = 'sshleifer_distilbart-xsum-12-3/processed_dataset__train_xsum50000_do_sampleFalse_top_pNone_top_kNone_num_beams8_num_return_sequences8_no_repeat_ngram_size0'
    num_examples: int = 50_000
    num_skip: int = 0
    num_summaries_per_text: int = 4
    max_seq_len: int = 512
    test_mapped_saved_path: str = None
    metric_for_best_model: str = None


@dataclass
class RankerConfig(RankingDatasetConfig):
    loss_fn: str = 'mse'
    tolerance: float = None
    learning_rate: float = 5e-3
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 10
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
