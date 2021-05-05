from dataclasses import dataclass


@dataclass
class RankerConfig:
    num_examples: int = 1
    num_skip: int = 2
    num_beams: int = 2
    learning_rate: float = 5e-3
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 25000
    # half_percision = compute.get_torch().cuda.is_available()
    half_percision: bool = False
    do_evaluation: bool = True
    use_dropout: bool = True  # use dropout in training
    print_logits: bool = True
