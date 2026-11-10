"""The 12 published Qwen3.5 method/data/checkpoint identities (IDs 19–30)."""

from dataclasses import dataclass


MAP_35B = (0, 1, 3, 3, 5, 6, 7, 7, 10, 11, 13, 15, 15, 16, 18, 19,
           20, 21, 23, 23, 25, 26, 27, 27, 30, 31, 33, 35, 35, 36, 38, 39)
MAP_397B = (0, 2, 4, 7, 8, 10, 11, 11, 15, 17, 19, 19, 23, 25, 27, 27,
            30, 32, 34, 35, 38, 40, 42, 43, 46, 48, 49, 51, 53, 55, 57, 59)
MAP_4B = tuple(range(32))


@dataclass(frozen=True)
class Experiment:
    identifier: int
    slug: str
    task: str
    method: str
    sender: str | None
    steps: int
    learning_rate: float
    layer_map: tuple[int, ...] = MAP_4B
    teacher_dim: int = 2560
    validation_steps: tuple[int, ...] = ()

    @property
    def tag(self):
        return f'{self.identifier:02d}_{self.slug}'

    @property
    def paper_role(self):
        if self.method == 'bridge':
            return 'Self-Communication' if self.sender == 'Qwen3.5-4B' else 'PLatour'
        if self.method == 'random':
            return 'Virtual Sender'
        if self.method in ('full', 'lora'):
            return 'Trained Receiver'
        raise ValueError('Unknown paper-role training method')

    @property
    def val_samples(self):
        return 4 if self.task == 'dapo' else 16

    @property
    def train_questions(self):
        return 16891 if self.task == 'dapo' else 18873

    @property
    def val_questions(self):
        return 500 if self.task == 'dapo' else 201

    @property
    def checkpoint_steps(self):
        return (0, *self.validation_steps)


MATH_EVAL = (33, 66, 99, 132, 165, 198, 231, 264)
CODE_EVAL = (42, 84, 126, 168, 210, 252, 295)
CODE_EVAL_WITH_294 = (42, 84, 126, 168, 210, 252, 294, 295)

EXPERIMENTS = (
    Experiment(19, 'qwen35_35b_to_4b_dapo', 'dapo', 'bridge', 'Qwen3.5-35B-A3B',
               264, 1e-5, MAP_35B, 2048, MATH_EVAL),
    Experiment(20, 'qwen35_397b_fp8_to_4b_dapo', 'dapo', 'bridge',
               'Qwen3.5-397B-A17B-FP8', 264, 1e-5, MAP_397B, 4096, MATH_EVAL),
    Experiment(21, 'qwen35_4b_to_4b_dapo', 'dapo', 'bridge', 'Qwen3.5-4B',
               264, 1e-5, MAP_4B, 2560, MATH_EVAL),
    Experiment(22, 'qwen35_random4b_to_4b_dapo', 'dapo', 'random', None,
               264, 1e-5, MAP_4B, 2560, MATH_EVAL),
    Experiment(23, 'qwen35_4b_full_grpo_dapo', 'dapo', 'full', None,
               264, 1e-6, MAP_4B, 2560, MATH_EVAL),
    Experiment(24, 'qwen35_4b_lora_grpo_dapo', 'dapo', 'lora', None,
               264, 1e-5, MAP_4B, 2560, MATH_EVAL),
    Experiment(25, 'qwen35_397b_fp8_to_4b_deepcoder', 'deepcoder', 'bridge',
               'Qwen3.5-397B-A17B-FP8', 295, 1e-5, MAP_397B, 4096, CODE_EVAL),
    Experiment(26, 'qwen35_35b_to_4b_deepcoder', 'deepcoder', 'bridge',
               'Qwen3.5-35B-A3B', 295, 1e-5, MAP_35B, 2048, CODE_EVAL_WITH_294),
    Experiment(27, 'qwen35_4b_to_4b_deepcoder', 'deepcoder', 'bridge', 'Qwen3.5-4B',
               295, 1e-5, MAP_4B, 2560, CODE_EVAL),
    Experiment(28, 'qwen35_random4b_to_4b_deepcoder', 'deepcoder', 'random', None,
               295, 1e-5, MAP_4B, 2560, CODE_EVAL_WITH_294),
    Experiment(29, 'qwen35_4b_full_grpo_deepcoder', 'deepcoder', 'full', None,
               295, 1e-6, MAP_4B, 2560, CODE_EVAL),
    Experiment(30, 'qwen35_4b_lora_grpo_deepcoder', 'deepcoder', 'lora', None,
               295, 1e-5, MAP_4B, 2560, CODE_EVAL),
)


def select(identifier: int) -> Experiment:
    return next(item for item in EXPERIMENTS if item.identifier == identifier)
