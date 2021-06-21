from dataclasses import dataclass


@dataclass
class EvalConfig:
    dataset: str = 'kspon'
    dataset_path: str = ''
    transcripts_path: str = './data/eval_transcript.txt'
    model_path: str = ''
    output_unit: str = 'character'
    batch_size: int = 32
    num_workers: int = 4
    print_every: int = 20
    decode: str = 'greedy'
    k: int = 3
    use_cuda: bool = True