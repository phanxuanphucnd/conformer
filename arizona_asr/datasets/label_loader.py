# -*- coding: utf-8 -*-

from typing import Tuple

def load_dataset(transcripts_path: str) -> Tuple[list, list]:
    """
    Provides dictionary of filename and labels
    Args:
        transcripts_path (str): path of transcripts
    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    audio_paths = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, vi_transcript, transcript = line.split('\t')
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts
