import torch
import torch.nn as nn


class TransducerLoss(nn.Module):
    """
    Transducer loss module.
    Args:
        blank_id (int): blank symbol id
    """

    def __init__(self, blank_id: int) -> None:
        """Construct an TransLoss object."""
        super().__init__()
        try:
            from warp_rnnt import rnnt_loss
        except ImportError:
            raise ImportError("warp-rnnt is not installed. Please re-setup")
            
        self.rnnt_loss = rnnt_loss
        self.blank_id = blank_id

    def forward(
            self,
            log_probs: torch.FloatTensor,
            targets: torch.IntTensor,
            input_lengths: torch.IntTensor,
            target_lengths: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Compute path-aware regularization transducer loss.
        Args:
            log_probs (torch.FloatTensor): Batch of predicted sequences (batch, maxlen_in, maxlen_out+1, odim)
            targets (torch.IntTensor): Batch of target sequences (batch, maxlen_out)
            input_lengths (torch.IntTensor): batch of lengths of predicted sequences (batch)
            target_lengths (torch.IntTensor): batch of lengths of target sequences (batch)
        Returns:
            loss (torch.FloatTensor): transducer loss
        """

        return self.rnnt_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            reduction="mean",
            blank=self.blank_id,
            gather=True,
        )