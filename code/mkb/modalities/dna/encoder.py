"""
DNA encoder: DNAConvFormer — same architecture as RNAConvFormer but as a
separate ``nn.Module`` instance with its own parameters.

We inherit from ``RNAConvFormer`` and forward a small adapter that maps
``Qwen3VLDNAConfig`` field names (``dna_encoder_hidden_size``,
``dna_vocab_size`` …) onto the names ``RNAConvFormer.__init__`` expects.
This avoids duplicating the ~230 lines of conv stem + transformer +
resampler code while keeping DNA's state-dict path independent
(``model.modality_router.encoders.dna.*``).
"""

from mkb.modalities.rna.encoder import RNAConvFormer


class _DNAToRNAConfigAdapter:
    """Tiny shim that exposes a DNA config under the field names that
    ``RNAConvFormer.__init__`` reads.

    Only the attributes accessed inside ``RNAConvFormer.__init__`` are
    forwarded; the rest of the config is preserved on ``self.dna_config``
    in case future helpers need it.
    """

    def __init__(self, dna_config):
        self.dna_config = dna_config
        # RNAConvFormer reads:
        #   config.rna_encoder_hidden_size
        #   config.rna_vocab_size
        #   config.rna_max_seq_length
        #   config.conv_kernel_size
        #   config.num_attention_heads
        #   config.num_encoder_layers
        #   config.num_latent_tokens
        #   config.dropout
        self.rna_encoder_hidden_size = dna_config.dna_encoder_hidden_size
        self.rna_vocab_size = dna_config.dna_vocab_size
        self.rna_max_seq_length = dna_config.dna_max_seq_length
        self.conv_kernel_size = dna_config.conv_kernel_size
        self.num_attention_heads = dna_config.num_attention_heads
        self.num_encoder_layers = dna_config.num_encoder_layers
        self.num_latent_tokens = dna_config.num_latent_tokens
        self.dropout = dna_config.dropout


class DNAConvFormer(RNAConvFormer):
    """Independent DNA encoder. Architecture is byte-identical to
    ``RNAConvFormer`` (conv stem + Transformer + perceiver resampler) but
    weights are not shared.
    """

    def __init__(self, config):
        super().__init__(_DNAToRNAConfigAdapter(config))


# Optional alias kept for symmetry with ``Qwen3VLRNAEncoder``.
Qwen3VLDNAEncoder = DNAConvFormer
