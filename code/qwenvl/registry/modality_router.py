"""
ModalityRouter: the central dispatch layer for all non-vision modalities.

Replaces the hard-coded if/elif branches in model forward() with a
generic loop over registered modalities.  Each modality registers its
encoder, projector, and optional decoder head once; the router handles
encode → project → scatter → dummy-gradient bookkeeping automatically.

Usage in model __init__:
    self.modality_router = ModalityRouter()
    self.modality_router.register_modality(
        "rna",
        encoder=RNAConvFormer(config),
        projector=Qwen3VLRNAProjector(config, llm_hidden),
        decoder=RNALMDecoder(llm_hidden),
        pad_token_ids=[rna_pad_id, dna_pad_id],
        is_image_like=True,
    )

Usage in model forward:
    active, image_like_grids = self.modality_router.scatter_all(
        input_ids, inputs_embeds, bio_token_ids, **kwargs
    )
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ModalityRouter(nn.Module):
    """Dispatches modality inputs to their registered encoder + projector.

    Stores raw encoder/projector/decoder modules directly (not wrapped in
    BaseEncoder/BaseProjector) so that ``state_dict`` paths remain clean
    and backward-compatible with old checkpoints after key remapping.
    """

    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        # Non-parameter metadata (not saved in state_dict)
        self._is_image_like: Dict[str, bool] = {}
        self._aliases: Dict[str, str] = {}  # alias → canonical name
        # Cached trainable flags — invalidated on register_modality()
        self._trainable_cache: Dict[str, Tuple[bool, bool]] = {}  # name → (enc, proj)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_modality(
        self,
        name: str,
        encoder: Optional[nn.Module] = None,
        projector: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        is_image_like: bool = True,
        aliases: Optional[List[str]] = None,
    ):
        """Register a modality with its components.

        Args:
            name: Canonical modality name (e.g. "rna", "protein").
            encoder: The encoder nn.Module.  Called as
                     ``encoder(input_ids, attention_mask) → (latent, mask)``.
            projector: Projects encoder output to LLM hidden dim.
                       Called as ``projector(latent) → projected``.
            decoder: Optional decoder head (e.g. for sequence generation).
                     Must implement ``decode(hidden)``, ``compute_loss()``,
                     and ``logits_to_vocab_space()`` (see BaseDecoder).
            is_image_like: If True, this modality's grid_thw is concatenated
                           with image_grid_thw for RoPE (1D sequence modalities).
                           If False, treated as video-like (2D).
            aliases: Other modality names that share this encoder/projector
                     (e.g. ["dna"] for the RNA encoder that handles both).
        """
        if encoder is not None:
            self.encoders[name] = encoder
        if projector is not None:
            self.projectors[name] = projector
        if decoder is not None:
            self.decoders[name] = decoder
        self._is_image_like[name] = is_image_like
        self._trainable_cache.clear()
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
        logger.info(
            f"[ModalityRouter] Registered '{name}': "
            f"encoder={'yes' if encoder else 'no'}, "
            f"projector={'yes' if projector else 'no'}, "
            f"decoder={'yes' if decoder else 'no'}, "
            f"aliases={aliases or []}"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> str:
        """Resolve an alias to its canonical modality name."""
        return self._aliases.get(name, name)

    @property
    def modality_names(self) -> List[str]:
        """All canonical modality names (not aliases)."""
        return list(self.encoders.keys())

    @property
    def all_names(self) -> List[str]:
        """All names including aliases."""
        return list(self.encoders.keys()) + list(self._aliases.keys())

    def has_encoder(self, name: str) -> bool:
        return self.resolve(name) in self.encoders

    def has_decoder(self, name: str) -> bool:
        return self.resolve(name) in self.decoders

    def get_decoder(self, name: str) -> Optional[nn.Module]:
        # nn.ModuleDict has no .get() — use explicit membership check.
        resolved = self.resolve(name)
        return self.decoders[resolved] if resolved in self.decoders else None

    def is_image_like(self, name: str) -> bool:
        return self._is_image_like.get(self.resolve(name), True)

    def clear_caches(self):
        """Clear cached trainable flags. Call after freeze policy changes."""
        self._trainable_cache.clear()

    # ------------------------------------------------------------------
    # Core: encode + project
    # ------------------------------------------------------------------

    def encode_and_project(
        self,
        name: str,
        input_ids: Tensor,
        attention_mask: Tensor,
        **extra_kwargs,
    ) -> Tensor:
        """Run encoder + projector for a modality.

        Args:
            name: Modality name (can be an alias).
            input_ids: [num_sequences, max_seq_len]
            attention_mask: [num_sequences, max_seq_len]
            **extra_kwargs: Additional keyword arguments passed to the encoder
                (e.g. edge_index, edge_attr for graph-based encoders).
                Ignored by sequence-based encoders (RNA, protein).

        Returns:
            Flat projected embeddings [total_tokens, D_llm] ready for
            masked_scatter into the LLM input embeddings.
        """
        actual = self.resolve(name)
        encoder = self.encoders[actual]

        # Frozen encoders: skip autograd graph to avoid wasting memory
        # and backward compute (e.g. ESM3's 48 transformer layers).
        # Projector still gets gradients via the LLM backward path.
        if actual not in self._trainable_cache:
            self._trainable_cache[actual] = (
                any(p.requires_grad for p in encoder.parameters()),
                actual in self.projectors and any(
                    p.requires_grad for p in self.projectors[actual].parameters()
                ),
            )
        has_trainable_enc = self._trainable_cache[actual][0]

        if has_trainable_enc:
            latent, _ = encoder(input_ids, attention_mask, **extra_kwargs)  # [B, K, D_enc]
        else:
            with torch.no_grad():
                latent, _ = encoder(input_ids, attention_mask, **extra_kwargs)
            latent = latent.detach()

        if actual in self.projectors:
            projected = self.projectors[actual](latent)  # [B, K, D_llm]
        else:
            projected = latent

        # Guard against NaN/Inf from encoder/projector — log and clamp
        # to prevent CUBLAS failures in downstream bf16 matmuls.
        if torch.isnan(projected).any() or torch.isinf(projected).any():
            logger.warning(
                f"[ModalityRouter] NaN/Inf in '{name}' encoder/projector output! "
                f"shape={projected.shape}, nan={torch.isnan(projected).sum()}, "
                f"inf={torch.isinf(projected).sum()}"
            )
            projected = torch.nan_to_num(projected, nan=0.0, posinf=1e4, neginf=-1e4)

        return projected.reshape(-1, projected.shape[-1])  # [B*K, D_llm]

    # ------------------------------------------------------------------
    # High-level: scatter all modalities into input embeddings
    # ------------------------------------------------------------------

    def scatter_all(
        self,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        bio_token_ids: Dict,
        **kwargs,
    ) -> Tuple[Tensor, Set[str], Optional[Tensor], Dict[str, Tensor]]:
        """Scatter all active modality embeddings into the LLM input.

        Looks up ``{name}_input_ids`` in kwargs for each registered
        modality.  If present, encodes, projects, and scatters the
        embeddings into the corresponding pad-token positions.

        Args:
            input_ids: [B, L] token IDs.
            inputs_embeds: [B, L, D] embeddings (modified in-place).
            bio_token_ids: Dict from config, e.g.
                ``{"rna": {"pad": 151900}, "dna": {"pad": 151903}, ...}``
            **kwargs: Must contain ``{name}_input_ids``,
                      ``{name}_attention_mask``, ``{name}_grid_thw`` for
                      each active modality.

        Returns:
            inputs_embeds: Updated embeddings.
            active_modalities: Set of canonical modality names that were active.
            image_mask: Combined 3D mask of all scattered positions (for
                        deepstack / visual_pos_masks).
            image_like_grids: ``{modality_name -> grid_thw}`` for RoPE.
                Returned as a dict (rather than a list) so callers can
                reorder grids in text-appearance order before feeding
                ``get_rope_index_3`` — see
                :func:`qwenvl.data.rope2d._reorder_image_like_grids_by_text_pos`.
        """
        active_modalities: Set[str] = set()
        image_mask: Optional[Tensor] = None
        image_like_grids: Dict[str, Tensor] = {}
        for mod_name in self.modality_names:
            mod_input_ids = kwargs.get(f"{mod_name}_input_ids")
            if mod_input_ids is not None:
                # ── Real forward: encode + project + scatter ──
                mod_attn_mask = kwargs.get(f"{mod_name}_attention_mask")
                mod_grid_thw = kwargs.get(f"{mod_name}_grid_thw")

                active_modalities.add(mod_name)

                # Collect extra kwargs for this modality (e.g. mol_edge_index → edge_index)
                # Skip standard fields (input_ids, attention_mask, grid_thw) and labels
                _standard = {f"{mod_name}_input_ids", f"{mod_name}_attention_mask", f"{mod_name}_grid_thw"}
                _prefix = f"{mod_name}_"
                extra_kwargs = {
                    k[len(_prefix):]: v for k, v in kwargs.items()
                    if k.startswith(_prefix) and k not in _standard and not k.endswith("_labels")
                }

                # Encode + project → flat [total_tokens, D_llm]
                embeds = self.encode_and_project(mod_name, mod_input_ids, mod_attn_mask, **extra_kwargs)
                embeds = embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                # Build scatter mask from all pad tokens for this modality
                # (includes aliases, e.g. both rna_pad and dna_pad)
                pad_ids = self._collect_pad_ids(mod_name, bio_token_ids)
                if pad_ids and input_ids is not None:
                    mask = torch.zeros(
                        input_ids.shape, dtype=torch.bool, device=input_ids.device
                    )
                    for pid in pad_ids:
                        mask = mask | (input_ids == pid)

                    mask_3d = mask.unsqueeze(-1).expand_as(inputs_embeds)
                    n_mask = mask.sum()
                    n_embeds = embeds.shape[0]
                    if n_mask != n_embeds:
                        raise ValueError(
                            f"Modality '{mod_name}': pad tokens ({n_mask}) != "
                            f"encoder features ({n_embeds})"
                        )

                    # Per-sample sanity: each sample's pad-token count must
                    # be a multiple of K (the per-sequence latent count) so
                    # that pad blocks line up with encoder rows 1:1.  Catches
                    # silent collator/processor errors that the global count
                    # check above would let through (e.g. swapped sequences
                    # between samples when K is identical).
                    if mod_grid_thw is not None and mod_grid_thw.numel() > 0:
                        per_seq_pads = (
                            mod_grid_thw[:, 0] * mod_grid_thw[:, 1] * mod_grid_thw[:, 2]
                        )
                        K = int(per_seq_pads[0].item())
                        if K > 0 and not torch.all(per_seq_pads == K):
                            raise ValueError(
                                f"Modality '{mod_name}': non-uniform per-sequence "
                                f"latent count {per_seq_pads.tolist()} not yet "
                                f"supported by scatter_all"
                            )
                        per_sample_pads = mask.sum(dim=-1)
                        if K > 0 and not torch.all(per_sample_pads % K == 0):
                            raise ValueError(
                                f"Modality '{mod_name}': per-sample pad-token "
                                f"count {per_sample_pads.tolist()} not divisible "
                                f"by K={K} (a sample's pad-block count is wrong)"
                            )

                    inputs_embeds = inputs_embeds.masked_scatter(mask_3d, embeds)

                    if image_mask is None:
                        image_mask = mask_3d
                    else:
                        image_mask = image_mask | mask_3d

                # Collect grid_thw for RoPE (image-like modalities only)
                if mod_grid_thw is not None and self._is_image_like.get(mod_name, True):
                    image_like_grids[mod_name] = mod_grid_thw

        return inputs_embeds, active_modalities, image_mask, image_like_grids

    def _collect_pad_ids(
        self, canonical_name: str, bio_token_ids: Dict
    ) -> List[int]:
        """Collect all pad token IDs for a canonical modality + its aliases."""
        pad_ids = []
        # Canonical pad token
        mod_ids = bio_token_ids.get(canonical_name, {})
        if "pad" in mod_ids:
            pad_ids.append(mod_ids["pad"])
        # Alias pad tokens (e.g. dna_pad for rna encoder)
        for alias, target in self._aliases.items():
            if target == canonical_name:
                alias_ids = bio_token_ids.get(alias, {})
                if "pad" in alias_ids:
                    pad_ids.append(alias_ids["pad"])
        return pad_ids
