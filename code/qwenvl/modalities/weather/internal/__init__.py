# SPDX-License-Identifier: Apache-2.0
# Portions adapted from the Polaris weather-forecasting codebase (released here
# under Apache-2.0 with the authors' permission) and from the Swin Transformer
# (Microsoft, MIT). See THIRD_PARTY_LICENSES.md at the repo root.
"""Internal Polaris support modules (attention, layers, swin, helpers).

These are adapted from the original Polaris codebase and keep the Swin
Transformer window-attention / patch-merging design. They are kept as a private
submodule so the public weather encoder/decoder/projector can pin against a
stable internal API.
"""
