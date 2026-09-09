"""SeqFormer architecture — framework-neutral notes; implementations live in the
backend modules (``seqformer.backends``).

Reference: "SeqFormer: Sequential Transformer for Video Instance Segmentation"
(Wu et al., ECCV 2022), built on Deformable DETR.

Components to build:
  backbone      ResNet-50 / Swin-T  -> multi-scale features C3..C5
  input_proj    1x1 convs -> common channel dim (256)
  encoder       deformable-attention encoder over the flattened feature maps
  query_embed   Nq instance queries, shared across the clip
  decoder       deformable-attention decoder; per-frame + cross-frame attention
  box_head      MLP -> per-frame boxes
  class_head    Linear -> class logits (clip-level)
  mask_head     dynamic conv / dot-product of instance embed with mask features

The cross-frame attention is what makes it "video": one instance query pools
evidence across the whole input clip before predicting.
"""
