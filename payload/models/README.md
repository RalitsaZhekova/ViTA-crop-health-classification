# Selected model

Expected local artifact:

```text
prithvi_crop_binary_single_frame_v1_weights.pt
```

The weights-only artifact is deliberately ignored by Git because it is
382,645,915 bytes. `selected_model.yaml` records the expected size and SHA-256
digest. Packaging must fail if either differs.

No `last.ckpt`, alternative epoch or training optimizer state should be chosen
automatically. Replacing this model requires explicit review.

The selected model accepts one four-band image and emits direct crop/non-crop
logits. Its full Lightning checkpoint and the preserved one- and three-frame
crop-type models remain under `outputs/` and are excluded from the flight
bundle. The previous weights-only payload artifact is preserved under
`outputs/model_archive/`.
