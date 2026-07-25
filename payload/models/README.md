# Selected model

Expected local artifact:

```text
prithvi_crop_classifier_v1_weights.pt
```

The weights-only artifact is deliberately ignored by Git because it is
409,025,203 bytes. `selected_model.yaml` records the expected size and SHA-256
digest. Packaging must fail if either differs.

No `last.ckpt`, alternative epoch or training optimizer state should be chosen
automatically. Replacing this model requires explicit review.

The full 536,560,975-byte Lightning checkpoint remains in
`outputs/prithvi_4band_europe_replay/checkpoints/` as the single provenance
copy. It contains training state and is excluded from the flight bundle.
