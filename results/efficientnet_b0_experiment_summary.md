# EfficientNet-B0 controlled experiment summary

Selection used internal tuning macro F1 only. The held-out split was not used.

| Rank | Run | Dropout | Learning rate | Best epoch | Tuning accuracy | Tuning macro F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `efficientnet_b0_baseline` | 0.3 | 0.001 | 10 | 0.9643 | 0.9646 |
| 2 | `efficientnet_b0_drop04` | 0.4 | 0.001 | 11 | 0.9619 | 0.9621 |
| 3 | `efficientnet_b0_lr3e4` | 0.3 | 0.0003 | 9 | 0.9405 | 0.9406 |

Selected run: `efficientnet_b0_baseline`.

[Inference] The lower learning rate underfit within the shared epoch budget. Increasing dropout to 0.4 remained competitive but did not exceed the 0.3 baseline.
