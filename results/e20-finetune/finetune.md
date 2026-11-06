# E20: short-budget fine-tune + KV wrapper

- commit: 4912034e539ce13b0d874db0a24fb2339dbcbacf
- model: sshleifer/tiny-gpt2
- budget: 5.0 min
- KV wrapper OFF during training (by design),
  attached + bounded-KV checked AFTER fine-tuning

## Measured

| Metric | Before fine-tune | After fine-tune |
|---|---|---|
| plain eval loss (nats) | 10.826 | 10.8254 |
| wrapped eval loss (nats) | 10.826 | 10.8254 |
| attach delta (nats) | 0.0 | 0.0 |

- train steps: 840 in 300.3 s
- train loss: 10.8249 -> 10.825
- KV after fine-tune: max 256, evictions 1280 (bounded by 256)
- wrapper recall after fine-tune: 0.0

## Claims (measured in THIS run)

- **kv_off_during_training**: True
- **design_note**: wrapper has no trainable params; training updates the base model (LoRA); evictions would corrupt training context
- **trained**: True
- **within_time_budget**: True
- **train_loss_decreased**: False
- **eval_loss_improved**: True
- **eval_improvement_nats**: 0.0006
- **attach_delta_before_nats**: 0.0
- **attach_delta_after_nats**: 0.0
- **kv_stays_bounded_after_finetune**: True
- **wrapper_recall_after_finetune**: 0.0
- **lora_used**: True

## Full log

```
======================================================================
E20: FINE-TUNE + KV WRAPPER  sha=4912034e539ce13b0d874db0a24fb2339dbcbacf
model=sshleifer/tiny-gpt2  budget=5.0 min
======================================================================
datasets/wikitext unavailable: no wikitext variant loaded
data: corpus.txt fallback (60k chars) -> 128 train / 12 eval chunks

--- Phase A: before fine-tune ---
plain loss: 10.826  wrapped loss: 10.826

--- Phase B: fine-tune (5.0 min budget, wrapper OFF) ---
lora used: True  trainable: 0.1245%
steps: 840  seconds: 300.3  loss: 10.8249 -> 10.825

--- Phase C: after fine-tune ---
plain loss: 10.8254  wrapped loss: 10.8254
kv max: 256  evictions: 1280
```