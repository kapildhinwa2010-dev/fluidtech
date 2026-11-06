"""
TEST SUITE FOR SUPERFLUID-MAMBA FLUIDLAYER

Honest testing (2026-11-06 review):
- A recall rate that is printed is now ASSERTED, not just displayed.
- The exact tier can only recall contexts it has SEEN before, so the
  recall scenario replays the same token sequence a second time and
  asserts the measured recall of the second pass.
- The fresh-sequence pass asserts that recall is 0 by construction
  (nothing has been stored yet) instead of silently printing 0.00%.
- All headline performance numbers are hypothetical and unverified; the
  suite no longer prints them as if they were results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time

print("=" * 80)
print("SUPERFLUID-MAMBA FLUIDLAYER - TEST SUITE")
print("=" * 80)

# Test 1: Configuration
print("\n[1/10] Testing configuration...")
try:
    from experiments.superfluid_mamba_layer import SuperfluidMambaConfig, SecurityLevel
    
    config = SuperfluidMambaConfig(
        M=64,
        r=8,
        d=64,
        ctx_len=12,
        mamba_d_state=16,
        mamba_d_conv=4,
        superfluid_viscosity=0.1,
        superfluid_decay=0.01,
        superfluid_k=8,
    )
    config.validate()
    print("OK - Configuration validation passed")
except Exception as e:
    print(f"FAIL - Configuration test failed: {e}")
    sys.exit(1)

# Test 2: Cryptographic Session
print("\n[2/10] Testing cryptographic session...")
try:
    from experiments.superfluid_mamba_layer import CryptographicSession, SecurityLevel
    
    session = CryptographicSession(security_level=SecurityLevel.ULTRA)
    token = 12345
    masked = session.mask_token(token)
    unmasked = session.unmask_token(masked)
    assert unmasked == token, f"Unmasking failed: {unmasked} != {token}"
    
    session2 = CryptographicSession(security_level=SecurityLevel.ULTRA)
    masked2 = session2.mask_token(token)
    assert masked != masked2, "Different sessions should have different masks"
    
    # Encryption round-trip must actually invert (the old version
    # generated a new random key per call and could never decrypt).
    try:
        data = b"payload"
        enc = session.encrypt(data)
        assert session.decrypt(enc) == data, "encrypt/decrypt round-trip failed"
    except ImportError:
        print("   (cryptography not installed; skipping round-trip)")
    
    print("OK - Cryptographic session test passed")
except Exception as e:
    print(f"FAIL - Cryptographic session test failed: {e}")
    sys.exit(1)

# Test 3: Hash Store
print("\n[3/10] Testing hash store...")
try:
    from experiments.superfluid_mamba_layer import PerfectHashStore, HashMode
    
    store = PerfectHashStore(
        initial_capacity=100,
        max_capacity=1000,
        hash_mode=HashMode.DOUBLE_64
    )
    
    store.put(123, 456)
    assert store.get(123) == 456, "Store get failed"
    assert store.contains(123), "Store contains failed"
    
    test_key = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    test_value = 999
    store.put(test_key, test_value)
    assert store.get(test_key) == test_value, "12-gram store failed"
    
    # Session isolation: a store with a different session must NOT see
    # entries written under another session.
    from experiments.superfluid_mamba_layer import CryptographicSession
    other = PerfectHashStore(
        initial_capacity=100,
        max_capacity=1000,
        hash_mode=HashMode.DOUBLE_64,
        session=CryptographicSession(),
    )
    assert other.get(123) is None, "cross-session leak: other store saw the entry"
    
    print("OK - Hash store test passed (including session isolation)")
except Exception as e:
    print(f"FAIL - Hash store test failed: {e}")
    sys.exit(1)

# Test 4: Mamba-SSM
print("\n[4/10] Testing Mamba-SSM...")
try:
    from experiments.superfluid_mamba_layer import MambaSSM
    
    mamba = MambaSSM(
        d_model=64,
        d_state=16,
        d_conv=4,
        dt_rank=16,
    )
    
    x = torch.randn(2, 10, 64)  # (batch, seq_len, d_model)
    output = mamba(x)
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"
    # A scan must be causal: changing a LATER input token must not
    # change EARLIER outputs.
    x2 = x.clone()
    x2[:, 8, :] += 10.0
    out2 = mamba(x2)
    assert torch.allclose(output[:, :8], out2[:, :8], atol=1e-5), \
        "Mamba scan is not causal: future tokens changed past outputs"
    
    print("OK - Mamba-SSM test passed (shape + causality)")
except Exception as e:
    print(f"FAIL - Mamba-SSM test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Superfluid Diffusion
print("\n[5/10] Testing Superfluid diffusion...")
try:
    from experiments.superfluid_mamba_layer import SuperfluidDiffusion
    
    superfluid = SuperfluidDiffusion(
        M=64,
        d=64,
        viscosity=0.1,
        decay=0.01,
        k=8,
    )
    
    state = torch.randn(64, 64)
    output = superfluid(state, t=1.0)
    assert output.shape == state.shape, f"Shape mismatch: {output.shape} != {state.shape}"
    
    # The diffusion must actually MIX information across nodes: with a
    # one-hot state, more than one output position must be non-zero.
    onehot = torch.zeros(64, 64)
    onehot[0, 0] = 1.0
    diff = superfluid(onehot, t=1.0)
    nonzero_rows = int((diff.abs().sum(dim=1) > 1e-9).sum())
    assert nonzero_rows > 1, \
        f"diffusion did not propagate: only {nonzero_rows} non-zero rows"
    
    print(f"OK - Superfluid diffusion test passed (spread to {nonzero_rows} rows)")
except Exception as e:
    print(f"FAIL - Superfluid diffusion test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Adaptive Gated Sheaf
print("\n[6/10] Testing adaptive gated sheaf...")
try:
    from experiments.superfluid_mamba_layer import AdaptiveGatedSheaf
    
    sheaf = AdaptiveGatedSheaf(
        M=16,
        r=4,
        d=8,
        k=4,
        seed=0,
    )
    
    # Adjacency must be fully initialized (no uninitialized memory from
    # np.divide(..., where=...) without out=) and row-normalized.
    adj = sheaf.adjacency
    assert torch.isfinite(adj).all(), "adjacency contains non-finite values"
    row_sums = adj.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(16), atol=1e-4), \
        f"adjacency is not row-stochastic: sums {row_sums[:4]}"
    
    state = torch.randn(16, 4)
    output = sheaf.apply_diffusion(state, t=1.0, cheb_m=8)
    assert output.shape == state.shape, f"Shape mismatch: {output.shape} != {state.shape}"
    
    print("OK - Adaptive gated sheaf test passed")
except Exception as e:
    print(f"FAIL - Adaptive gated sheaf test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Gated Fusion
print("\n[7/10] Testing gated fusion...")
try:
    from experiments.superfluid_mamba_layer import GatedFusion
    
    fusion = GatedFusion(
        d=64,
        num_tiers=4,
        hidden_size=128,
    )
    
    h = torch.randn(10, 64)
    tier_outputs = [torch.randn(10, 64) for _ in range(4)]
    output = fusion(h, tier_outputs)
    assert output.shape == h.shape, f"Shape mismatch: {output.shape} != {h.shape}"
    
    # The gate must be per-tier (softmax over tiers, weights sum to 1).
    # The old version applied one gate to every tier, which could never
    # favour one tier over another.
    concat = torch.cat([h] + [fusion.projections[i](tier_outputs[i]) for i in range(4)], dim=-1)
    gate = fusion.fusion(concat)  # (seq_len, num_tiers)
    assert gate.shape == (10, 4), f"gate shape wrong: {gate.shape}"
    assert torch.allclose(gate.sum(dim=-1), torch.ones(10), atol=1e-5), \
        "per-tier gates do not sum to 1"
    assert not torch.allclose(gate[:, 0], gate[:, 1]), \
        "gate weights are identical across tiers (fusion is a no-op)"
    
    print("OK - Gated fusion test passed (per-tier gating)")
except Exception as e:
    print(f"FAIL - Gated fusion test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Full PyTorch Layer, with ASSERTED recall
print("\n[8/10] Testing full PyTorch Superfluid-Mamba layer (with recall assertions)...")
try:
    from experiments.superfluid_mamba_layer import SuperfluidMambaFluidLayer
    
    config = SuperfluidMambaConfig(
        M=8,
        r=4,
        d=8,
        ctx_len=4,
        mamba_d_state=8,
        mamba_d_conv=2,
        superfluid_viscosity=0.1,
        superfluid_decay=0.01,
        superfluid_k=4,
        capacity=1000,
        max_capacity=10000,
    )
    
    layer = SuperfluidMambaFluidLayer(config)
    
    seq_len = 20
    h = torch.randn(seq_len, 8)
    content = torch.randn(seq_len, 8)
    token_ids = torch.randint(0, 100, (seq_len,))
    
    # Pass 1 (fresh sequence): nothing has been stored yet, so the exact
    # tier recall must be exactly 0 by construction.
    output1, info1 = layer(h, content, token_ids)
    assert output1.shape == (seq_len, 8), f"output shape wrong: {output1.shape}"
    r1 = info1['exact_tier']['recall_rate']
    assert r1 == 0.0, f"fresh pass should have 0 recall, got {r1}"
    
    # Pass 2 (same sequence replayed): every context of length ctx_len
    # that was stored in pass 1 must now be recalled.  Positions with a
    # complete context are hits; the first ctx_len-1 positions and the
    # last position are misses (the final context was never stored).
    output2, info2 = layer(h, content, token_ids)
    assert output2.shape == (seq_len, 8), f"output shape wrong: {output2.shape}"
    r2 = info2['exact_tier']['recall_rate']
    expected_hits = seq_len - config.ctx_len  # 20 - 4 = 16
    expected_recall = expected_hits / seq_len
    assert abs(r2 - expected_recall) < 0.01, \
        f"replay recall {r2:.2f} != expected {expected_recall:.2f}"
    assert r2 > 0, "replay recall must be positive, not just printed"
    
    print(f"OK - PyTorch layer: fresh recall {r1:.0%} (asserted 0), "
          f"replay recall {r2:.0%} (asserted {expected_recall:.0%})")
except Exception as e:
    print(f"FAIL - Full PyTorch layer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Full NumPy Layer, with ASSERTED recall
print("\n[9/10] Testing full NumPy Superfluid-Mamba layer (with recall assertions)...")
try:
    from experiments.superfluid_mamba_layer import SuperfluidMambaFluidLayerNP
    
    config = SuperfluidMambaConfig(
        M=8,
        r=4,
        d=8,
        ctx_len=4,
        mamba_d_state=8,
        mamba_d_conv=2,
        superfluid_viscosity=0.1,
        superfluid_decay=0.01,
        superfluid_k=4,
        capacity=1000,
        max_capacity=10000,
    )
    
    layer = SuperfluidMambaFluidLayerNP(config)
    
    seq_len = 20
    h = np.random.randn(seq_len, 8).astype(np.float32)
    content = np.random.randn(seq_len, 8).astype(np.float32)
    token_ids = np.random.randint(0, 100, (seq_len,))
    
    output1, info1 = layer(h, content, token_ids)
    assert output1.shape == (seq_len, 8), f"output shape wrong: {output1.shape}"
    r1 = info1['recall_rate']
    assert r1 == 0.0, f"fresh pass should have 0 recall, got {r1}"
    
    output2, info2 = layer(h, content, token_ids)
    assert output2.shape == (seq_len, 8), f"output shape wrong: {output2.shape}"
    r2 = info2['recall_rate']
    expected_recall = (seq_len - config.ctx_len) / seq_len
    assert abs(r2 - expected_recall) < 0.01, \
        f"replay recall {r2:.2f} != expected {expected_recall:.2f}"
    
    print(f"OK - NumPy layer: fresh recall {r1:.0%} (asserted 0), "
          f"replay recall {r2:.0%} (asserted {expected_recall:.0%})")
except Exception as e:
    print(f"FAIL - Full NumPy layer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Recall across a 100-token replay
# Honest name: this measures EXACT-TIER REPLAY RECALL over a long
# sequence.  It does NOT demonstrate learned long-range dependency --
# that would require training against a language model.
print("\n[10/10] Testing exact-tier replay recall over 100 tokens...")
try:
    from experiments.superfluid_mamba_layer import SuperfluidMambaFluidLayer
    
    config = SuperfluidMambaConfig(
        M=16,
        r=8,
        d=16,
        ctx_len=12,
        mamba_d_state=16,
        mamba_d_conv=4,
        superfluid_viscosity=0.1,
        superfluid_decay=0.01,
        superfluid_k=8,
        capacity=10000,
        max_capacity=100000,
    )
    
    layer = SuperfluidMambaFluidLayer(config)
    
    seq_len = 100
    h = torch.randn(seq_len, 16)
    content = torch.randn(seq_len, 16)
    token_ids = torch.randint(0, 1000, (seq_len,))
    
    # Pass 1: store
    _, info1 = layer(h, content, token_ids)
    assert info1['exact_tier']['recall_rate'] == 0.0
    
    # Pass 2: replay
    output, info2 = layer(h, content, token_ids)
    assert output.shape == (seq_len, 16), "Output shape mismatch"
    r = info2['exact_tier']['recall_rate']
    expected_recall = (seq_len - config.ctx_len) / seq_len  # 0.88
    assert abs(r - expected_recall) < 0.01, \
        f"100-token replay recall {r:.2f} != expected {expected_recall:.2f}"
    
    print(f"OK - 100-token replay recall {r:.0%} (asserted {expected_recall:.0%})")
except Exception as e:
    print(f"FAIL - Replay recall test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED")
print("=" * 80)
print("\nWhat is verified:")
print("  - Configuration validation")
print("  - Session masking, and encrypt/decrypt round-trip (when cryptography is installed)")
print("  - Hash store put/get + cross-session isolation")
print("  - Mamba scan shape AND causality")
print("  - Superfluid diffusion actually spreads information (not just shape)")
print("  - Sheaf adjacency is finite and row-stochastic")
print("  - Per-tier gated fusion (gates differ and sum to 1)")
print("  - Exact tier: 0 recall on fresh input (asserted),")
print(f"    and replay recall matching the computed expectation (asserted)")
print("\nWhat is NOT verified here:")
print("  - Perplexity / accuracy / speed improvements: hypothetical only,")
print("    never measured against a real LLM.")
print("  - Learned long-range dependency: requires training.")
print("\nBranch: experiments (experimental code, not production-ready)")
print("=" * 80)
