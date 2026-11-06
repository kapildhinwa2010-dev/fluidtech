"""
TEST SUITE FOR ULTRA-FLUID LAYER

Honest testing (2026-11-06 review):
- The single-layer test now REPLAYS the same sequence twice and asserts
  the measured exact-tier recall of the replay pass (the old suite
  printed a recall without ever asserting it).
- The hypothetical headline numbers are no longer printed as results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time

print("=" * 80)
print("ULTRA-FLUID LAYER - TEST SUITE")
print("=" * 80)

# Test 1: Adaptive Router
print("\n[1/10] Testing adaptive router...")
try:
    from ultra_fluid_layer import AdaptiveRouter
    
    router = AdaptiveRouter(d=64, num_components=4, hidden_size=128)
    
    h = torch.randn(10, 64)
    weights = router(h)
    assert weights.shape == (10, 4), f"Expected (10, 4), got {weights.shape}"
    assert torch.allclose(weights.sum(dim=1), torch.ones(10)), "Weights should sum to 1"
    
    print("OK - Adaptive router test passed")
except Exception as e:
    print(f"FAIL - Adaptive router test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Cross-Attention Fusion
print("\n[2/10] Testing cross-attention fusion...")
try:
    from ultra_fluid_layer import CrossAttentionFusion
    
    fusion = CrossAttentionFusion(d=64, num_heads=4)
    
    query = torch.randn(10, 64)
    key = torch.randn(10, 64)
    value = torch.randn(10, 64)
    
    output = fusion(query, key, value)
    assert output.shape == query.shape, f"Expected {query.shape}, got {output.shape}"
    
    print("OK - Cross-attention fusion test passed")
except Exception as e:
    print(f"FAIL - Cross-attention fusion test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Memory Compressor
print("\n[3/10] Testing memory compressor...")
try:
    from ultra_fluid_layer import MemoryCompressor
    
    compressor = MemoryCompressor(d=64, compression_factor=0.5)
    
    x = torch.randn(10, 64)
    compressed = compressor(x, compress=True)
    assert compressed.shape == (10, 32), f"Expected (10, 32), got {compressed.shape}"
    
    decompressed = compressor(compressed, compress=False)
    assert decompressed.shape == (10, 64), f"Expected (10, 64), got {decompressed.shape}"
    
    print("OK - Memory compressor test passed")
except Exception as e:
    print(f"FAIL - Memory compressor test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Ultra-Fluid Layer (Single), with ASSERTED recall
print("\n[4/10] Testing single Ultra-Fluid layer (with recall assertions)...")
try:
    from ultra_fluid_layer import UltraFluidLayer, UltraFluidConfig
    
    config = UltraFluidConfig(
        M=8,
        r=4,
        d=8,
        ctx_len=4,
        num_layers=1,
        use_residual=True,
        use_cross_attention=True,
        use_adaptive_routing=True,
    )
    
    layer = UltraFluidLayer(config)
    
    seq_len = 20
    h = torch.randn(seq_len, 8)
    content = torch.randn(seq_len, 8)
    token_ids = torch.randint(0, 100, (seq_len,))
    
    # Pass 1 (fresh): nothing stored yet -> recall must be 0.
    output1, info1 = layer(h, content, token_ids)
    assert output1.shape == (seq_len, 8), f"Expected ({seq_len}, 8), got {output1.shape}"
    r1 = info1['exact_tier']['recall_rate']
    assert r1 == 0.0, f"fresh pass should have 0 recall, got {r1}"
    
    # Pass 2 (replay): recall must match the computed expectation.
    output2, info2 = layer(h, content, token_ids)
    assert output2.shape == (seq_len, 8), f"Expected ({seq_len}, 8), got {output2.shape}"
    r2 = info2['exact_tier']['recall_rate']
    expected_recall = (seq_len - config.ctx_len) / seq_len
    assert abs(r2 - expected_recall) < 0.01, \
        f"replay recall {r2:.2f} != expected {expected_recall:.2f}"
    
    print(f"OK - Single layer: fresh recall {r1:.0%} (asserted 0), "
          f"replay recall {r2:.0%} (asserted {expected_recall:.0%})")
except Exception as e:
    print(f"FAIL - Single Ultra-Fluid layer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Ultra-Fluid Layer (No Residual)
print("\n[5/10] Testing Ultra-Fluid layer without residual...")
try:
    from ultra_fluid_layer import UltraFluidLayer, UltraFluidConfig
    
    config = UltraFluidConfig(
        M=8,
        r=4,
        d=8,
        ctx_len=4,
        num_layers=1,
        use_residual=False,
        use_cross_attention=False,
        use_adaptive_routing=False,
    )
    
    layer = UltraFluidLayer(config)
    
    h = torch.randn(5, 8)
    content = torch.randn(5, 8)
    token_ids = torch.randint(0, 100, (5,))
    
    output, info = layer(h, content, token_ids)
    assert output.shape == (5, 8), f"Expected (5, 8), got {output.shape}"
    
    # The non-residual branch must actually differ from the residual one:
    # with all-zero W matrices and no routing/cross-attention, the output
    # should be exactly the tier sum (zero here), NOT h + 0.
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6), \
        "non-residual path still adds the residual"
    
    print("OK - No-residual Ultra-Fluid layer test passed (residual actually off)")
except Exception as e:
    print(f"FAIL - No-residual Ultra-Fluid layer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Ultra-Fluid Stack (2 Layers)
print("\n[6/10] Testing Ultra-Fluid stack (2 layers)...")
try:
    from ultra_fluid_layer import UltraFluidStack, UltraFluidConfig
    
    config = UltraFluidConfig(
        M=8,
        r=4,
        d=8,
        ctx_len=4,
        num_layers=2,
        use_residual=True,
        use_cross_attention=True,
        use_adaptive_routing=True,
    )
    
    stack = UltraFluidStack(config)
    
    h = torch.randn(5, 8)
    content = torch.randn(5, 8)
    token_ids = torch.randint(0, 100, (5,))
    
    output, all_info = stack(h, content, token_ids)
    assert output.shape == (5, 8), f"Expected (5, 8), got {output.shape}"
    assert len(all_info) == 2, f"Expected 2 infos, got {len(all_info)}"
    
    print("OK - Ultra-Fluid stack test passed (2 layers)")
except Exception as e:
    print(f"FAIL - Ultra-Fluid stack test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Ultra-Fluid Stack (4 Layers)
print("\n[7/10] Testing Ultra-Fluid stack (4 layers)...")
try:
    from ultra_fluid_layer import UltraFluidStack, UltraFluidConfig
    
    config = UltraFluidConfig(
        M=8,
        r=4,
        d=8,
        ctx_len=4,
        num_layers=4,
        use_residual=True,
        use_cross_attention=True,
        use_adaptive_routing=True,
    )
    
    stack = UltraFluidStack(config)
    
    h = torch.randn(5, 8)
    content = torch.randn(5, 8)
    token_ids = torch.randint(0, 100, (5,))
    
    output, all_info = stack(h, content, token_ids)
    assert output.shape == (5, 8), f"Expected (5, 8), got {output.shape}"
    assert len(all_info) == 4, f"Expected 4 infos, got {len(all_info)}"
    
    print("OK - Ultra-Fluid stack test passed (4 layers)")
except Exception as e:
    print(f"FAIL - Ultra-Fluid stack test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Compressed Ultra-Fluid (pass-through)
print("\n[8/10] Testing compressed Ultra-Fluid (currently a pass-through)...")
try:
    from ultra_fluid_layer import CompressedUltraFluid, UltraFluidConfig
    
    config = UltraFluidConfig(
        M=8,
        r=4,
        d=8,
        ctx_len=4,
        num_layers=1,
        compress_state=True,
        compression_factor=0.5,
    )
    
    compressed = CompressedUltraFluid(config)
    
    h = torch.randn(5, 8)
    content = torch.randn(5, 8)
    token_ids = torch.randint(0, 100, (5,))
    
    output, info = compressed(h, content, token_ids)
    assert output.shape == (5, 8), f"Expected (5, 8), got {output.shape}"
    
    print("OK - Compressed Ultra-Fluid test passed (no memory reduction claimed)")
except Exception as e:
    print(f"FAIL - Compressed Ultra-Fluid test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Factory Function
print("\n[9/10] Testing factory function...")
try:
    from ultra_fluid_layer import create_ultra_fluid_layer
    
    stack = create_ultra_fluid_layer(
        d=8,
        M=8,
        r=4,
        num_layers=2,
        use_residual=True,
        use_cross_attention=True,
        use_adaptive_routing=True,
    )
    
    h = torch.randn(5, 8)
    content = torch.randn(5, 8)
    token_ids = torch.randint(0, 100, (5,))
    
    output, all_info = stack(h, content, token_ids)
    assert output.shape == (5, 8), f"Expected (5, 8), got {output.shape}"
    
    print("OK - Factory function test passed")
except Exception as e:
    print(f"FAIL - Factory function test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Long Sequence Test
print("\n[10/10] Testing long sequence (100 tokens)...")
try:
    from ultra_fluid_layer import UltraFluidStack, UltraFluidConfig
    
    config = UltraFluidConfig(
        M=16,
        r=8,
        d=16,
        ctx_len=12,
        num_layers=2,
        use_residual=True,
        use_cross_attention=True,
        use_adaptive_routing=True,
    )
    
    stack = UltraFluidStack(config)
    
    seq_len = 100
    h = torch.randn(seq_len, 16)
    content = torch.randn(seq_len, 16)
    token_ids = torch.randint(0, 1000, (seq_len,))
    
    output, all_info = stack(h, content, token_ids)
    assert output.shape == (seq_len, 16), f"Expected ({seq_len}, 16), got {output.shape}"
    
    print(f"OK - Long sequence test passed ({seq_len} tokens)")
except Exception as e:
    print(f"FAIL - Long sequence test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED")
print("=" * 80)
print("\nWhat is verified:")
print("  - Router / cross-attention / compressor shapes and normalisation")
print("  - Single layer forward, with fresh recall asserted 0 and replay")
print("    recall asserted against the computed expectation")
print("  - The non-residual path actually omits the residual")
print("  - Stack (2 and 4 layers) and factory forward shapes")
print("\nWhat is NOT verified here:")
print("  - The hypothetical perplexity / accuracy / speed / memory numbers:")
print("    unmeasured, and the compression variant is a pass-through.")
print("\nBranch: experiments (experimental code, not production-ready)")
print("=" * 80)
