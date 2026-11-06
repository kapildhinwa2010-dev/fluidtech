"""
ULTRA-FLUID LAYER - NEXT GENERATION SUPERGLUED MAMBA (experimental)

Combines:
- Superfluid-Mamba FluidLayer (base)
- Multi-Layer Stacking (2-4 layers deep)
- Residual Connections
- Cross-Attention Fusion
- Adaptive Routing
- Memory Compression (currently a placeholder)

Honest status (2026-11-06 review):
- All headline performance numbers are HYPOTHETICAL and unmeasured.
- The "compression" variant is currently a pass-through: it does not
  reduce memory yet.
- The cross-attention K/V projection is now a registered, trainable
  module (it used to be a fresh random Linear created inside forward()).

Author: FluidTech Research Team
Version: 2.1.0 (Ultra-Experimental)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid
import threading
import time
import logging
import math

# Configure logging
logger = logging.getLogger("ultra_fluid")
logger.setLevel(logging.INFO)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UltraFluidConfig:
    """Configuration for Ultra-Fluid Layer"""
    
    # Base layer parameters
    M: int = 64                   # Number of anchors
    r: int = 8                    # Rank for low-rank projections
    d: int = 64                   # Hidden dimension
    ctx_len: int = 12             # Context length for exact matching
    
    # Mamba-SSM parameters
    mamba_d_state: int = 16       # State dimension for Mamba
    mamba_d_conv: int = 4         # Convolution kernel size
    mamba_dt_rank: int = 16       # Rank for dt projection
    
    # Superfluid parameters
    superfluid_viscosity: float = 0.1
    superfluid_decay: float = 0.01
    superfluid_k: int = 8
    
    # Ultra-Fluid enhancements
    num_layers: int = 2           # Number of stacked layers
    use_residual: bool = True     # Use residual connections
    use_cross_attention: bool = True  # Use cross-attention fusion
    use_adaptive_routing: bool = True  # Use adaptive routing
    routing_hidden: int = 128     # Hidden size for router
    
    # Memory optimization
    compress_state: bool = True   # Compress Mamba state
    compression_factor: float = 0.5  # Compression ratio
    
    # Memory parameters
    capacity: int = 10000
    max_capacity: int = 100000
    
    # Security
    security_level: str = "ULTRA"
    
    def validate(self) -> None:
        """Validate configuration"""
        assert self.M > 0, f"M must be positive, got {self.M}"
        assert self.r > 0, f"r must be positive, got {self.r}"
        assert self.d > 0, f"d must be positive, got {self.d}"
        assert self.num_layers >= 1, f"num_layers must be >= 1, got {self.num_layers}"
        assert 0 < self.superfluid_viscosity <= 1
        assert 0 <= self.superfluid_decay < 1
        assert 0 < self.compression_factor <= 1


# =============================================================================
# ADAPTIVE ROUTER
# =============================================================================

class AdaptiveRouter(nn.Module):
    """
    Adaptive router that learns which components to use.
    
    Features:
    - Learnable routing weights (softmax over components)
    - Context-aware decisions
    - Differentiable
    """
    
    def __init__(self, d: int, num_components: int = 4, hidden_size: int = 128):
        super().__init__()
        self.d = d
        self.num_components = num_components
        self.hidden_size = hidden_size
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(d, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_components),
            nn.Softmax(dim=-1),
        )
        
        logger.info(f"AdaptiveRouter initialized (d={d}, components={num_components})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute routing weights.
        
        Args:
            x: Input of shape (seq_len, d)
        
        Returns:
            Routing weights of shape (seq_len, num_components)
        """
        return self.router(x)


# =============================================================================
# CROSS-ATTENTION FUSION
# =============================================================================

class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention over sequence positions.
    """
    
    def __init__(self, d: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        
        assert d % num_heads == 0, f"d must be divisible by num_heads"
        
        # Attention projections
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        
        # Output projection
        self.out_proj = nn.Linear(d, d)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"CrossAttentionFusion initialized (d={d}, heads={num_heads})")
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention fusion.
        
        Args:
            query: Query tensor of shape (seq_len, d)
            key: Key tensor of shape (seq_len, d)
            value: Value tensor of shape (seq_len, d)
        
        Returns:
            Fused output of shape (seq_len, d)
        """
        seq_len = query.shape[0]
        
        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention: (heads, seq_len, head_dim)
        Q = Q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        # Reshape back: (seq_len, d)
        output = output.transpose(0, 1).contiguous().view(seq_len, self.d)
        
        return self.out_proj(output)


# =============================================================================
# MEMORY COMPRESSOR
# =============================================================================

class MemoryCompressor(nn.Module):
    """
    Learnable low-rank compression / decompression pair.
    
    Note: decompression is a learned projection, NOT lossless.
    """
    
    def __init__(self, d: int, compression_factor: float = 0.5):
        super().__init__()
        self.d = d
        self.compressed_d = int(d * compression_factor)
        
        # Compression network
        self.compress = nn.Linear(d, self.compressed_d)
        
        # Decompression network
        self.decompress = nn.Linear(self.compressed_d, d)
        
        logger.info(f"MemoryCompressor initialized (d={d}, compressed_d={self.compressed_d})")
    
    def forward(self, x: torch.Tensor, compress: bool = True) -> torch.Tensor:
        """
        Compress or decompress memory.
        
        Args:
            x: Input tensor of shape (..., d) when compressing,
               (..., compressed_d) when decompressing
            compress: If True, compress; else decompress
        
        Returns:
            Output tensor
        """
        if compress:
            return self.compress(x)
        else:
            return self.decompress(x)


# =============================================================================
# ULTRA-FLUID LAYER (Single Layer)
# =============================================================================

class UltraFluidLayer(nn.Module):
    """
    Single Ultra-Fluid layer with all enhancements.
    """
    
    def __init__(self, config: UltraFluidConfig):
        super().__init__()
        self.config = config
        config.validate()
        
        # Import base components
        from experiments.superfluid_mamba_layer import (
            CryptographicSession,
            PerfectHashStore,
            AdaptiveGatedSheaf,
            MambaSSM,
            SuperfluidDiffusion,
            HashMode,
            SecurityLevel,
        )
        
        # Initialize session
        self.session = CryptographicSession(
            security_level=SecurityLevel[config.security_level.upper()]
        )
        
        # Initialize store
        self.store = PerfectHashStore(
            initial_capacity=config.capacity,
            max_capacity=config.max_capacity,
            hash_mode=HashMode.DOUBLE_64,
            session=self.session,
        )
        
        # Initialize components
        self.sheaf = AdaptiveGatedSheaf(
            M=config.M,
            r=config.r,
            d=config.d,
            k=config.superfluid_k,
        )
        
        self.mamba = MambaSSM(
            d_model=config.d,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            dt_rank=config.mamba_dt_rank,
        )
        
        self.superfluid = SuperfluidDiffusion(
            M=config.M,
            d=config.d,
            viscosity=config.superfluid_viscosity,
            decay=config.superfluid_decay,
            k=config.superfluid_k,
        )
        
        # Ultra-Fluid enhancements
        if config.use_adaptive_routing:
            self.router = AdaptiveRouter(
                d=config.d,
                num_components=4,  # exact, lossy, mamba, superfluid
                hidden_size=config.routing_hidden,
            )
        else:
            self.router = None
        
        if config.use_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                d=config.d,
                num_heads=4,
            )
            # FIX: the K/V projection used to be a fresh random nn.Linear
            # created INSIDE forward() -- unregistered, untrained, and
            # different on every call.  Register it here instead.
            self.kv_proj = nn.Linear(3 * config.d, config.d)
        else:
            self.cross_attention = None
            self.kv_proj = None
        
        # Memory compression
        if config.compress_state:
            self.compressor = MemoryCompressor(
                d=config.d,
                compression_factor=config.compression_factor,
            )
        else:
            self.compressor = None
        
        # Output projections (ZERO initialized for no-op)
        self.W_e = nn.Parameter(torch.zeros(config.d, config.d))
        self.W_l = nn.Parameter(torch.zeros(config.d, config.d))
        self.W_m = nn.Parameter(torch.zeros(config.d, config.d))
        self.W_sf = nn.Parameter(torch.zeros(config.d, config.d))
        
        # Context buffer
        self._context_buffer: List[int] = []
        
        # Statistics
        self.total_tokens = 0
        self.total_hits = 0
        self.total_misses = 0
        self.start_time = time.time()
        
        logger.info("UltraFluidLayer initialized successfully!")
    
    def forward(
        self,
        h: torch.Tensor,
        content: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through Ultra-Fluid layer.
        
        Args:
            h: Hidden states of shape (seq_len, d)
            content: Content embeddings of shape (seq_len, d)
            token_ids: Token IDs of shape (seq_len,)
        
        Returns:
            Tuple of (output_hidden_states, info_dict)
        """
        seq_len, d = h.shape
        device = h.device
        
        # Convert to numpy for FluidLayer operations
        h_np = h.detach().cpu().numpy().astype(np.float32)
        content_np = content.detach().cpu().numpy().astype(np.float32)
        token_ids_np = token_ids.detach().cpu().numpy()
        
        # Process exact tier
        out_e, info_e = self._process_exact_tier(h_np, content_np, token_ids_np)
        out_e = torch.tensor(out_e, dtype=torch.float32, device=device)
        
        # Process lossy tier
        out_l, info_l = self._process_lossy_tier(h_np, content_np, token_ids_np)
        out_l = torch.tensor(out_l, dtype=torch.float32, device=device)
        
        # Process Mamba tier
        out_m = self.mamba(h.unsqueeze(0)).squeeze(0)
        
        # Process Superfluid tier
        h_for_sf = h.mean(dim=0, keepdim=True).expand(self.config.M, -1)
        state_sf = self.superfluid(h_for_sf, t=1.0)
        # FIX: (state_sf.mean(dim=0) @ W_sf) is 1-D; .expand(seq_len, -1)
        # on a 1-D tensor raises a RuntimeError.  Unsqueeze first.
        sf_vec = state_sf.mean(dim=0) @ self.W_sf          # (d,)
        out_sf = sf_vec.unsqueeze(0).expand(seq_len, -1)  # (seq_len, d)
        
        # Apply adaptive routing if enabled
        if self.router is not None:
            routing_weights = self.router(h)  # (seq_len, 4)
            
            out_e = out_e * routing_weights[:, 0].unsqueeze(1)
            out_l = out_l * routing_weights[:, 1].unsqueeze(1)
            out_m = out_m * routing_weights[:, 2].unsqueeze(1)
            out_sf = out_sf * routing_weights[:, 3].unsqueeze(1)
        
        # Apply cross-attention fusion if enabled
        if self.cross_attention is not None:
            # Use Mamba output as query; exact/lossy/superfluid as K/V
            kv_input = torch.cat([out_e, out_l, out_sf], dim=-1)  # (seq_len, 3*d)
            kv = self.kv_proj(kv_input)  # (seq_len, d)
            out_m = self.cross_attention(out_m, kv, kv)
        
        # Combine all tiers
        tiers = (
            out_e @ self.W_e.T
            + out_l @ self.W_l.T
            + out_m @ self.W_m.T
            + out_sf @ self.W_sf.T
        )
        # FIX: the use_residual True/False branches were IDENTICAL code;
        # now the non-residual path actually omits the residual.
        if self.config.use_residual:
            h_prime = h + tiers
        else:
            h_prime = tiers
        
        # Update statistics
        self.total_tokens += seq_len
        self.total_hits += info_e.get('hits', 0)
        self.total_misses += info_e.get('misses', 0)
        
        info = {
            'exact_tier': info_e,
            'lossy_tier': info_l,
            'total_tokens': self.total_tokens,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'recall_rate': self.total_hits / (self.total_hits + self.total_misses + 1e-12),
        }
        
        return h_prime, info
    
    def _process_exact_tier(self, h, content, token_ids):
        """Process exact tier"""
        T = token_ids.shape[0]
        out_e = np.zeros((T, self.config.d))
        hits = 0
        
        for t in range(T):
            prev_tokens = self._context_buffer[-(self.config.ctx_len):] if self._context_buffer else []
            if len(prev_tokens) >= self.config.ctx_len:
                prev_key = tuple(prev_tokens[-self.config.ctx_len:])
            else:
                prev_key = None
            
            self._context_buffer.append(int(token_ids[t]))
            
            curr_tokens = self._context_buffer[-(self.config.ctx_len):]
            if len(curr_tokens) >= self.config.ctx_len:
                curr_key = tuple(curr_tokens)
            else:
                curr_key = None
            
            if curr_key is not None:
                value = self.store.get(curr_key, None)
                if value is not None:
                    out_e[t] = value
                    hits += 1
            
            if prev_key is not None:
                self.store.put(prev_key, h[t] if t < T else np.zeros(self.config.d))
        
        return out_e, {'hits': hits, 'misses': T - hits, 'recall_rate': hits / T if T > 0 else 0.0}
    
    def _process_lossy_tier(self, h, content, token_ids):
        """Process lossy tier"""
        T = token_ids.shape[0]
        
        if not hasattr(self, 'state'):
            self.state = np.zeros((self.config.M, self.config.r))
        
        w, _ = self._route(content)
        out_l = None
        
        if True:
            self.state += w.T @ (h @ self._get_Wv())
            state_tensor = torch.tensor(self.state, dtype=torch.float32, requires_grad=False)
            diffused = self.sheaf.apply_diffusion(state_tensor, t=1.0, cheb_m=12)
            self.state = diffused.detach().numpy()
            state_read = w @ self.state
            out_l = state_read @ self._get_Wr()
        
        return out_l if out_l is not None else np.zeros((T, self.config.d)), {}
    
    def _route(self, content):
        """Route content to anchors"""
        T, d = content.shape
        scores = content @ self._get_Wq().T
        w = np.exp(scores) / (np.sum(np.exp(scores), axis=1, keepdims=True) + 1e-12)
        return w, np.max(scores, axis=1)
    
    def _get_Wq(self):
        if not hasattr(self, '_Wq'):
            self._Wq = np.random.randn(self.config.d, self.config.M).astype(np.float32) * 0.02
        return self._Wq
    
    def _get_Wv(self):
        if not hasattr(self, '_Wv'):
            self._Wv = np.random.randn(self.config.d, self.config.r).astype(np.float32) * 0.02
        return self._Wv
    
    def _get_Wr(self):
        if not hasattr(self, '_Wr'):
            self._Wr = np.random.randn(self.config.r, self.config.d).astype(np.float32) * 0.02
        return self._Wr


# =============================================================================
# ULTRA-FLUID STACK (Multi-Layer)
# =============================================================================

class UltraFluidStack(nn.Module):
    """
    Stack of multiple Ultra-Fluid layers.
    
    Each UltraFluidLayer.forward() already adds its own residual
    (h + tiers) when use_residual=True, so the stack applies the layer
    output directly.  (The previous version added h + skip * h_layer on
    top of the layer's internal residual, double-counting the input.)
    """
    
    def __init__(self, config: UltraFluidConfig):
        super().__init__()
        self.config = config
        
        # Create stack of layers
        self.layers = nn.ModuleList([
            UltraFluidLayer(config) for _ in range(config.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.d) for _ in range(config.num_layers)
        ])
        
        logger.info(f"UltraFluidStack initialized with {config.num_layers} layers")
    
    def forward(
        self,
        h: torch.Tensor,
        content: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Forward pass through stack of Ultra-Fluid layers.
        
        Args:
            h: Hidden states of shape (seq_len, d)
            content: Content embeddings of shape (seq_len, d)
            token_ids: Token IDs of shape (seq_len,)
        
        Returns:
            Tuple of (output_hidden_states, list_of_info_dicts)
        """
        all_info = []
        
        for layer, norm in zip(self.layers, self.layer_norms):
            h_norm = norm(h)
            h_layer, info = layer(h_norm, content, token_ids)
            all_info.append(info)
            # The layer already contains the residual connection.
            h = h_layer
        
        return h, all_info


# =============================================================================
# COMPRESSED ULTRA-FLUID (Memory Optimized)
# =============================================================================

class CompressedUltraFluid(nn.Module):
    """
    HONEST status: this variant is currently a PASS-THROUGH.  The
    compressors are instantiated but the forward pass simply delegates to
    the base layer; no memory reduction happens yet.  The earlier
    "50-80% reduction" claim was not true.
    """
    
    def __init__(self, config: UltraFluidConfig):
        super().__init__()
        self.config = config
        
        # Create base Ultra-Fluid layer
        self.base_layer = UltraFluidLayer(config)
        
        # Compression for each component (not yet used in forward)
        self.compressors = nn.ModuleDict({
            'exact': MemoryCompressor(config.d, config.compression_factor),
            'lossy': MemoryCompressor(config.r, config.compression_factor),
            'mamba': MemoryCompressor(config.mamba_d_state, config.compression_factor),
            'superfluid': MemoryCompressor(config.d, config.compression_factor),
        })
        
        logger.info("CompressedUltraFluid initialized (pass-through; compression not wired)")
    
    def forward(self, h, content, token_ids):
        """Forward pass (currently delegates to the base layer)"""
        return self.base_layer(h, content, token_ids)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_ultra_fluid_layer(
    d: int = 64,
    M: int = 64,
    r: int = 8,
    num_layers: int = 2,
    use_residual: bool = True,
    use_cross_attention: bool = True,
    use_adaptive_routing: bool = True,
) -> UltraFluidStack:
    """
    Factory function to create Ultra-Fluid layer stack.
    """
    config = UltraFluidConfig(
        M=M,
        r=r,
        d=d,
        ctx_len=12,
        mamba_d_state=16,
        mamba_d_conv=4,
        superfluid_viscosity=0.1,
        superfluid_decay=0.01,
        superfluid_k=8,
        num_layers=num_layers,
        use_residual=use_residual,
        use_cross_attention=use_cross_attention,
        use_adaptive_routing=use_adaptive_routing,
        routing_hidden=128,
        compress_state=True,
        compression_factor=0.5,
    )
    
    return UltraFluidStack(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ULTRA-FLUID LAYER - INITIALIZATION TEST")
    print("=" * 80)
    
    # Test single layer
    print("\nTesting single UltraFluidLayer...")
    try:
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
        print("OK - Single layer initialized")
        
        h = torch.randn(5, 8)
        content = torch.randn(5, 8)
        token_ids = torch.randint(0, 100, (5,))
        
        output, info = layer(h, content, token_ids)
        print(f"OK - Forward pass successful (input: {h.shape}, output: {output.shape})")
        print(f"   Recall rate (fresh sequence, expected low): {info['recall_rate']:.2%}")
        
    except Exception as e:
        print(f"FAIL - Single layer test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test multi-layer stack
    print("\nTesting UltraFluidStack (2 layers)...")
    try:
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
        print("OK - Stack initialized")
        
        h = torch.randn(5, 8)
        content = torch.randn(5, 8)
        token_ids = torch.randint(0, 100, (5,))
        
        output, all_info = stack(h, content, token_ids)
        print(f"OK - Stack forward pass successful (input: {h.shape}, output: {output.shape})")
        print(f"   Layers: {len(all_info)}")
        
    except Exception as e:
        print(f"FAIL - Stack test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test compressed version
    print("\nTesting CompressedUltraFluid (pass-through)...")
    try:
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
        print("OK - Compressed layer initialized")
        
        h = torch.randn(5, 8)
        content = torch.randn(5, 8)
        token_ids = torch.randint(0, 100, (5,))
        
        output, info = compressed(h, content, token_ids)
        print(f"OK - Compressed forward pass successful (input: {h.shape}, output: {output.shape})")
        
    except Exception as e:
        print(f"FAIL - Compressed test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE (see test_ultra_fluid.py for assertions)")
    print("=" * 80)
    print("\nAll performance numbers for this layer are HYPOTHETICAL (unmeasured).")
