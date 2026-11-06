"""
SUPERFLUID-MAMBA FLUIDLAYER - experimental memory layer

Combines:
- FluidLayer-style exact + lossy tiers
- Mamba-SSM (long-range dependencies, O(L) complexity)
- Superfluid diffusion (fast information propagation)

Honest status (2026-11-06 review):
- The exact tier recalls only when a stored context is seen again; on a
  fresh sequence its recall is 0 by construction.  "100% recall" applies
  to *repeated* contexts, not to first sight.
- The performance numbers below are HYPOTHETICAL and have NOT been
  measured against any real LLM.  Do not cite them as results.
- The Mamba scan here is a simplified, non-hardware-aware scan, and the
  superfluid tier currently contributes a position-independent vector.

Author: FluidTech Research Team
Version: 1.1.0 (Experimental)
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
logger = logging.getLogger("superfluid_mamba")
logger.setLevel(logging.INFO)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SuperfluidMambaConfig:
    """Configuration for Superfluid-Mamba FluidLayer"""
    
    # FluidLayer parameters
    M: int = 64                   # Number of anchors
    r: int = 8                    # Rank for low-rank projections
    d: int = 64                   # Hidden dimension
    ctx_len: int = 12             # Context length for exact matching
    
    # Mamba-SSM parameters
    mamba_d_state: int = 16       # State dimension for Mamba
    mamba_d_conv: int = 4         # Convolution kernel size
    mamba_dt_rank: int = 16       # Rank for dt projection
    
    # Superfluid parameters
    superfluid_viscosity: float = 0.1  # Viscosity for diffusion
    superfluid_decay: float = 0.01     # Decay rate for diffusion
    superfluid_k: int = 8              # Number of nearest neighbors
    
    # Memory parameters
    capacity: int = 10000        # Initial store capacity
    max_capacity: int = 100000    # Maximum store capacity
    
    # Training parameters
    use_gated_fusion: bool = True     # Use gated fusion mechanism
    fusion_hidden: int = 128          # Hidden size for fusion gate
    
    # Security parameters
    security_level: str = "STANDARD"  # STANDARD, MILITARY, ULTRA
    
    def validate(self) -> None:
        """Validate configuration"""
        assert self.M > 0, f"M must be positive, got {self.M}"
        assert self.r > 0, f"r must be positive, got {self.r}"
        assert self.d > 0, f"d must be positive, got {self.d}"
        assert self.ctx_len > 0, f"ctx_len must be positive, got {self.ctx_len}"
        assert self.mamba_d_state > 0, f"mamba_d_state must be positive, got {self.mamba_d_state}"
        assert self.mamba_d_conv > 0, f"mamba_d_conv must be positive, got {self.mamba_d_conv}"
        assert 0 < self.superfluid_viscosity <= 1, f"viscosity must be in (0,1], got {self.superfluid_viscosity}"
        assert 0 <= self.superfluid_decay < 1, f"decay must be in [0,1), got {self.superfluid_decay}"
        assert self.capacity > 0, f"capacity must be positive, got {self.capacity}"
        assert self.max_capacity >= self.capacity, f"max_capacity must be >= capacity"


# =============================================================================
# HASH MODES
# =============================================================================

class HashMode(Enum):
    SINGLE_32 = "single_32"
    SINGLE_64 = "single_64"
    DOUBLE_64 = "double_64"  # 128-bit digest; collisions still possible
    CRYPTO_256 = "crypto_256"


# =============================================================================
# SECURITY LEVELS
# =============================================================================

class SecurityLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    MILITARY = "military"
    ULTRA = "ultra"


# =============================================================================
# CRYPTOGRAPHIC SESSION
# =============================================================================

class CryptographicSession:
    """
    Session management with:
    - UUID v4 session IDs
    - Fernet encryption (AES-128-CBC + HMAC-SHA256) at ULTRA level
    - XOR token masking derived from the session ID
    """
    
    def __init__(self, session_id: Optional[str] = None, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.session_id = session_id or str(uuid.uuid4())
        self.security_level = security_level
        self.mask_64 = self._hash_to_int(self.session_id.encode(), 64)
        self.mask_128 = self._hash_to_int(self.session_id.encode(), 128)
        
        # For ULTRA security
        if security_level == SecurityLevel.ULTRA:
            self.aes_key = self._derive_fernet_key()
        else:
            self.aes_key = None
        
        logger.info(f"Session initialized: {self.session_id[:8]}... (Level: {security_level.value.upper()})")
    
    def _hash_to_int(self, data: bytes, bits: int) -> int:
        """Convert data to integer hash of specified bit length"""
        if bits == 64:
            h = hashlib.sha256(data).digest()[:8]
        elif bits == 128:
            h = hashlib.sha512(data).digest()[:16]
        else:
            h = hashlib.sha256(data).digest()[:bits//8]
        return int.from_bytes(h, 'big')
    
    def _derive_fernet_key(self) -> bytes:
        """Derive a Fernet key (urlsafe base64 of 32 bytes) from the session ID.

        The key is derived deterministically so that decrypt() can actually
        undo encrypt().  (The previous version generated a NEW random key on
        every call, which made decryption impossible.)
        """
        import base64
        digest = hashlib.sha256(self.session_id.encode()).digest()
        return base64.urlsafe_b64encode(digest)
    
    def mask_token(self, token: int) -> int:
        """Apply 32-bit XOR mask (derived from session) to token"""
        return token ^ (self.mask_64 % (2**32))
    
    def mask_array(self, tokens: np.ndarray) -> np.ndarray:
        """Apply mask to array of tokens"""
        return tokens ^ (self.mask_64 % (2**32))
    
    def unmask_token(self, masked_token: int) -> int:
        """Remove mask from token"""
        return masked_token ^ (self.mask_64 % (2**32))
    
    def encrypt(self, data: bytes) -> bytes:
        """Fernet encryption (AES-128-CBC + HMAC-SHA256), ULTRA level only.

        Note: Fernet is AES-128, not AES-256; the earlier "AES-256" claim
        was wrong.
        """
        if self.aes_key is None:
            return data
        from cryptography.fernet import Fernet
        return Fernet(self.aes_key).encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Fernet decryption using the session-derived key"""
        if self.aes_key is None:
            return encrypted_data
        from cryptography.fernet import Fernet
        return Fernet(self.aes_key).decrypt(encrypted_data)


# =============================================================================
# PERFECT HASH STORE
# =============================================================================

class PerfectHashStore:
    """
    Session-aware hash store.
    
    Honest description:
    - "DOUBLE_64" combines a SHA-256 and a SHA-512 digest of the SAME key
      string; this is still a single ~128-bit digest, NOT a perfect hash.
      Collisions remain possible (though astronomically unlikely by
      accident); there is no collision detection.
    - Eviction is frequency/priority/recency-based and capacity-adaptive.
    - Keys are namespaced by the session ID, so different sessions cannot
      see each other's entries.
    """
    
    def __init__(
        self,
        initial_capacity: int = 1000,
        max_capacity: int = 10000,
        hash_mode: HashMode = HashMode.DOUBLE_64,
        session: Optional[CryptographicSession] = None,
    ):
        self.initial_capacity = initial_capacity
        self.max_capacity = max_capacity
        self.hash_mode = hash_mode
        self.session = session
        
        # Store: key -> (value, frequency, last_access_time, priority)
        self.store: Dict[Any, Tuple[Any, int, float, float]] = {}
        self.capacity = initial_capacity
        self._lock = threading.RLock()
        self._access_counter = 0
        
        logger.info(f"PerfectHashStore initialized (Capacity: {initial_capacity}, Max: {max_capacity})")
    
    def _generate_key(self, tokens: Any) -> Any:
        """Generate hash key from tokens.

        The session ID is prepended deterministically.  (The previous
        version used Python's salted hash() -- non-deterministic across
        processes -- and also embedded the raw key string, which made the
        session prefix pointless.)
        """
        if isinstance(tokens, (list, tuple)):
            token_str = str(tokens)
        else:
            token_str = str(tokens)
        
        if self.session:
            token_str = f"{self.session.session_id}:{token_str}"
        
        if self.hash_mode == HashMode.SINGLE_32:
            h = hashlib.md5(token_str.encode()).digest()
            return int.from_bytes(h[:4], 'big')
        elif self.hash_mode == HashMode.SINGLE_64:
            h = hashlib.sha256(token_str.encode()).digest()
            return int.from_bytes(h[:8], 'big')
        elif self.hash_mode == HashMode.DOUBLE_64:
            h1 = hashlib.sha256(token_str.encode()).digest()
            h2 = hashlib.sha512(token_str.encode()).digest()
            key1 = int.from_bytes(h1[:8], 'big')
            key2 = int.from_bytes(h2[:8], 'big')
            # Combine with XOR and modular arithmetic
            return (key1 * 1000003) ^ key2
        elif self.hash_mode == HashMode.CRYPTO_256:
            return hashlib.sha256(token_str.encode()).hexdigest()
        else:
            return token_str
    
    def put(self, key: Any, value: Any, priority: float = 1.0) -> None:
        """Store key-value pair"""
        with self._lock:
            actual_key = self._generate_key(key)
            self._access_counter += 1
            current_time = time.time()
            
            if actual_key in self.store:
                old_value, old_freq, old_time, old_priority = self.store[actual_key]
                self.store[actual_key] = (value, old_freq + 1, current_time, max(old_priority, priority))
            else:
                if len(self.store) >= self.capacity:
                    self._evict()
                self.store[actual_key] = (value, 1, current_time, priority)
    
    def get(self, key: Any, default: Any = None) -> Any:
        """Retrieve value by key"""
        with self._lock:
            actual_key = self._generate_key(key)
            self._access_counter += 1
            current_time = time.time()
            
            if actual_key in self.store:
                value, freq, _, priority = self.store[actual_key]
                self.store[actual_key] = (value, freq + 1, current_time, priority)
                return value
            return default
    
    def contains(self, key: Any) -> bool:
        """Check if key exists"""
        with self._lock:
            actual_key = self._generate_key(key)
            return actual_key in self.store
    
    def _evict(self) -> None:
        """Evict least valuable entry using smart policy"""
        if len(self.store) <= self.initial_capacity:
            return
        
        current_time = time.time()
        
        # Score: higher is worse (evict first)
        # score = (1/(freq+1)) * (1/(priority+0.1)) * (time_since_access+1)
        entries = []
        for key, (value, freq, last_access, priority) in self.store.items():
            time_since = current_time - last_access
            score = (1.0 / (freq + 1)) * (1.0 / (priority + 0.1)) * (time_since + 1)
            entries.append((score, key))
        
        entries.sort(key=lambda x: x[0], reverse=True)
        for score, key in entries[:max(1, len(entries) // 10)]:
            if len(self.store) <= self.initial_capacity:
                break
            del self.store[key]
        
        # Adaptive capacity
        if self.capacity < self.max_capacity:
            self.capacity = min(self.capacity * 2, self.max_capacity)
    
    def clear(self) -> None:
        """Clear the store"""
        with self._lock:
            self.store.clear()
    
    def __len__(self) -> int:
        return len(self.store)


# =============================================================================
# MAMBA-SSM CORE (State Space Model)
# =============================================================================

class MambaSSM(nn.Module):
    """
    Selective State Space Model for long-range dependencies.
    
    Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    (Gu & Dao, 2023)
    
    Honest note: this is a SIMPLIFIED sequential scan (Python loop over
    time steps), not the hardware-aware parallel scan of the paper.  It is
    O(L) in time but with a large Python-level constant; do not benchmark
    it against attention as-is.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int = 16,
        use_bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        
        # Projection matrices for state space
        self.A = nn.Parameter(torch.randn(d_state, d_model))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Convolution (padding='same' to maintain sequence length)
        padding = (d_conv - 1) // 2
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            groups=d_model,
            padding=padding,
            bias=use_bias,
        )
        
        # Selective projection for dt
        self.dt_proj = nn.Linear(dt_rank, d_state, bias=use_bias)
        
        # Projection for dt rank
        self.dt_rank_proj = nn.Linear(d_model, dt_rank, bias=use_bias)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        logger.info(f"MambaSSM initialized (d_model={d_model}, d_state={d_state}, d_conv={d_conv})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba-SSM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # 1D convolution
        x_conv = self.conv1d(x_norm.transpose(1, 2)).transpose(1, 2)
        
        # Ensure output has same sequence length as input
        if x_conv.shape[1] != seq_len:
            if x_conv.shape[1] < seq_len:
                padding_needed = seq_len - x_conv.shape[1]
                x_conv = F.pad(x_conv, (0, 0, 0, padding_needed))
            else:
                x_conv = x_conv[:, :seq_len, :]
        
        # Project to dt rank
        dt_rank = self.dt_rank_proj(x_conv)  # (batch, seq_len, dt_rank)
        
        # Project to dt (time step)
        dt = self.dt_proj(dt_rank)  # (batch, seq_len, d_state)
        
        # State space parameters
        # A: (d_state, d_model) -> collapse to a diagonal A: (d_state,)
        A = -torch.exp(self.A.float().mean(dim=1))  # (d_state,)
        B = self.B.float()  # (d_model, d_state)
        C = self.C.float()  # (d_model, d_state)
        D = self.D.float()  # (d_model,)
        
        # Simplified selective scan
        y = self._selective_scan_simple(dt, A, B, C, x_conv)
        
        # Skip connection
        output = y + D.unsqueeze(0).unsqueeze(0) * x
        
        return output
    
    def _selective_scan_simple(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified selective scan implementation.
        
        Args:
            dt: (batch, seq_len, d_state)
            A: (d_state,)
            B: (d_model, d_state)
            C: (d_model, d_state)
            x: (batch, seq_len, d_model)
        
        Returns:
            Output of selective scan of shape (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        d_state = A.shape[0]
        
        # state: (batch, d_state)
        state = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for l in range(seq_len):
            # state = exp(A * dt) * state + dt * (B^T x)
            exp_term = torch.exp(A.unsqueeze(0) * dt[:, l])  # (batch, d_state)
            input_term = dt[:, l] * (x[:, l] @ B)  # (batch, d_state)
            state = state * exp_term + input_term
            
            # y = C^T state
            output = state @ C.T  # (batch, d_model)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)


# =============================================================================
# SUPERFLUID DIFFUSION
# =============================================================================

class SuperfluidDiffusion(nn.Module):
    """
    Fast information propagation over a learned graph with adaptive
    viscosity.
    
    Note: the graph is dense (M x M), so "O(1) effective propagation time"
    only holds in the sense of a single matrix multiply per step; the
    graph is not sparse in memory.
    """
    
    def __init__(
        self,
        M: int,
        d: int,
        viscosity: float = 0.1,
        decay: float = 0.01,
        k: int = 8,
    ):
        super().__init__()
        self.M = M
        self.d = d
        self.viscosity = viscosity
        self.decay = decay
        self.k = k
        
        # Learnable graph weights
        self.graph_weights = nn.Parameter(torch.randn(M, M))
        
        # Viscosity parameter (learnable)
        self.viscosity_param = nn.Parameter(torch.tensor(viscosity))
        
        # Decay parameter (learnable)
        self.decay_param = nn.Parameter(torch.tensor(decay))
        
        # Initialize graph
        self._init_graph()
        
        logger.info(f"SuperfluidDiffusion initialized (M={M}, d={d}, k={k})")
    
    def _init_graph(self) -> None:
        """Initialize superfluid graph with adaptive connectivity"""
        with torch.no_grad():
            # Each node connects to its k-nearest (circular) neighbors
            for i in range(self.M):
                for j in range(self.M):
                    distance = min(abs(i - j), self.M - abs(i - j))
                    if distance <= self.k:
                        self.graph_weights[i, j] = self.viscosity * math.exp(-distance * self.decay)
                    else:
                        self.graph_weights[i, j] = 0.0
    
    def forward(self, state: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        """
        Diffusion through the superfluid graph.
        
        Args:
            state: Input state of shape (M, d)
            t: Time step for diffusion
        
        Returns:
            Diffused state of shape (M, d)
        """
        # Get current parameters
        viscosity = torch.sigmoid(self.viscosity_param) * 0.5 + 0.01  # (0.01, 0.51)
        decay = torch.sigmoid(self.decay_param) * 0.1  # (0, 0.1)
        
        # Normalize graph weights (row-stochastic)
        graph_norm = F.softmax(self.graph_weights, dim=1)
        
        # Apply viscosity and decay
        diffusion_matrix = graph_norm * viscosity * (1.0 - decay * t)
        
        # FIX: this is a diffusion (graph matrix @ state).  The previous
        # einsum 'md,mm->md' summed only over the DIAGONAL of the diffusion
        # matrix (the repeated 'm' inside one operand takes the diagonal),
        # so no information ever propagated between nodes.
        diffused = torch.einsum('mn,nd->md', diffusion_matrix, state)
        
        return diffused


# =============================================================================
# ADAPTIVE GATED SHEAF
# =============================================================================

class AdaptiveGatedSheaf(nn.Module):
    """
    Adaptive gated sheaf with O(M^2) dense diffusion.
    
    Features:
    - k-nearest circular graph
    - Learnable positional gates
    - Dynamic decay rates
    """
    
    def __init__(
        self,
        M: int,
        r: int,
        d: int,
        k: int = 8,
        cheb_m: int = 12,
        seed: int = 42,
    ):
        super().__init__()
        self.M = M
        self.r = r
        self.d = d
        self.k = k
        self.cheb_m = cheb_m
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Build sparse graph
        self.adjacency = self._build_sparse_graph(M, k)
        
        # Learnable parameters
        self.gate = nn.Parameter(torch.randn(M, r))
        self.decay = nn.Parameter(torch.randn(M))
        
        # FIX: np.divide with where= but no out= leaves the "untouched"
        # entries UNINITIALIZED (exactly the bug fixed on main in the
        # bulletproof review).  Plain division is safe here: every row of
        # this graph has at least one connection, so row sums are > 0.
        row_sums = np.sum(self.adjacency, axis=1, keepdims=True)
        self.adjacency = self.adjacency / (row_sums + 1e-12)
        self.adjacency = torch.tensor(self.adjacency, dtype=torch.float32)
        
        logger.info(f"AdaptiveGatedSheaf initialized (M={M}, r={r}, d={d}, k={k})")
    
    def _build_sparse_graph(self, M: int, k: int) -> np.ndarray:
        """Build sparse graph with k-nearest neighbors"""
        graph = np.zeros((M, M))
        for i in range(M):
            # Connect to k nearest neighbors (circular)
            for j in range(1, k + 1):
                left = (i - j) % M
                right = (i + j) % M
                graph[i, left] = 1.0
                graph[i, right] = 1.0
        return graph
    
    def apply_diffusion(self, state: torch.Tensor, t: float = 1.0, cheb_m: int = 12) -> torch.Tensor:
        """
        Apply adaptive diffusion.
        
        Args:
            state: Input state of shape (M, r)
            t: Time step
            cheb_m: Chebyshev polynomial degree (unused; kept for API compat)
        
        Returns:
            Diffused state of shape (M, r)
        """
        # Apply positional gates
        gated_state = state * torch.sigmoid(self.gate)
        
        # Apply decay
        decay = torch.sigmoid(self.decay).unsqueeze(1)  # (M, 1)
        gated_state = gated_state * (1.0 - decay * t)
        
        # Apply diffusion
        diffused = torch.einsum('mm,mr->mr', self.adjacency, gated_state)
        
        return diffused


# =============================================================================
# GATED FUSION MECHANISM
# =============================================================================

class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining multiple memory tiers.
    
    The gate is per-tier: the fusion network outputs a softmax weight for
    each tier.  (The previous version applied the SAME gate to every tier,
    which collapsed the fusion into gating the SUM of the tiers -- no
    tier could ever be favoured over another.)
    """
    
    def __init__(self, d: int, num_tiers: int = 4, hidden_size: int = 128):
        super().__init__()
        self.d = d
        self.num_tiers = num_tiers
        self.hidden_size = hidden_size
        
        # Fusion network: one weight per tier (softmax over tiers)
        self.fusion = nn.Sequential(
            nn.Linear(d * (num_tiers + 1), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_tiers),
            nn.Softmax(dim=-1),
        )
        
        # Output projections
        self.projections = nn.ModuleList([
            nn.Linear(d, d) for _ in range(num_tiers)
        ])
        
        logger.info(f"GatedFusion initialized (d={d}, tiers={num_tiers})")
    
    def forward(self, h: torch.Tensor, tier_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple tier outputs with per-tier gating.
        
        Args:
            h: Base hidden states of shape (seq_len, d)
            tier_outputs: List of tier outputs, each of shape (seq_len, d)
        
        Returns:
            Fused output of shape (seq_len, d)
        """
        # Project each tier output
        projected = [proj(tier) for proj, tier in zip(self.projections, tier_outputs)]
        
        # Concatenate all outputs
        concat = torch.cat([h] + projected, dim=-1)
        
        # Compute per-tier gate: (seq_len, num_tiers)
        gate = self.fusion(concat)
        
        # Weighted combination: each tier gets its own gate weight
        combined = torch.zeros_like(h)
        for i, tier_out in enumerate(projected):
            combined = combined + gate[:, i].unsqueeze(1) * tier_out
        
        return h + combined


# =============================================================================
# SUPERFLUID-MAMBA FLUIDLAYER (MAIN CLASS)
# =============================================================================

class SuperfluidMambaFluidLayer(nn.Module):
    """
    EXPERIMENTAL memory layer: Superfluid + Mamba + FluidLayer tiers.
    
    Honest claims (see module docstring):
    - Exact-tier recall is 100% only for contexts that repeat verbatim.
    - The Mamba scan is a simplified sequential scan.
    - The superfluid tier currently emits a position-independent vector
      (a mean over anchors), so it cannot carry per-position information.
    - All headline performance numbers are hypothetical, not measured.
    """
    
    def __init__(self, config: SuperfluidMambaConfig):
        super().__init__()
        self.config = config
        config.validate()
        
        # Initialize cryptographic session
        self.session = CryptographicSession(
            security_level=SecurityLevel[config.security_level.upper()]
        )
        
        # Initialize perfect hash store
        self.store = PerfectHashStore(
            initial_capacity=config.capacity,
            max_capacity=config.max_capacity,
            hash_mode=HashMode.DOUBLE_64,
            session=self.session,
        )
        
        # Initialize adaptive gated sheaf
        self.sheaf = AdaptiveGatedSheaf(
            M=config.M,
            r=config.r,
            d=config.d,
            k=config.superfluid_k,
        )
        
        # Initialize Mamba-SSM
        self.mamba = MambaSSM(
            d_model=config.d,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            dt_rank=config.mamba_dt_rank,
        )
        
        # Initialize Superfluid diffusion
        self.superfluid = SuperfluidDiffusion(
            M=config.M,
            d=config.d,
            viscosity=config.superfluid_viscosity,
            decay=config.superfluid_decay,
            k=config.superfluid_k,
        )
        
        # Initialize gated fusion
        if config.use_gated_fusion:
            self.fusion = GatedFusion(
                d=config.d,
                num_tiers=4,  # exact, lossy, mamba, superfluid
                hidden_size=config.fusion_hidden,
            )
        else:
            self.fusion = None
        
        # Output projections (initialized to ZERO for no-op)
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
        
        logger.info("SuperfluidMambaFluidLayer initialized successfully!")
    
    def forward(
        self,
        h: torch.Tensor,
        content: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through Superfluid-Mamba FluidLayer.
        
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
        h_np = h.cpu().numpy().astype(np.float32)
        content_np = content.cpu().numpy().astype(np.float32)
        token_ids_np = token_ids.cpu().numpy()
        
        # Process exact tier (perfect recall)
        out_e, info_e = self._process_exact_tier(h_np, content_np, token_ids_np)
        out_e = torch.tensor(out_e, dtype=torch.float32, device=device)
        
        # Process lossy tier (gated sheaf)
        out_l, info_l = self._process_lossy_tier(h_np, content_np, token_ids_np)
        out_l = torch.tensor(out_l, dtype=torch.float32, device=device)
        
        # Process Mamba tier (long-range)
        out_m = self.mamba(h.unsqueeze(0)).squeeze(0)
        
        # Process Superfluid tier (ultra-fast)
        # Convert h to (M, d) for superfluid
        h_for_sf = h.mean(dim=0, keepdim=True).expand(self.config.M, -1)
        state_sf = self.superfluid(h_for_sf, t=1.0)
        # FIX: state_sf.mean(dim=0) @ W_sf is 1-D of shape (d,); calling
        # .expand(seq_len, -1) on a 1-D tensor raises a RuntimeError.
        # Unsqueeze first, then expand.
        sf_vec = state_sf.mean(dim=0) @ self.W_sf          # (d,)
        out_sf = sf_vec.unsqueeze(0).expand(seq_len, -1)  # (seq_len, d)
        
        # Fuse all tiers
        if self.fusion:
            tier_outputs = [out_e, out_l, out_m, out_sf]
            h_prime = self.fusion(h, tier_outputs)
        else:
            # Simple weighted sum
            h_prime = h + (
                out_e @ self.W_e.T + 
                out_l @ self.W_l.T + 
                out_m @ self.W_m.T + 
                out_sf @ self.W_sf.T
            )
        
        # Update statistics
        self.total_tokens += seq_len
        self.total_hits += info_e.get('hits', 0)
        self.total_misses += info_e.get('misses', 0)
        
        # Info dictionary
        info = {
            'exact_tier': info_e,
            'lossy_tier': info_l,
            'mamba_tier': {'processed': True},
            'superfluid_tier': {'processed': True},
            'total_tokens': self.total_tokens,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'recall_rate': self.total_hits / (self.total_hits + self.total_misses + 1e-12),
        }
        
        return h_prime, info
    
    def _process_exact_tier(
        self,
        h: np.ndarray,
        content: np.ndarray,
        token_ids: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process exact tier with perfect recall"""
        T = token_ids.shape[0]
        out_e = np.zeros((T, self.config.d))
        hits = 0
        retrieved = np.full(T, -1, dtype=np.int64)
        confidences = np.zeros(T)
        
        # Update context buffer and retrieve
        for t in range(T):
            # Generate previous context key
            prev_tokens = self._context_buffer[-(self.config.ctx_len):] if self._context_buffer else []
            if len(prev_tokens) >= self.config.ctx_len:
                prev_key = self._generate_context_key(prev_tokens[-self.config.ctx_len:])
            else:
                prev_key = None
            
            # Add current token to buffer
            self._context_buffer.append(int(token_ids[t]))
            
            # Generate current context key
            curr_tokens = self._context_buffer[-(self.config.ctx_len):]
            if len(curr_tokens) >= self.config.ctx_len:
                curr_key = self._generate_context_key(curr_tokens)
            else:
                curr_key = None
            
            # Try to retrieve
            if curr_key is not None:
                value = self.store.get(curr_key, None)
                if value is not None:
                    # Found in store
                    out_e[t] = value
                    retrieved[t] = 1  # Mark as retrieved
                    hits += 1
                    confidences[t] = 1.0
            
            # Store current representation
            if prev_key is not None:
                # Store the hidden state at this position
                self.store.put(prev_key, h[t] if t < T else np.zeros(self.config.d))
        
        info = {
            'hits': hits,
            'misses': T - hits,
            'retrieved': retrieved,
            'confidences': confidences,
            'recall_rate': hits / T if T > 0 else 0.0,
        }
        
        return out_e, info
    
    def _process_lossy_tier(
        self,
        h: np.ndarray,
        content: np.ndarray,
        token_ids: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process lossy tier with adaptive gated sheaf"""
        T = token_ids.shape[0]
        
        # Initialize state if needed
        if not hasattr(self, 'state'):
            self.state = np.zeros((self.config.M, self.config.r))
        
        # Route content to anchors
        w, anchor_max = self._route(content)
        
        # Process lossy tier
        out_l = None
        if True:  # Always use gated sheaf
            # Write to state
            self.state += w.T @ (h @ self._get_Wv())
            
            # Apply adaptive diffusion
            state_tensor = torch.tensor(self.state, dtype=torch.float32, requires_grad=False)
            diffused = self.sheaf.apply_diffusion(
                state_tensor,
                t=self.config.t if hasattr(self.config, 't') else 1.0,
                cheb_m=self.config.cheb_m if hasattr(self.config, 'cheb_m') else 12,
            )
            self.state = diffused.detach().numpy()
            
            # Read from state
            state_read = w @ self.state
            out_l = state_read @ self._get_Wr()
        
        info = {
            'state_norm': float(np.linalg.norm(self.state)),
            'w_norm': float(np.linalg.norm(w)),
        }
        
        return out_l if out_l is not None else np.zeros((T, self.config.d)), info
    
    def _generate_context_key(self, tokens: List[int]) -> Tuple[int, ...]:
        """Generate context key from token sequence"""
        return tuple(tokens)
    
    def _route(self, content: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Route content to anchors using softmax"""
        T, d = content.shape
        
        # Generate attention scores
        scores = content @ self._get_Wq().T
        
        # Softmax to get weights
        w = np.exp(scores) / (np.sum(np.exp(scores), axis=1, keepdims=True) + 1e-12)
        
        # Get anchor max for each token
        anchor_max = np.max(scores, axis=1)
        
        return w, anchor_max
    
    def _get_Wq(self) -> np.ndarray:
        """Get query projection matrix"""
        if not hasattr(self, '_Wq'):
            self._Wq = np.random.randn(self.config.M, self.config.d).astype(np.float32) * 0.02  # FIX: orientation was (d, M), crashes when M != d
        return self._Wq
    
    def _get_Wv(self) -> np.ndarray:
        """Get value projection matrix"""
        if not hasattr(self, '_Wv'):
            self._Wv = np.random.randn(self.config.d, self.config.r).astype(np.float32) * 0.02
        return self._Wv
    
    def _get_Wr(self) -> np.ndarray:
        """Get read projection matrix"""
        if not hasattr(self, '_Wr'):
            self._Wr = np.random.randn(self.config.r, self.config.d).astype(np.float32) * 0.02
        return self._Wr
    
    def reset(self) -> None:
        """Reset the layer state"""
        self._context_buffer = []
        self.state = np.zeros((self.config.M, self.config.r))
        self.store.clear()
        self.total_tokens = 0
        self.total_hits = 0
        self.total_misses = 0
        self.start_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics"""
        return {
            'total_tokens': self.total_tokens,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'recall_rate': self.total_hits / (self.total_hits + self.total_misses + 1e-12),
            'store_size': len(self.store),
            'uptime': time.time() - self.start_time,
        }


# =============================================================================
# NUMPY VERSION (for compatibility)
# =============================================================================

class SuperfluidMambaFluidLayerNP:
    """
    NumPy version of Superfluid-Mamba FluidLayer for compatibility.
    
    Note: Mamba and Superfluid tiers are placeholders (None) in this
    version; only the exact and lossy tiers are active.
    """
    
    def __init__(self, config: SuperfluidMambaConfig):
        self.config = config
        config.validate()
        
        # Initialize components
        self.session = CryptographicSession(
            security_level=SecurityLevel[config.security_level.upper()]
        )
        self.store = PerfectHashStore(
            initial_capacity=config.capacity,
            max_capacity=config.max_capacity,
            hash_mode=HashMode.DOUBLE_64,
            session=self.session,
        )
        self.sheaf = AdaptiveGatedSheaf(
            M=config.M,
            r=config.r,
            d=config.d,
            k=config.superfluid_k,
        )
        
        # Mamba (simplified numpy version)
        self.mamba = self._init_mamba_np(config)
        
        # Superfluid
        self.superfluid = self._init_superfluid_np(config)
        
        # Output projections (ZERO initialized)
        self.W_e = np.zeros((config.d, config.d))
        self.W_l = np.zeros((config.d, config.d))
        self.W_m = np.zeros((config.d, config.d))
        self.W_sf = np.zeros((config.d, config.d))
        
        # Context buffer
        self._context_buffer: List[int] = []
        
        # Statistics
        self.total_tokens = 0
        self.total_hits = 0
        self.total_misses = 0
        self.start_time = time.time()
        
        logger.info("SuperfluidMambaFluidLayerNP initialized successfully!")
    
    def _init_mamba_np(self, config: SuperfluidMambaConfig) -> Any:
        """Initialize simplified Mamba for NumPy"""
        # Placeholder: Mamba tier is not implemented in the NP twin
        return None
    
    def _init_superfluid_np(self, config: SuperfluidMambaConfig) -> Any:
        """Initialize simplified Superfluid for NumPy"""
        # Placeholder: Superfluid tier is not implemented in the NP twin
        return None
    
    def __call__(
        self,
        h: np.ndarray,
        content: np.ndarray,
        token_ids: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass (NumPy version)"""
        return self.forward(h, content, token_ids)
    
    def forward(
        self,
        h: np.ndarray,
        content: np.ndarray,
        token_ids: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Forward pass (NumPy version)"""
        # For now, use only FluidLayer components
        # Full implementation would include Mamba and Superfluid
        T = token_ids.shape[0]
        out_e, info_e = self._process_exact_tier(h, content, token_ids)
        out_l, info_l = self._process_lossy_tier(h, content, token_ids)
        
        # Simple combination
        h_prime = h + out_e @ self.W_e.T + out_l @ self.W_l.T
        
        info = {
            'exact_tier': info_e,
            'lossy_tier': info_l,
            'recall_rate': info_e.get('recall_rate', 0.0),
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
                prev_key = self._generate_context_key(prev_tokens[-self.config.ctx_len:])
            else:
                prev_key = None
            
            self._context_buffer.append(int(token_ids[t]))
            
            curr_tokens = self._context_buffer[-(self.config.ctx_len):]
            if len(curr_tokens) >= self.config.ctx_len:
                curr_key = self._generate_context_key(curr_tokens)
            else:
                curr_key = None
            
            # FIX: the old code used get(curr_key, -1) and compared the
            # stored VALUE (an ndarray) against -1, which raises a truth-
            # value ambiguity error the moment a hit occurs.  Use None as
            # the default, like the torch twin.
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
            state_read = w @ self.state
            out_l = state_read @ self._get_Wr()
        
        return out_l if out_l is not None else np.zeros((T, self.config.d)), {}
    
    def _generate_context_key(self, tokens):
        return tuple(tokens)
    
    def _route(self, content):
        T, d = content.shape
        scores = content @ self._get_Wq().T
        w = np.exp(scores) / (np.sum(np.exp(scores), axis=1, keepdims=True) + 1e-12)
        return w, np.max(scores, axis=1)
    
    def _get_Wq(self):
        if not hasattr(self, '_Wq'):
            self._Wq = np.random.randn(self.config.M, self.config.d).astype(np.float32) * 0.02  # FIX: orientation was (d, M), crashes when M != d
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
# UTILITY FUNCTIONS
# =============================================================================

def create_superfluid_mamba_layer(
    d: int = 64,
    M: int = 64,
    r: int = 8,
    ctx_len: int = 12,
    use_pytorch: bool = True,
) -> Any:
    """
    Factory function to create Superfluid-Mamba FluidLayer.
    
    Args:
        d: Hidden dimension
        M: Number of anchors
        r: Rank for low-rank projections
        ctx_len: Context length
        use_pytorch: Whether to use PyTorch version
    
    Returns:
        Superfluid-Mamba FluidLayer instance
    """
    config = SuperfluidMambaConfig(
        M=M,
        r=r,
        d=d,
        ctx_len=ctx_len,
        mamba_d_state=16,
        mamba_d_conv=4,
        superfluid_viscosity=0.1,
        superfluid_decay=0.01,
        superfluid_k=8,
    )
    
    if use_pytorch:
        return SuperfluidMambaFluidLayer(config)
    else:
        return SuperfluidMambaFluidLayerNP(config)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SUPERFLUID-MAMBA FLUIDLAYER - INITIALIZATION TEST")
    print("=" * 80)
    
    # Test configuration
    config = SuperfluidMambaConfig(
        M=64,
        r=8,
        d=64,
        ctx_len=12,
        mamba_d_state=16,
        mamba_d_conv=4,
    )
    config.validate()
    print("OK - Configuration validated")
    
    # Test PyTorch version
    print("\nTesting PyTorch version...")
    try:
        layer = SuperfluidMambaFluidLayer(config)
        print("OK - PyTorch layer initialized")
        
        # Test forward pass
        h = torch.randn(10, 64)
        content = torch.randn(10, 64)
        token_ids = torch.randint(0, 1000, (10,))
        
        output, info = layer(h, content, token_ids)
        print(f"OK - Forward pass successful (input: {h.shape}, output: {output.shape})")
        print(f"   Recall rate (fresh sequence, expected low): {info['recall_rate']:.2%}")
        
    except Exception as e:
        print(f"FAIL - PyTorch test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test NumPy version
    print("\nTesting NumPy version...")
    try:
        layer_np = SuperfluidMambaFluidLayerNP(config)
        print("OK - NumPy layer initialized")
        
        # Test forward pass
        h_np = np.random.randn(10, 64).astype(np.float32)
        content_np = np.random.randn(10, 64).astype(np.float32)
        token_ids_np = np.random.randint(0, 1000, (10,))
        
        output_np, info_np = layer_np(h_np, content_np, token_ids_np)
        print(f"OK - Forward pass successful (input: {h_np.shape}, output: {output_np.shape})")
        print(f"   Recall rate (fresh sequence, expected low): {info_np['recall_rate']:.2%}")
        
    except Exception as e:
        print(f"FAIL - NumPy test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE (see test_superfluid_mamba.py for assertions)")
    print("=" * 80)
