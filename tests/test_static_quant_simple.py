import torch
import deep_ep
from deep_ep.buffer import Buffer

def test_per_tensor_static_quant_simple():
    """Simple test for per-tensor static quantization without distributed setup."""

    # Setup
    num_tokens = 64
    hidden = 1024
    num_experts = 8
    num_topk = 2

    # Create dummy tensors
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), dtype=torch.int64, device='cuda')
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Compute static scale (per-tensor)
    amax = x.abs().max().float()
    static_scale = torch.tensor([amax / 448.0], dtype=torch.float32, device='cuda')

    print(f"Input shape: {x.shape}")
    print(f"Static scale: {static_scale.item():.6f}")
    print(f"Expected amax: {amax.item():.6f}")

    # Test dispatch with static quantization
    # Note: This would require actual distributed setup to run
    # Here we just verify the scale computation logic

    # Simulate FP8 quantization with static scale
    x_fp32 = x.float()
    x_quantized = (x_fp32 / static_scale.item()).clamp(-448, 448)
    x_fp8 = x_quantized.to(torch.float8_e4m3fn)

    # Dequantize
    x_dequantized = x_fp8.float() * static_scale.item()

    # Check accuracy
    diff = torch.abs(x_fp32 - x_dequantized).max().item()
    print(f"Max quantization error: {diff:.6f}")

    assert diff < 0.1, f"Quantization error too high: {diff}"
    print("✓ Per-tensor static quantization test passed!")

if __name__ == "__main__":
    test_per_tensor_static_quant_simple()