import argparse
import random
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Optional

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_tensor_cast_back


def test_per_tensor_static_quant(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                                  rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
                                  use_logfmt: bool = False, seed: int = 0):
    """Test per-tensor static quantization for low-latency dispatch and combine."""
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    # Create test data with different patterns to verify static quantization
    x_list = []

    # Test 1: Uniform values (easy to quantize)
    x_uniform = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x_uniform[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    x_list.append(x_uniform)

    # Test 2: Random values with small range (good for static quant)
    x_small_range = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1
    x_list.append(x_small_range)

    # Test 3: Random values with larger range
    x_large_range = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 2.0
    x_list.append(x_large_range)

    # Test 4: Mixed scales (some channels large, some small)
    x_mixed = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    # Make first half of channels have larger values
    x_mixed[:, :hidden//2] *= 4.0
    x_list.append(x_mixed)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    for i in range(10):
        topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    def compute_static_scale(x: torch.Tensor, quantize_type: str = "per_tensor"):
        """Compute static scale for quantization."""
        if quantize_type == "per_tensor":
            # Single scale for entire tensor
            amax = x.abs().max().float()
            scale = amax / 448.0  # FP8 E4M3 max value
            return scale.view(1)
        elif quantize_type == "per_channel":
            # Scale per 128 channels
            assert hidden % 128 == 0
            x_view = x.view(-1, hidden // 128, 128)
            amax = x_view.abs().float().amax(dim=-1).amax(dim=0)
            scale = amax / 448.0
            return scale
        else:
            raise ValueError(f"Unknown quantize_type: {quantize_type}")

    # Test different static quantization configurations
    hash_value, num_times = 0, 0
    for current_x in x_list:
        for quantize_type in ["per_tensor"]:
            # Compute static scale
            static_scale = compute_static_scale(current_x, quantize_type)

            for return_recv_hook in (False, True):
                num_times += 1
                for i in range((num_times % 2) + 1):
                    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')
                    packed_recv_x, packed_recv_count, handle, event, hook = \
                        buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                                    static_scale=static_scale,
                                                    use_fp8=True, round_scale=False, use_ue8m0=False,
                                                    cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                                    async_finish=not return_recv_hook, return_recv_hook=return_recv_hook)
                    hook() if return_recv_hook else event.current_stream_wait()

                # Cast back to verify correctness
                all_topk_idx = torch.empty((num_ranks, num_tokens, num_topk), dtype=topk_idx.dtype, device='cuda')
                dist.all_gather_into_tensor(all_topk_idx, topk_idx, group=group)

                for i in range(num_local_experts):
                    expert_id = rank * num_local_experts + i
                    recv_count, recv_src_info, recv_layout_range = packed_recv_count[i], handle[0][i], handle[1][i]
                    num_valid_tokens = recv_count.item()

                    if num_valid_tokens == 0:
                        continue

                    # Cast back FP8 to BF16 using static scale
                    recv_x_fp8 = packed_recv_x[0][i, :num_valid_tokens]
                    recv_x = per_tensor_cast_back(recv_x_fp8, static_scale)

                    # Verify the received data matches expected pattern
                    if current_x is x_uniform:
                        # For uniform data, check that values are approximately correct
                        expected_value = rank - rank_offset
                        actual_mean = recv_x[:, :-128].mean().item()
                        assert abs(actual_mean - expected_value) < 0.1, f"Mean mismatch: {actual_mean} vs {expected_value}"

                        # Check source indices
                        recv_src_info = recv_src_info[:num_valid_tokens]
                        assert (recv_x[:, -128:] - recv_src_info.view(-1, 1) % num_tokens).sum().item() == 0

                    # Verify expert counts
                    assert cumulative_local_expert_recv_stats[i].item() == num_valid_tokens
                    assert num_valid_tokens == (all_topk_idx == expert_id).sum().item()

                    # Update hash for later verification
                    hash_value ^= hash_tensor(packed_recv_x[0][i, :num_valid_tokens])
                    hash_value ^= hash_tensor(packed_recv_x[1][i, :num_valid_tokens])

                # Test combine with static quantization
                simulated_gemm_x = (packed_recv_x[0].clone(), packed_recv_x[1].clone())

                for zero_copy in (False, True):
                    if zero_copy:
                        buffer.get_next_low_latency_combine_buffer(handle)[:, :, :] = simulated_gemm_x[0]

                    out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
                    combined_x, event, hook = buffer.low_latency_combine(
                        simulated_gemm_x[0], topk_idx, topk_weights, handle,
                        use_logfmt=use_logfmt,
                        async_finish=not return_recv_hook, zero_copy=zero_copy,
                        return_recv_hook=return_recv_hook, out=out)
                    hook() if return_recv_hook else event.current_stream_wait()

                    # Verify combine correctness
                    expected = current_x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1)
                    diff = calc_diff(expected, combined_x)
                    assert torch.isnan(combined_x).sum().item() == 0
                    assert diff < 9e-4, f'Combine error too high: {diff=}, quantize_type={quantize_type}'

                    hash_value ^= hash_tensor(combined_x)

    return hash_value


def test_static_quant_edge_cases(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
                                  rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer):
    """Test edge cases for static quantization."""
    torch.manual_seed(42 + rank)

    # Test with very small values (near quantization limit)
    x_small = torch.full((num_tokens, hidden), 1e-4, dtype=torch.bfloat16, device='cuda')
    static_scale = torch.tensor([1e-4 / 448.0], dtype=torch.float32, device='cuda')

    scores = torch.ones((num_tokens, num_experts), dtype=torch.float32, device='cuda')
    topk_idx = torch.zeros((num_tokens, num_topk), dtype=torch.int64, device='cuda')
    topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device='cuda')

    # Dispatch
    recv_x, recv_count, handle, _, _ = buffer.low_latency_dispatch(
        x_small, topk_idx, num_tokens, num_experts,
        static_scale=static_scale,
        use_fp8=True, round_scale=False, use_ue8m0=False)

    # Combine
    combined_x, _, _ = buffer.low_latency_combine(
        recv_x, topk_idx, topk_weights, handle,
        use_logfmt=False, zero_copy=False)

    # Verify small values are handled correctly
    assert not torch.isnan(combined_x).any()
    assert (combined_x.abs() < 1e-3).all()


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink, explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl)

    # Run main test
    hash_value = test_per_tensor_static_quant(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                                              use_logfmt=args.use_logfmt, seed=1)

    # Run edge case tests
    test_static_quant_edge_cases(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer)

    # Pressure test with multiple seeds
    do_pressure_test = args.pressure_test
    for seed in range(int(1e6) if do_pressure_test else 0):
        if local_rank == 0 and seed % 100 == 0:
            print(f'Testing with seed {seed} ...', flush=True)
        ref_hash = test_per_tensor_static_quant(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
                                                use_logfmt=args.use_logfmt, seed=seed)
        assert ref_hash == hash_value, f'Hash mismatch with seed={seed}'

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test per-tensor static quantization for low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128,
                       help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288,
                       help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true",
                        help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true',
                        help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true',
                        help='Whether to test LogFMT combine')
    parser.add_argument("--pressure-test", action='store_true',
                        help='Whether to do pressure test')
    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)