"""
Reporting and Analysis Module.
Processes raw benchmark records, calculates aggregates and percentage deltas,
computes roofline model bounds, and renders markdown comparison reports.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BenchmarkReporter:
    @staticmethod
    def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
        """Computes mean, median, p95, and std across repetitions for each engine and quantization."""
        if df.empty:
            return pd.DataFrame()

        # Group by engine and quantization
        group_cols = ["engine", "model_id", "quantization", "device"]
        metrics = [
            "ttft_ms",
            "median_itl_ms",
            "p95_itl_ms",
            "time_per_output_token_ms",
            "end_to_end_ms",
            "output_tokens_per_second",
            "process_peak_ram_mb",
        ]

        agg_dict = {}
        for m in metrics:
            if m in df.columns:
                agg_dict[f"{m}_median"] = (m, "median")
                agg_dict[f"{m}_p95"] = (m, lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) > 0 else np.nan)
                agg_dict[f"{m}_mean"] = (m, "mean")

        summary = df.groupby(group_cols).agg(**agg_dict).reset_index()
        return summary

    @staticmethod
    def calculate_delta_table(
        baseline_row: pd.Series,
        optimized_row: pd.Series,
    ) -> List[Dict[str, Any]]:
        """
        Calculates percentage changes between baseline and optimized runs.
        Explicitly notes whether higher or lower is better.
        """
        comparisons = [
            ("TTFT (ms)", "ttft_ms_median", "lower"),
            ("Inter-Token Latency (ms)", "median_itl_ms_median", "lower"),
            ("End-to-End Latency (ms)", "end_to_end_ms_median", "lower"),
            ("Throughput (tokens/s)", "output_tokens_per_second_median", "higher"),
            ("Peak Process RAM (MB)", "process_peak_ram_mb_median", "lower"),
        ]

        results = []
        for label, col, desired in comparisons:
            base_val = baseline_row.get(col, np.nan)
            opt_val = optimized_row.get(col, np.nan)

            if pd.notna(base_val) and pd.notna(opt_val) and base_val != 0:
                pct_change = ((opt_val - base_val) / abs(base_val)) * 100.0
                speedup_factor = (
                    base_val / opt_val if opt_val > 0 and desired == "lower"
                    else opt_val / base_val if desired == "higher" and base_val > 0
                    else 1.0
                )
                improvement = (pct_change < 0) if desired == "lower" else (pct_change > 0)
            else:
                pct_change = None
                speedup_factor = None
                improvement = None

            results.append({
                "Metric": label,
                "Baseline": round(base_val, 2) if pd.notna(base_val) else "N/A",
                "Optimized": round(opt_val, 2) if pd.notna(opt_val) else "N/A",
                "Delta %": f"{pct_change:+.1f}%" if pct_change is not None else "N/A",
                "Effective Speedup": f"{speedup_factor:.2f}x" if speedup_factor else "N/A",
                "Desired": "Lower is better" if desired == "lower" else "Higher is better",
                "Status": "Improved" if improvement else "Degraded" if improvement is False else "Neutral",
            })

        return results

    @staticmethod
    def compute_roofline_analysis(
        param_count_billions: float = 0.49,
        quant_bits: float = 4.5,
        memory_bandwidth_gb_s: float = 16.0,
        cpu_peak_gflops: float = 64.0,
    ) -> Dict[str, Any]:
        """
        Educational Roofline Model Analysis for Autoregressive LLM Decoding.
        """
        bytes_per_param = quant_bits / 8.0
        active_model_size_gb = param_count_billions * bytes_per_param

        # Batch=1 token generation arithmetic intensity: 2 FLOPs per param / bytes per param
        arithmetic_intensity = 2.0 / bytes_per_param  # FLOPs/byte

        # Memory bandwidth roofline ceiling: max tokens per second
        # Throughput <= Memory_Bandwidth / Active_Model_Size
        max_throughput_tokens_per_s = memory_bandwidth_gb_s / active_model_size_gb

        # Compute ceiling (at peak GFLOPs)
        # FLOPs per token = 2 * param_count * 10^9
        flops_per_token = 2.0 * param_count_billions * 1e9
        compute_ceiling_tokens_per_s = (cpu_peak_gflops * 1e9) / flops_per_token

        # Operational regime
        knee_point_intensity = cpu_peak_gflops / memory_bandwidth_gb_s
        is_memory_bound = arithmetic_intensity < knee_point_intensity

        return {
            "model_parameters_b": param_count_billions,
            "quantization_bits": quant_bits,
            "active_model_size_gb": round(active_model_size_gb, 3),
            "arithmetic_intensity_flops_per_byte": round(arithmetic_intensity, 2),
            "hardware_memory_bandwidth_gb_s": memory_bandwidth_gb_s,
            "hardware_peak_compute_gflops": cpu_peak_gflops,
            "roofline_inflection_knee": round(knee_point_intensity, 2),
            "operational_regime": "Memory-Bandwidth Bound" if is_memory_bound else "Compute Bound",
            "theoretical_max_tokens_per_s": round(max_throughput_tokens_per_s, 2),
            "compute_ceiling_tokens_per_s": round(compute_ceiling_tokens_per_s, 2),
        }
