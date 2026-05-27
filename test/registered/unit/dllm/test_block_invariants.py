"""Unit tests for dLLM block scheduling."""

import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.flashinfer_backend import (  # noqa: E402
    FlashInferAttnBackend,
)
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDllmBlockInvariants(CustomTestCase):
    def test_flashinfer_dllm_prefix_lens_use_seq_len_minus_block(self):
        """Check FlashInfer prefix lengths follow the dLLM invariant."""
        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.dllm_config = SimpleNamespace(block_size=32)

        seq_lens = torch.tensor([32, 96], dtype=torch.int32)
        expected = torch.tensor([0, 64], dtype=torch.int32)

        self.assertTrue(
            torch.equal(
                backend._get_dllm_prefix_lens(seq_lens, num_tokens=64), expected
            )
        )

    def test_flashinfer_dllm_metadata_rejects_bad_token_count(self):
        """Check FlashInfer rejects non-block query shapes."""
        backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        backend.dllm_config = SimpleNamespace(block_size=4)

        with self.assertRaisesRegex(AssertionError, "one dLLM block"):
            backend._get_dllm_prefix_lens(
                torch.tensor([4, 8], dtype=torch.int32),
                num_tokens=7,
            )

    def test_prefill_adder_rejects_partial_dllm_budget(self):
        """Check scheduler admission refuses partial dLLM blocks."""
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.dllm_config = SimpleNamespace()
        adder.dllm_block_size = 4
        adder.rem_dllm_tokens = 3
        adder.rem_input_tokens = 128
        adder.can_run_list = []

        req = SimpleNamespace(
            extend_input_len=9, fill_ids=list(range(9)), prefix_indices=[]
        )

        with (
            patch.object(
                PrefillAdder, "rem_total_tokens", new_callable=PropertyMock
            ) as total,
            patch.object(
                PrefillAdder, "cur_rem_tokens", new_callable=PropertyMock
            ) as current,
        ):
            total.return_value = 128
            current.return_value = 128

            self.assertEqual(adder.add_dllm_staging_req(req), AddReqResult.NO_TOKEN)

        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(req.extend_input_len, 9)

    def test_prefill_adder_adds_only_full_dllm_blocks(self):
        """Check scheduler admission truncates to exactly one dLLM block."""
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.dllm_block_size = 4
        adder.rem_dllm_tokens = 4
        adder.rem_input_tokens = 128
        adder.can_run_list = []

        req = SimpleNamespace(extend_input_len=9, fill_ids=list(range(9)))

        with (
            patch.object(
                PrefillAdder, "rem_total_tokens", new_callable=PropertyMock
            ) as total,
            patch.object(
                PrefillAdder, "cur_rem_tokens", new_callable=PropertyMock
            ) as current,
            patch.object(adder, "_update_prefill_budget") as update_budget,
        ):
            total.return_value = 128
            current.return_value = 128

            adder._add_dllm_req(req, prefix_len=2)

        self.assertEqual(req.extend_input_len, 4)
        self.assertEqual(req.fill_ids, [0, 1, 2, 3, 4, 5])
        self.assertEqual(adder.can_run_list, [req])
        update_budget.assert_called_once_with(2, 4, 0)


if __name__ == "__main__":
    unittest.main()
