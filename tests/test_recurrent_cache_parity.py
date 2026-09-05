"""Real tiny Qwen decoders: physical gather equals masking the same old KV."""
import unittest
from types import SimpleNamespace

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from vlm_diagnosis.core.session_cache import AttentionMass, ColdKV


class PhysicalCacheParityTest(unittest.TestCase):
    @torch.no_grad()
    def test_sparse_prefill_and_decode_match_masked_full_for_both_qwen_families(self):
        for config_type, model_type in (
            (Qwen2_5_VLTextConfig, Qwen2_5_VLTextModel),
            (Qwen3VLTextConfig, Qwen3VLTextModel),
        ):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(12)
                config = config_type(
                    vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, max_position_embeddings=128,
                    rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
                    pad_token_id=0, bos_token_id=1, eos_token_id=2,
                )
                config._attn_implementation = "eager"
                model = model_type(config).eval()
                prefix_ids = torch.tensor([[4, 5, 6, 7, 8, 9]])
                positions = torch.arange(6)[None, None].expand(3, 1, 6)
                initial = model(input_ids=prefix_ids, position_ids=positions, use_cache=True)
                cold = ColdKV(initial.past_key_values.to_legacy_cache())
                selected = torch.tensor([0, 2, 5])
                compact = cold.gather(selected, "cpu")
                dense = cold.gather(torch.arange(6), "cpu")

                # A logical position far beyond physical length makes an
                # accidental get_seq_length-based RoPE assignment observable.
                for ids, logical_start in ((torch.tensor([[10, 11]]), 20),
                                           (torch.tensor([[12]]), 22)):
                    n = ids.shape[1]
                    cp, dp = compact.get_seq_length(), dense.get_seq_length()
                    pos = torch.arange(logical_start, logical_start+n)[None, None].expand(3, 1, n)
                    mask = torch.zeros(1, 1, n, dp+n)
                    mask[..., [1, 3, 4]] = float("-inf")
                    for row in range(n):
                        mask[:, :, row, dp+row+1:] = float("-inf")
                    with AttentionMass(SimpleNamespace(model=model)) as ccap:
                        compressed = model(
                            input_ids=ids, position_ids=pos,
                            cache_position=torch.arange(cp, cp+n),
                            attention_mask=torch.ones(1, cp+n, dtype=torch.long),
                            past_key_values=compact, use_cache=True)
                    with AttentionMass(SimpleNamespace(model=model)) as dcap:
                        reference = model(
                            input_ids=ids, position_ids=pos,
                            cache_position=torch.arange(dp, dp+n), attention_mask=mask,
                            past_key_values=dense, use_cache=True)
                    torch.testing.assert_close(compressed.last_hidden_state,
                                               reference.last_hidden_state, atol=1e-5, rtol=1e-5)
                    mapping = torch.cat((selected, torch.arange(6, dp+n)))
                    torch.testing.assert_close(ccap.mean(), dcap.mean()[mapping], atol=1e-6, rtol=1e-5)
                    self.assertTrue(bool((dcap.mean()[[1, 3, 4]] == 0).all()))
                    compact, dense = compressed.past_key_values, reference.past_key_values


if __name__ == "__main__":
    unittest.main()
