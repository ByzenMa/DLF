"""
DLF backbone with Separated Layer Extraction (SLE).

The SLE block follows the PLE idea of routing each input through modality-specific
feature experts and shared experts across configurable stacked extraction
layers, then fusing them with per-modality gates before the regular DLF
disentanglement, reconstruction, and fusion pipeline.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...subNets import BertTextEncoder
from ...subNets.transformers_encoder.transformer import TransformerEncoder


class GatedExpertFusion(nn.Module):
    """Fuse feature and shared experts with PLE-style feature/shared gates."""

    def __init__(self, feature_experts, embed_dim, num_shared_experts):
        super().__init__()
        self.feature_experts = nn.ModuleList(feature_experts)
        num_total_experts = len(self.feature_experts) + num_shared_experts
        self.feature_gate = nn.Linear(embed_dim, num_total_experts)
        self.shared_gate = nn.Linear(embed_dim, num_total_experts)

    @staticmethod
    def _gated_sum(expert_outputs, gate, gate_input):
        gate_weights = F.softmax(gate(gate_input.mean(dim=0)), dim=-1).transpose(0, 1)
        return (expert_outputs * gate_weights[:, None, :, None]).sum(dim=0)

    def forward(self, feature_x, shared_x, shared_experts):
        expert_outputs = [expert(feature_x) for expert in self.feature_experts]
        expert_outputs.extend(expert(shared_x) for expert in shared_experts)
        expert_outputs = torch.stack(expert_outputs, dim=0)

        feature_out = self._gated_sum(expert_outputs, self.feature_gate, feature_x)
        shared_out = self._gated_sum(expert_outputs, self.shared_gate, shared_x)
        return feature_out, shared_out


class SLEExtractionLayer(nn.Module):
    """One PLE-style extraction layer for text, audio, and video streams."""

    def __init__(self, build_experts, embed_dim, num_feature_experts, num_shared_experts):
        super().__init__()
        self.shared_experts = nn.ModuleList(build_experts("l", num_shared_experts))
        self.text_fusion = self._build_fusion(build_experts, "l", embed_dim, num_feature_experts, num_shared_experts)
        self.audio_fusion = self._build_fusion(build_experts, "a", embed_dim, num_feature_experts, num_shared_experts)
        self.video_fusion = self._build_fusion(build_experts, "v", embed_dim, num_feature_experts, num_shared_experts)

    @staticmethod
    def _build_fusion(build_experts, self_type, embed_dim, num_feature_experts, num_shared_experts):
        feature_experts = build_experts(self_type, num_feature_experts)
        return GatedExpertFusion(feature_experts, embed_dim, num_shared_experts)

    def forward(self, x_l, x_a, x_v, shared_l, shared_a, shared_v):
        x_l, shared_l = self.text_fusion(x_l, shared_l, self.shared_experts)
        x_a, shared_a = self.audio_fusion(x_a, shared_a, self.shared_experts)
        x_v, shared_v = self.video_fusion(x_v, shared_v, self.shared_experts)
        return x_l, x_a, x_v, shared_l, shared_a, shared_v


class MINE(nn.Module):
    """Mutual Information Neural Estimator used as an optional auxiliary loss."""

    def __init__(self, data_dim, hidden_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, y):
        y_marginal = torch.roll(y, shifts=1, dims=0)
        joint_logits = self.layers(torch.cat([x, y], dim=1))
        marginal_logits = self.layers(torch.cat([x, y_marginal], dim=1))
        log_mean_exp = torch.logsumexp(marginal_logits, dim=0) - math.log(marginal_logits.size(0))
        return -math.log2(math.e) * (torch.mean(joint_logits) - log_mean_exp.squeeze(0))


class DLF_sle(nn.Module):
    """DLF with PLE-style separated layer extraction for text/audio/video inputs."""

    def __init__(self, args):
        super().__init__()
        self.use_bert = args.use_bert
        if self.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=args.use_finetune,
                transformers=args.transformers,
                pretrained=args.pretrained,
            )

        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.len_l, self.len_v, self.len_a = self._get_sequence_lengths(args)
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        output_dim = 1
        combined_dim_low = self.d_l
        combined_dim_high = self.d_l
        combined_dim = self.d_l + self.d_a + self.d_v + self.d_l * 3

        # Initial temporal projections.
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # Separated Layer Extraction (SLE) experts.
        self.use_sle = bool(getattr(args, "use_sle", True))
        self.sle_num_layers = self._get_positive_int_arg(args, "sle_num_layers", default=2)
        self.sle_num_feature_experts = self._get_positive_int_arg(
            args, "sle_num_feature_experts", "sle_num_specific_experts", default=2
        )
        self.sle_num_shared_experts = self._get_positive_int_arg(args, "sle_num_shared_experts", default=2)
        if self.use_sle:
            self.sle_layers = nn.ModuleList(
                SLEExtractionLayer(
                    self._build_sle_experts,
                    self.d_l,
                    self.sle_num_feature_experts,
                    self.sle_num_shared_experts,
                )
                for _ in range(self.sle_num_layers)
            )
        else:
            self.sle_layers = nn.ModuleList()

        # Optional MINE auxiliary loss over the actual DLF backbone inputs.
        self.use_mine_loss = bool(getattr(args, "use_mine_loss", False))
        self.mine_loss_weight = float(getattr(args, "mine_loss_weight", 0.1))
        mine_hidden_size = int(getattr(args, "mine_hidden_size", 10))
        if self.use_mine_loss:
            mine_data_dim = self.d_l * 2
            self.mine_l_a = MINE(mine_data_dim, mine_hidden_size)
            self.mine_l_v = MINE(mine_data_dim, mine_hidden_size)
            self.mine_a_v = MINE(mine_data_dim, mine_hidden_size)

        # Disentanglement encoders.
        self.encoder_s_l = self.get_network(self_type="l", layers=self.layers)
        self.encoder_s_v = self.get_network(self_type="v", layers=self.layers)
        self.encoder_s_a = self.get_network(self_type="a", layers=self.layers)
        self.encoder_c = self.get_network(self_type="l", layers=self.layers)

        # Reconstruction decoders used by reconstruction and specific losses.
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # Shared-feature alignment used by the triplet/hinge similarity loss.
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1), combined_dim_low)
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1), combined_dim_low)
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1), combined_dim_low)

        # Shared-feature prediction head.
        self.self_attentions_c_l = self.get_network(self_type="l")
        self.self_attentions_c_v = self.get_network(self_type="v")
        self.self_attentions_c_a = self.get_network(self_type="a")
        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)

        # Language-focused cross-modal attention path used by final fusion.
        self.trans_l_with_a = self.get_network(self_type="la", layers=self.layers)
        self.trans_l_with_v = self.get_network(self_type="lv", layers=self.layers)
        self.trans_l_mem = self.get_network(self_type="l_mem", layers=self.layers)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

        # Specific-feature prediction heads.
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        # Fusion and final prediction head.
        self.projector_l = nn.Linear(self.d_l, self.d_l)
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    @staticmethod
    def _get_sequence_lengths(args):
        if args.dataset_name == "mosi":
            return (50, 50, 50) if args.need_data_aligned else (50, 500, 375)
        if args.dataset_name == "mosei":
            return (50, 50, 50) if args.need_data_aligned else (50, 500, 500)
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")

    @staticmethod
    def _get_positive_int_arg(args, primary_name, fallback_name=None, default=2):
        value = getattr(args, primary_name, None)
        if value is None and fallback_name is not None:
            value = getattr(args, fallback_name, None)
        value = default if value is None else int(value)
        if value < 1:
            raise ValueError(f"{primary_name} must be >= 1, got {value}")
        return value

    def _build_sle_experts(self, self_type, num_experts):
        return [
            self.get_network(self_type=self_type, layers=self.layers)
            for _ in range(num_experts)
        ]

    def _apply_sle_layers(self, x_l, x_a, x_v):
        shared_l, shared_a, shared_v = x_l, x_a, x_v
        for sle_layer in self.sle_layers:
            x_l, x_a, x_v, shared_l, shared_a, shared_v = sle_layer(
                x_l, x_a, x_v, shared_l, shared_a, shared_v
            )
        return x_l, x_a, x_v

    def _compute_mine_loss(self, x_l, x_a, x_v):
        h_l = x_l.mean(dim=0)
        h_a = x_a.mean(dim=0)
        h_v = x_v.mean(dim=0)
        return (
            self.mine_l_a(h_l, h_a)
            + self.mine_l_v(h_l, h_v)
            + self.mine_a_v(h_a, h_v)
        ) / 3

    def get_network(self, self_type="l", layers=-1):
        if self_type in ["l", "al", "vl", "l_mem"]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ["a", "la", "va"]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ["v", "lv", "av"]:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == "a_mem":
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == "v_mem":
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError(f"Unknown network type: {self_type}")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    @staticmethod
    def _unwrap_transformer_output(output):
        return output[0] if isinstance(output, tuple) else output

    def _residual_head(self, x, proj1, proj2):
        residual = x
        x = proj1(x)
        x = F.dropout(F.relu(x, inplace=True), p=self.output_dropout, training=self.training)
        x = proj2(x)
        return x + residual

    def forward(self, text, audio, video):
        if self.use_bert:
            text = self.text_model(text)

        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        if self.use_sle:
            model_x_l, model_x_a, model_x_v = self._apply_sle_layers(proj_x_l, proj_x_a, proj_x_v)
        else:
            model_x_l, model_x_a, model_x_v = proj_x_l, proj_x_a, proj_x_v

        s_l = self.encoder_s_l(model_x_l)
        s_v = self.encoder_s_v(model_x_v)
        s_a = self.encoder_s_a(model_x_a)
        c_l = self.encoder_c(model_x_l)
        c_v = self.encoder_c(model_x_v)
        c_a = self.encoder_c(model_x_a)

        s_l_bct, s_v_bct, s_a_bct = s_l.permute(1, 2, 0), s_v.permute(1, 2, 0), s_a.permute(1, 2, 0)
        c_l_bct, c_v_bct, c_a_bct = c_l.permute(1, 2, 0), c_v.permute(1, 2, 0), c_a.permute(1, 2, 0)

        batch_size = x_l.size(0)
        c_l_sim = self.align_c_l(c_l_bct.contiguous().view(batch_size, -1))
        c_v_sim = self.align_c_v(c_v_bct.contiguous().view(batch_size, -1))
        c_a_sim = self.align_c_a(c_a_bct.contiguous().view(batch_size, -1))

        recon_l = self.decoder_l(torch.cat([s_l_bct, c_l_bct], dim=1)).permute(2, 0, 1)
        recon_v = self.decoder_v(torch.cat([s_v_bct, c_v_bct], dim=1)).permute(2, 0, 1)
        recon_a = self.decoder_a(torch.cat([s_a_bct, c_a_bct], dim=1)).permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l).permute(1, 2, 0)
        s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)

        # Shared branch.
        c_l_att = self._unwrap_transformer_output(self.self_attentions_c_l(c_l))[-1]
        c_v_att = self._unwrap_transformer_output(self.self_attentions_c_v(c_v))[-1]
        c_a_att = self._unwrap_transformer_output(self.self_attentions_c_a(c_a))[-1]
        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)
        c_proj = self._residual_head(c_fusion, self.proj1_c, self.proj2_c)
        logits_c = self.out_layer_c(c_proj)

        # Specific/cross-modal branch.
        h_ls = self._unwrap_transformer_output(self.trans_l_mem(s_l))
        last_h_l = h_ls[-1]

        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
        h_as = self._unwrap_transformer_output(self.trans_a_mem(h_l_with_as))
        last_h_a = h_as[-1]

        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
        h_vs = self._unwrap_transformer_output(self.trans_v_mem(h_l_with_vs))
        last_h_v = h_vs[-1]

        hs_proj_l_high = self._residual_head(last_h_l, self.proj1_l_high, self.proj2_l_high)
        hs_proj_v_high = self._residual_head(last_h_v, self.proj1_v_high, self.proj2_v_high)
        hs_proj_a_high = self._residual_head(last_h_a, self.proj1_a_high, self.proj2_a_high)
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        # Final fusion and prediction.
        last_h_l = torch.sigmoid(self.projector_l(hs_proj_l_high))
        last_h_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        last_h_a = torch.sigmoid(self.projector_a(hs_proj_a_high))
        c_fusion = torch.sigmoid(self.projector_c(c_fusion))
        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)
        output = self.out_layer(self._residual_head(last_hs, self.proj1, self.proj2))

        res = {
            "origin_l": model_x_l,
            "origin_v": model_x_v,
            "origin_a": model_x_a,
            "projected_l": proj_x_l,
            "projected_v": proj_x_v,
            "projected_a": proj_x_a,
            "s_l": s_l,
            "s_v": s_v,
            "s_a": s_a,
            "c_l": c_l,
            "c_v": c_v,
            "c_a": c_a,
            "s_l_r": s_l_r,
            "s_v_r": s_v_r,
            "s_a_r": s_a_r,
            "recon_l": recon_l,
            "recon_v": recon_v,
            "recon_a": recon_a,
            "c_l_sim": c_l_sim,
            "c_v_sim": c_v_sim,
            "c_a_sim": c_a_sim,
            "logits_l_hetero": logits_l_high,
            "logits_v_hetero": logits_v_high,
            "logits_a_hetero": logits_a_high,
            "logits_c": logits_c,
            "output_logit": output,
        }
        if self.use_mine_loss:
            res["mine_loss"] = self._compute_mine_loss(model_x_l, model_x_a, model_x_v)
        return res


# Compatibility alias: allows `getattr(DLF_sle, "DLF")` in existing run scripts.
DLF = DLF_sle
