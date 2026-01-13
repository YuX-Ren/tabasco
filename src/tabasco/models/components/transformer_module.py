from typing import Optional

import torch
import torch.nn as nn

from torch import Tensor

from tabasco.models.components.common import SwiGLU
from tabasco.models.components.positional_encoder import (
    SinusoidEncoding,
    TimeFourierEncoding,
)
from tabasco.models.components.transformer import Transformer
from tabasco.models.components.fsq import FSQ

class TransformerModule(nn.Module):
    """Basic Transformer model for molecule generation."""

    def __init__(
        self,
        spatial_dim: int,
        atom_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        activation: str = "SiLU",
        implementation: str = "pytorch",  # "pytorch" "reimplemented"
        cross_attention: bool = False,
        add_sinusoid_posenc: bool = True,
        concat_combine_input: bool = False,
        custom_weight_init: Optional[str] = None,
        levels: list[int] = [4, 4, 4, 4, 4, 4],
        kl_dim: int = 6,
        kl_weight: float = 1e-6,
        train_diffusion: bool = False,
    ):
        """
        Args:
            custom_weight_init: None, "xavier", "kaiming", "orthogonal", "uniform", "eye", "normal"
            (uniform does not work well)
        """
        super().__init__()

        # Normalize custom_weight_init if it's the string "None"
        if isinstance(custom_weight_init, str) and custom_weight_init.lower() == "none":
            custom_weight_init = None

        self.input_dim = spatial_dim + atom_dim
        self.time_dim = 1
        self.comb_input_dim = self.input_dim + self.time_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.implementation = implementation
        self.cross_attention = cross_attention
        self.add_sinusoid_posenc = add_sinusoid_posenc
        self.concat_combine_input = concat_combine_input
        self.custom_weight_init = custom_weight_init
        print(f"Implementation: {self.implementation}")
        self.kl_weight = kl_weight
        self.cond_embed = nn.Embedding(2, hidden_dim)
        self.train_diffusion = train_diffusion
        self.enc_linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.enc_atom_type_embed = nn.Embedding(atom_dim, hidden_dim)

        self.linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.atom_type_embed = nn.Embedding(atom_dim, hidden_dim)

        if self.add_sinusoid_posenc:
            self.positional_encoding = SinusoidEncoding(
                posenc_dim=hidden_dim, max_len=90
            )

        if self.concat_combine_input:
            self.combine_input = nn.Linear(4 * hidden_dim, hidden_dim)

        self.time_encoding = TimeFourierEncoding(posenc_dim=hidden_dim, max_len=200)

        if activation == "SiLU":
            activation = nn.SiLU(inplace=False)
        elif activation == "ReLU":
            activation = nn.ReLU(inplace=False)
        elif activation == "SwiGLU":
            activation = SwiGLU()
        else:
            raise ValueError(f"Invalid activation: {activation}")

        if self.implementation == "pytorch":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.enc_transformer = nn.TransformerEncoder(
                enc_layer, num_layers=num_layers//4
            )
            diff_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                diff_layer, num_layers=num_layers
            )
        elif self.implementation == "reimplemented":
            self.enc_transformer = Transformer(
                dim=hidden_dim,
                num_heads=num_heads,
                depth=num_layers//4,
            )
            self.transformer = Transformer(
                dim=hidden_dim,
                num_heads=num_heads,
                depth=num_layers,
            )
        else:
            raise ValueError(f"Invalid implementation: {self.implementation}")
        self.quant_conv = nn.Linear(hidden_dim, kl_dim*2)
        # self.quantizer = FSQ(levels)
        self.quant_conv_out = nn.Linear(kl_dim, hidden_dim)
        # self.quant_conv_out_norm = nn.LayerNorm(6)
        self.out_coord_linear = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )

        self.out_atom_type_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=False),
            nn.Linear(hidden_dim, atom_dim),
        )

        # Add cross attention layers
        if self.cross_attention:
            self.coord_cross_attention = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )

            self.atom_cross_attention = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                norm_first=True,
            )

        self._atom_size_tuples = []

        if self.custom_weight_init is not None:
            print(f"Initializing weights with {self.custom_weight_init}!!!!")
            self.apply(self._custom_weight_init)

    def _custom_weight_init(self, module):
        """Initialize the weights of the module with a custom method."""
        for name, param in module.named_parameters():
            if "weight" in name and param.data.dim() >= 2:
                if self.custom_weight_init == "xavier":
                    nn.init.xavier_uniform_(param)
                elif self.custom_weight_init == "kaiming":
                    nn.init.kaiming_uniform_(param)
                elif self.custom_weight_init == "orthogonal":
                    nn.init.orthogonal_(param)
                elif self.custom_weight_init == "uniform":
                    nn.init.uniform_(param)
                elif self.custom_weight_init == "eye":
                    nn.init.eye_(param)
                elif self.custom_weight_init == "normal":
                    nn.init.normal_(param)
                else:
                    raise ValueError(
                        f"Invalid custom weight init: {self.custom_weight_init}"
                    )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # 如果是训练模式，加入随机噪声；如果是推理模式，直接返回均值
        if self.training or self.train_diffusion:
            return mu + eps * std
        else:
            return mu

    def encode(self, coord_ori, atomics_ori, padding_mask):
        """Encode the input."""
        embed_coords = self.enc_linear_embed(coord_ori)
        embed_atom_types = self.enc_atom_type_embed(atomics_ori.argmax(dim=-1))
        h_in = embed_coords + embed_atom_types
        if self.implementation == "pytorch":
            encode_embed = self.enc_transformer(h_in, src_key_padding_mask=padding_mask)
        elif self.implementation == "reimplemented":
            encode_embed = self.enc_transformer(h_in, padding_mask=padding_mask)
        # encode_embed = self.quant_conv(encode_embed)
        # encode_embed = self.quant_conv_out_norm(encode_embed)
        # quant_embed, indices = self.quantizer(encode_embed)
        # quant_embed = self.quant_conv_out(quant_embed)
        moments = self.quant_conv(encode_embed) 
        mu, logvar = torch.chunk(moments, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        z_projected = self.quant_conv_out(z)
        kl_item = (1 + logvar - mu.pow(2) - logvar.exp()) * ~padding_mask.unsqueeze(-1)
        kl_loss = -0.5 * torch.mean(kl_item.sum(dim=-1)) * self.kl_weight
        if self.training:
            return z_projected, kl_loss
        else:
            return z_projected, 0.0

    def encode_z(self, coord_ori, atomics_ori, padding_mask, return_kl =False):
        """Encode the input."""
        embed_coords = self.enc_linear_embed(coord_ori)
        embed_atom_types = self.enc_atom_type_embed(atomics_ori.argmax(dim=-1))
        h_in = embed_coords + embed_atom_types
        if self.implementation == "pytorch":
            encode_embed = self.enc_transformer(h_in, src_key_padding_mask=padding_mask)
        elif self.implementation == "reimplemented":
            encode_embed = self.enc_transformer(h_in, padding_mask=padding_mask)
        moments = self.quant_conv(encode_embed) 
        mu, logvar = torch.chunk(moments, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        kl_item = (1 + logvar - mu.pow(2) - logvar.exp()) * ~padding_mask.unsqueeze(-1)
        kl_loss = -0.5 * torch.mean(kl_item.sum(dim=-1)) * self.kl_weight
        if return_kl:
            return z, kl_loss
        else:
            return z

    def decode_z(self, z, coord_t, atomics_t, padding_mask, t, mode = "val") -> Tensor:
        """Decode the input."""
        encode_embed = self.quant_conv_out(z)
        """Forward pass of the module."""

        real_mask = 1 - padding_mask.int()

        embed_coords = self.linear_embed(coord_t)
        embed_atom_types = self.atom_type_embed(atomics_t.argmax(dim=-1))

        if self.add_sinusoid_posenc:
            embed_posenc = self.positional_encoding(
                batch_size=coord_t.shape[0], seq_len=coord_t.shape[1]
            )
        else:
            embed_posenc = torch.zeros(
                coord_t.shape[0], coord_t.shape[1], self.hidden_dim
            ).to(coord_t.device)
        encode_embed = encode_embed * real_mask.unsqueeze(-1) + embed_posenc * real_mask.unsqueeze(-1)

        embed_time = self.time_encoding(t).unsqueeze(1)

        assert embed_posenc.shape == embed_coords.shape == embed_atom_types.shape, (
            f"embed_posenc.shape: {embed_posenc.shape}, embed_coords.shape: {embed_coords.shape}, embed_atom_types.shape: {embed_atom_types.shape}"
        )

        if self.concat_combine_input:
            embed_time = embed_time.repeat(1, coord_t.shape[1], 1)
            h_in = torch.cat(
                [embed_coords, embed_atom_types, embed_posenc, embed_time], dim=-1
            )
            assert h_in.shape == (
                coord_t.shape[0],
                coord_t.shape[1],
                4 * self.hidden_dim,
            ), f"h_in.shape: {h_in.shape}"
            h_in = self.combine_input(h_in)
            assert h_in.shape == (coord_t.shape[0], coord_t.shape[1], self.hidden_dim), (
                f"h_in.shape: {h_in.shape}"
            )
        else:
            h_in = embed_coords + embed_atom_types + embed_posenc + embed_time
        h_in = h_in * real_mask.unsqueeze(-1)
        # h_in = h_in + encode_embed
        h_in = torch.cat( [h_in, encode_embed], dim=1)
        cond_pos = torch.cat(
                (
                    torch.zeros(coord_t.shape[0], coord_t.shape[1], dtype=torch.long, device=coord_t.device),
                    torch.ones(coord_t.shape[0], coord_t.shape[1], dtype=torch.long, device=coord_t.device),
                ),
                dim=-1,
            )
        h_in = h_in + self.cond_embed(cond_pos)
        
        if self.implementation == "pytorch":
            h_out = self.transformer(h_in, src_key_padding_mask=torch.cat( [padding_mask, padding_mask], dim=1))
        elif self.implementation == "reimplemented":
            h_out = self.transformer(h_in, padding_mask=torch.cat( [padding_mask, padding_mask], dim=1))

        # load the half of the input to the output
        seq_len = coord_t.shape[1]
        h_out = h_out[:, :seq_len, :]
        h_out = h_out * real_mask.unsqueeze(-1)

        if self.cross_attention:
            h_coord = self.coord_cross_attention(
                h_out,
                h_in[:, :seq_len, :],
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )
            coords = self.out_coord_linear(h_coord)
        else:
            coords = self.out_coord_linear(h_out)

        if self.cross_attention:
            h_atom = self.atom_cross_attention(
                h_out,
                h_in[:, :seq_len, :],
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )
            atom_logits = self.out_atom_type_linear(h_atom)
        else:
            atom_logits = self.out_atom_type_linear(h_out)

        return coords, atom_logits

    def forward(self, coord_ori, atomics_ori, padding_mask, coord_t, atomics_t, t, mode = "val") -> Tensor:
        """Forward pass of the module."""

        real_mask = 1 - padding_mask.int()

        embed_coords = self.linear_embed(coord_t)
        embed_atom_types = self.atom_type_embed(atomics_t.argmax(dim=-1))

        if self.add_sinusoid_posenc:
            embed_posenc = self.positional_encoding(
                batch_size=coord_t.shape[0], seq_len=coord_t.shape[1]
            )
        else:
            embed_posenc = torch.zeros(
                coord_t.shape[0], coord_t.shape[1], self.hidden_dim
            ).to(coord_t.device)
        z, kl_loss = self.encode_z(coord_ori, atomics_ori, padding_mask, return_kl=True)
        coords, atom_logits = self.decode_z(z, coord_t, atomics_t, padding_mask, t)

        return coords, atom_logits, kl_loss
