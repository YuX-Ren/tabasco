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
        train_materials: bool = False,
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
        self.train_materials = train_materials
        self.cond_embed = nn.Embedding(2, hidden_dim)

        self.enc_linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.enc_atom_type_embed = nn.Embedding(atom_dim + 1, hidden_dim) # + 1 for the pesudo lattice token
        self.lattice_token = atom_dim
        if self.train_materials:
            self.enc_lattice_embed = nn.Linear(3, hidden_dim, bias=False)

        self.linear_embed = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.atom_type_embed = nn.Embedding(atom_dim + 1, hidden_dim) # + 1 for the pesudo lattice token
        if self.train_materials:
            self.lattice_embed = nn.Linear(3, hidden_dim, bias=False)
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
        self.quant_conv = nn.Linear(hidden_dim, 6)
        self.quantizer = FSQ(levels)
        self.quant_conv_out = nn.Linear(6, hidden_dim)
        # self.quant_conv_out_norm = nn.LayerNorm(6)
        self.out_coord_linear = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, spatial_dim, bias=False),
        )
        if self.train_materials:
            self.lattice_linear = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 3, bias=False),
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
            if self.train_materials:
                self.lattice_cross_attention = nn.TransformerDecoderLayer(
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

    def encode(self, coord_ori, atomics_ori, padding_mask, lattices_ori=None):
        """Encode the input."""
        embed_coords = self.enc_linear_embed(coord_ori)
        embed_atom_types = self.enc_atom_type_embed(atomics_ori.argmax(dim=-1))
        if self.train_materials:
            embed_lattices_pos = self.enc_lattice_embed(lattices_ori)
            embed_pesudo_atoms = self.enc_atom_type_embed(torch.ones(embed_lattices_pos.shape[0], embed_lattices_pos.shape[1], device=embed_lattices_pos.device, dtype=torch.long) * self.lattice_token)
            embed_lattices = embed_lattices_pos + embed_pesudo_atoms
        h_in = embed_coords + embed_atom_types
        if self.train_materials:
            h_in = torch.cat([embed_lattices,h_in], dim=1)
        if self.implementation == "pytorch":
            encode_embed = self.enc_transformer(h_in, src_key_padding_mask=padding_mask)
        elif self.implementation == "reimplemented":
            encode_embed = self.enc_transformer(h_in, padding_mask=padding_mask)
        encode_embed = self.quant_conv(encode_embed)
        # encode_embed = self.quant_conv_out_norm(encode_embed)
        quant_embed, indices = self.quantizer(encode_embed)
        quant_embed = self.quant_conv_out(encode_embed)
        return quant_embed

    def forward(self, coord_ori, atomics_ori, padding_mask, coord_t, atomics_t, t, lattices_ori=None, lattices_t=None, mode="val") -> Tensor:
        """Forward pass of the module. Compatible: if self.train_materials is False, behavior equals the second version."""
        seq_len = coord_t.shape[1]
        real_mask = 1 - padding_mask.int()

        # If materials are trained, extend lengths by 3; otherwise keep seq_len
        if self.train_materials:
            dec_seq_len = seq_len + 3
        else:
            dec_seq_len = seq_len

        # build padding mask that matches h_in length
        if self.train_materials:
            dec_padding_mask = torch.cat([padding_mask, torch.zeros(coord_t.shape[0], 3, device=padding_mask.device)], dim=1)
        else:
            dec_padding_mask = padding_mask
        # build masks conditionally (only append 3 tokens if train_materials=True)
        if self.train_materials:
            dec_real_mask = torch.cat([torch.ones(coord_t.shape[0], 3, device=padding_mask.device, dtype=real_mask.dtype), real_mask], dim=1)
        else:
            dec_real_mask = real_mask

        embed_coords = self.linear_embed(coord_t)
        embed_atom_types = self.atom_type_embed(atomics_t.argmax(dim=-1))

        if self.train_materials:
            embed_lattices_pos = self.lattice_embed(lattices_t)
            embed_pesudo_atoms = self.atom_type_embed(torch.ones(embed_lattices_pos.shape[0], embed_lattices_pos.shape[1], device=embed_lattices_pos.device, dtype=torch.long) * self.lattice_token)
            embed_lattices = embed_lattices_pos + embed_pesudo_atoms
        # positional encoding: always generate length dec_seq_len, but when not training materials, dec_seq_len==seq_len
        if self.add_sinusoid_posenc:
            embed_posenc = self.positional_encoding(batch_size=coord_t.shape[0], seq_len=dec_seq_len) * dec_real_mask.unsqueeze(-1)
        else:
            embed_posenc = torch.zeros(coord_t.shape[0], dec_seq_len, self.hidden_dim, device=coord_t.device)

        # encode_embed uses only the first seq_len positions of embed_posenc
        if self.train_materials:
            encode_embed = self.encode(coord_ori, atomics_ori, dec_padding_mask, lattices_ori) * dec_real_mask.unsqueeze(-1) + embed_posenc * dec_real_mask.unsqueeze(-1)
        else:
            encode_embed = self.encode(coord_ori, atomics_ori, padding_mask) * real_mask.unsqueeze(-1) + embed_posenc * real_mask.unsqueeze(-1)

        embed_time = self.time_encoding(t).unsqueeze(1)

        # Build h_in: include lattices only if train_materials=True
        if self.train_materials:
            h_in_atoms = embed_coords + embed_atom_types
            h_in = torch.cat([embed_lattices, h_in_atoms], dim=1) + embed_posenc + embed_time
        else:
            # here embed_posenc has length seq_len (dec_seq_len==seq_len)
            h_in = embed_coords + embed_atom_types + embed_posenc + embed_time

        # Apply mask that matches h_in length (dec_real_mask is consistent with dec_seq_len)
        h_in = h_in * dec_real_mask.unsqueeze(-1)

        # concatenate encode_embed (length seq_len)
        h_in = torch.cat([h_in, encode_embed], dim=1)

        # construct cond_pos aligned to actual sequence lengths
        cond_pos = torch.cat(
            (
                torch.zeros(coord_t.shape[0], dec_seq_len, dtype=torch.long, device=coord_t.device),
                torch.ones(coord_t.shape[0], dec_seq_len, dtype=torch.long, device=coord_t.device),
            ),
            dim=-1,
        )
        h_in = h_in + self.cond_embed(cond_pos)



        # transformer expects a padding mask of length left_len + seq_len, which equals h_in.shape[1]
        full_padding_mask = torch.cat([dec_padding_mask, dec_padding_mask], dim=1)

        if self.implementation == "pytorch":
            h_out = self.transformer(h_in, src_key_padding_mask=full_padding_mask)
        elif self.implementation == "reimplemented":
            h_out = self.transformer(h_in, padding_mask=full_padding_mask)

        # keep only the decoder part
        h_out = h_out[:, :dec_seq_len, :]
        h_out = h_out * dec_real_mask.unsqueeze(-1)

        # compute coords / atom_logits like original
        if self.cross_attention:
            h_coord = self.coord_cross_attention(
                h_out[:, 3:dec_seq_len, :],
                h_in[:, 3:dec_seq_len, :],
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )
            coords = self.out_coord_linear(h_coord)
        else:
            coords = self.out_coord_linear(h_out[:, 3:dec_seq_len, :])

        if self.cross_attention:
            h_atom = self.atom_cross_attention(
                h_out[:, 3:dec_seq_len, :],
                h_in[:, 3:dec_seq_len, :],
                tgt_key_padding_mask=padding_mask,
                memory_key_padding_mask=padding_mask,
            )
            atom_logits = self.out_atom_type_linear(h_atom)
        else:
            atom_logits = self.out_atom_type_linear(h_out[:, 3:dec_seq_len, :])

        if self.train_materials:
            if self.cross_attention:
                h_lattice = self.lattice_cross_attention(
                    h_out[:, :3, :],
                    h_in[:, :3, :],
                )
                lattices = self.lattice_linear(h_lattice)
            return coords, atom_logits, lattices
        return coords, atom_logits
