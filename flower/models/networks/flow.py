import torch
import torch.nn as nn


class StandardNormal:
    def log_prob(self, z):
        return -0.5 * torch.sum(z**2 + torch.log(torch.tensor(2 * torch.pi)), dim=-1)

    def sample(self, shape):
        return torch.randn(*shape)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class ZeroLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class InvertiblePLU(nn.Module):
    def __init__(
            self,
            features: int,
    ):
        super().__init__()
        self.features = features
        w_shape = (self.features, self.features)
        w = torch.empty(w_shape)
        nn.init.orthogonal_(w)
        P, L, U = torch.linalg.lu(w)

        self.s = nn.Parameter(torch.diag(U))
        self.U = nn.Parameter(U - torch.diag(self.s))
        self.L = nn.Parameter(L)

        self.P = nn.Parameter(P, requires_grad=False)
        self.P_inv = nn.Parameter(torch.linalg.inv(P), requires_grad=False)

    def forward(self, x, cond=None):
        dims = x.shape
        if len(dims) == 3:
            x = x.reshape(dims[0], dims[1] * dims[2])
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.features, device=x.device)
        U = torch.triu(self.U, diagonal=1)
        s = self.s

        W = self.P @ L @ (U + torch.diag(s))

        z = x @ W
        logdet = torch.sum(torch.log(torch.abs(s)), dim=0, keepdim=True)
        if len(dims) == 3:
            z = z.reshape(*dims)
        return z, logdet

    def inverse(self, x, cond=None):
        dims = x.shape
        if len(dims) == 3:
            x = x.reshape(dims[0], dims[1] * dims[2])
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.features, device=x.device)
        U = torch.triu(self.U, diagonal=1)
        s = self.s

        eye = torch.eye(self.features, device=x.device, dtype=U.dtype)

        U_inv = torch.linalg.solve_triangular(U + torch.diag(s), eye, upper=True)
        L_inv = torch.linalg.solve_triangular(L, eye, upper=False, unitriangular=True)

        W_inv = U_inv @ L_inv @ self.P_inv
        z = x @ W_inv
        if len(dims) == 3:
            z = z.reshape(*dims)
        return z


class SubBlock(nn.Module):
    def __init__(self, cond_dim, dim=64, num_heads=8, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=attn_pdrop)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True,
                                                dropout=attn_pdrop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout_resid = nn.Dropout(resid_pdrop)  # for residual connections
        self.cond_proj = nn.Linear(cond_dim, dim)  # cond: [B, cond_dim] -> [B, hidden_dim]

    def forward(self, inp):
        x, cond = inp
        # Self-attention
        x1 = self.norm1(x)
        sa_out, _ = self.self_attn(x1, x1, x1)
        x = x + self.dropout_resid(sa_out)

        # Cross-attention with condition
        x2 = self.norm2(x)
        cond_proj = self.cond_proj(cond).unsqueeze(1)  # [B, 1, hidden_dim]
        ca_out, _ = self.cross_attn(x2, cond_proj, cond_proj)
        x = x + self.dropout_resid(ca_out)

        return x, cond


class TransformerBlock(nn.Module):
    def __init__(self, in_dim, cond_dim, dim=64, num_heads=8, attn_pdrop=0.1, resid_pdrop=0.1, mlp_pdrop=0.1, block_depth=1):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, dim)  # project scalar token to hidden dim
        # self.output_proj_mult = nn.Linear(dim, in_dim)  # project back to scalar
        # self.output_proj_shift = nn.Linear(dim, in_dim)  # project back to scalar
        self.dropout_ff = nn.Dropout(mlp_pdrop)  # for MLP outputs

        self.subblocks = nn.Sequential(
            *[SubBlock(cond_dim, dim, num_heads, attn_pdrop, resid_pdrop) for _ in range(block_depth)]
        )
        self.norm3 = nn.LayerNorm(dim)

        self.ff_mult = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            self.dropout_ff,
            # ZeroLinear(dim * 2, in_dim)
            nn.Linear(dim * 2, in_dim)
        )
        self.ff_shift = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            self.dropout_ff,
            # ZeroLinear(dim * 2, in_dim)
            nn.Linear(dim * 2, in_dim)
        )

        self.cond_proj = nn.Linear(cond_dim, dim)  # cond: [B, cond_dim] -> [B, hidden_dim]

    def forward(self, x, cond):
        # print("INPUT SHAPE:", x.shape, flush=True)
        # x: [B, D] -> [B, D, 1] -> [B, D, hidden_dim]
        x = self.input_proj(x)  # [B, D, hidden_dim]

        x, _ = self.subblocks((x, cond))
        x = self.norm3(x)
        x_mult = self.ff_mult(x)
        x_shift = self.ff_shift(x)

        # x_mult = self.output_proj_mult(x_mult)
        # x_shift = self.output_proj_mult(x_shift)
        return x_mult, x_shift


# class AffineCoupling(nn.Module):
#     def __init__(self, mask, input_dim, cond_dim, affine_dim):
#         super().__init__()
#         self.mask = nn.Parameter(mask, requires_grad=False)
#         self.scale_net = MLP(input_dim + cond_dim, input_dim, affine_dim)
#         self.shift_net = MLP(input_dim + cond_dim, input_dim, affine_dim)
#         self.tanh = nn.Tanh()
#
#     def forward(self, x, cond):
#         x_masked = x * self.mask
#         net_in = torch.concatenate([x_masked, cond], dim=-1)
#         s = self.tanh(self.scale_net(net_in)) * (1 - self.mask)
#         t = self.shift_net(net_in) * (1 - self.mask)
#         y = x_masked + (1 - self.mask) * (x * torch.exp(s) + t)
#         log_det_jacobian = torch.sum(s, dim=-1)
#         return y, log_det_jacobian
#
#     def inverse(self, y, cond):
#         y_masked = y * self.mask
#         net_in = torch.concatenate([y_masked, cond], dim=-1)
#         s = self.tanh(self.scale_net(net_in)) * (1 - self.mask)
#         t = self.shift_net(net_in) * (1 - self.mask)
#         x = y_masked + (1 - self.mask) * ((y - t) * torch.exp(-s))
#         return x

class AffineCoupling(nn.Module):
    def __init__(self, act_window_size, action_dim, cond_dim, affine_dim=128, backbone='trans', n_heads=8, attn_pdrop=0.1, resid_pdrop=0.1, mlp_pdrop=0.1, block_depth=1):
        super().__init__()
        # assert dim % 2 == 0, "Input dimension must be even"
        self.backbone = backbone
        if backbone == 'trans':
            self.half_dim = act_window_size // 2
            self.net = TransformerBlock(action_dim, cond_dim, affine_dim, num_heads=n_heads, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, mlp_pdrop=mlp_pdrop, block_depth=block_depth)
        elif backbone == 'mse':
            self.net = TransformerBlock(action_dim, cond_dim, affine_dim, num_heads=n_heads, attn_pdrop=attn_pdrop,
                                        resid_pdrop=resid_pdrop, mlp_pdrop=mlp_pdrop, block_depth=block_depth)
        else:
            self.dim = act_window_size * action_dim
            self.half_dim = act_window_size * action_dim // 2
            self.scale_net = MLP(self.half_dim + cond_dim, self.half_dim, affine_dim)
            self.shift_net = MLP(self.half_dim + cond_dim, self.half_dim, affine_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, cond):
        if self.backbone != 'mse':
            x1, x2 = x[:, :self.half_dim], x[:, self.half_dim:]
            if self.backbone == 'trans':
                s, t = self.net(x1, cond)
                s = self.tanh(s)
            else:
                inp = torch.cat([x1, cond], dim=-1)
                s = self.tanh(self.scale_net(inp))
                t = self.shift_net(inp)
            y2 = x2 * torch.exp(s) + t
            if self.backbone == 'trans':
                log_det_jacobian = torch.sum(s, dim=-1).sum(-1)
            else:
                log_det_jacobian = torch.sum(s, dim=-1)
            y = torch.cat([x1, y2], dim=1)
            return y, log_det_jacobian
        else:
            s, _ = self.net(x, cond)
            return s, 0

    def inverse(self, y, cond):
        if self.backbone != 'mse':
            y1, y2 = y[:, :self.half_dim], y[:, self.half_dim:]
            if self.backbone == 'trans':
                s, t = self.net(y1, cond)
                s = self.tanh(s)
            else:
                inp = torch.cat([y1, cond], dim=-1)
                s = self.tanh(self.scale_net(inp))
                t = self.shift_net(inp)
            x2 = (y2 - t) * torch.exp(-s)
            x = torch.cat([y1, x2], dim=1)
            return x
        else:
            s, _ = self.net(y, cond)
            return s


def create_alternating_masks(dim, num_masks):
    masks = []
    base = torch.tensor([i % 2 for i in range(dim)], dtype=torch.float32)
    for i in range(num_masks):
        mask = base if i % 2 == 0 else 1 - base
        masks.append(mask)
    return masks


def create_random_masks(dim, num_layers):
    masks = []
    for _ in range(num_layers):
        mask = torch.randint(0, 2, (dim,), dtype=torch.float32)
        masks.append(mask)
    return masks


class Permutation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("perm", torch.randperm(dim))
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def forward(self, x, cond=None):
        return x[:, self.perm], 0.0  # no change to log-det

    def inverse(self, y, cond=None):
        return y[:, self.inv_perm]


class RealNVP(nn.Module):
    def __init__(self, act_window_size, action_dim, cond_dim, num_layers, affine_dim=128, backbone='trans', n_heads=8, attn_pdrop=0.1, resid_pdrop=0.1, mlp_pdrop=0.1, block_depth=1, use_plu=True):
        super().__init__()
        # self.dim = dim
        self.base_dist = StandardNormal()
        self.layers = nn.ModuleList()
        self.backbone = backbone
        for i in range(num_layers):
            if backbone == 'trans':
                self.layers.append(Permutation(act_window_size))
            elif backbone != 'mse':
                self.layers.append(Permutation(act_window_size * action_dim))
            self.layers.append(AffineCoupling(act_window_size, action_dim, cond_dim, affine_dim, backbone=backbone, n_heads=n_heads, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, mlp_pdrop=mlp_pdrop, block_depth=block_depth))
            if use_plu:
                if backbone == 'mlp':
                    self.layers.append(InvertiblePLU(act_window_size * action_dim))
                else:
                    self.layers.append(InvertiblePLU(act_window_size * action_dim))

    def forward(self, x, cond):
        log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ldj = layer(x, cond)
            log_det += ldj
        return x, log_det

    def inverse(self, z, cond):
        if self.backbone != 'mse':
            for layer in reversed(self.layers):
                z = layer.inverse(z, cond)
        else:
            for layer in self.layers:
                z = layer.inverse(z, cond)
        return z

    def log_prob(self, x, cond):
        z, log_det = self.forward(x, cond)
        if self.backbone == 'trans':
            z = z.reshape(z.shape[0], -1)
        return self.base_dist.log_prob(z) + log_det

# class RealNVP(nn.Module):
#     def __init__(self, dim, cond_dim, num_layers, affine_dim=128):
#         super().__init__()
#         self.dim = dim
#         self.masks = create_random_masks(dim, num_layers)
#         self.layers = nn.ModuleList([AffineCoupling(mask, dim, cond_dim, affine_dim) for mask in self.masks])
#         self.base_dist = StandardNormal()
#
#     def forward(self, x, cond):
#         log_det = torch.zeros(x.shape[0]).to(next(self.parameters()).device)
#         for layer in self.layers:
#             x, ldj = layer(x, cond)
#             log_det += ldj
#         return x, log_det
#
#     def inverse(self, z, cond):
#         for layer in reversed(self.layers):
#             z = layer.inverse(z, cond)
#         return z
#
#     def log_prob(self, x, cond):
#         z, log_det = self.forward(x, cond)
#         return self.base_dist.log_prob(z) + log_det