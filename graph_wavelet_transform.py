import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_dense_adj


def construct_knn_graph_batch(point_clouds, k=10, self_loop=True, weight_mode='sqrt'):
    B, N, _ = point_clouds.shape
    device = point_clouds.device

    pos = point_clouds.view(B * N, 3)
    batch = torch.arange(B, device=device).repeat_interleave(N)

    edge_index = knn_graph(pos, k=k, batch=batch, loop=False)  # [2, E]
    row, col = edge_index
    dist = torch.norm(pos[row] - pos[col], dim=1)

    if weight_mode == 'sqrt':
        weights = dist ** 0.5
    elif weight_mode == 'euclidean':
        weights = dist
    elif weight_mode == 'inverse':
        weights = 1.0 / (dist + 1e-8)
    elif weight_mode == 'gaussian':
        sigma = torch.mean(dist)
        weights = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    A_dense_batch = to_dense_adj(edge_index, edge_attr=weights, batch=batch, max_num_nodes=N)  # (B, N, N)

    A_dense_batch = 0.5 * (A_dense_batch + A_dense_batch.transpose(1, 2))

    if self_loop:
        eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        A_dense_batch = A_dense_batch + eye

    return A_dense_batch


def batch_chebyshev_wavelet_transform(point_clouds, k=10, K=3, kernel_type='highpass'):
    B, N, _ = point_clouds.shape
    device = point_clouds.device
    A_dense_batch = construct_knn_graph_batch(point_clouds, k)
    D_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(torch.sum(A_dense_batch, dim=-1) + 1e-6))  # [B,N,N]
    L_batch = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1) - torch.bmm(
        torch.bmm(D_inv_sqrt, A_dense_batch), D_inv_sqrt)
    wavelet_operator_batch = chebyshev_wavelet_operator_batch(L_batch, K, kernel_type)
    return wavelet_operator_batch


def chebyshev_wavelet_operator_batch(L_batch, K=3, kernel_type='highpass'):
    B, N, _ = L_batch.shape
    device = L_batch.device

    I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
    lambda_max = 2.0
    L_tilde = (2.0 / lambda_max) * L_batch - I

    T_k_minus_two = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)  # T_0(L)
    T_k_minus_one = L_tilde.clone()  # T_1(L)
    cheb_polys = [T_k_minus_two, T_k_minus_one]

    for _ in range(2, K + 1):
        T_k = 2 * torch.bmm(L_tilde, T_k_minus_one) - T_k_minus_two
        cheb_polys.append(T_k)
        T_k_minus_two, T_k_minus_one = T_k_minus_one, T_k

    if kernel_type == 'lowpass':
        coeffs = [1.0 / (k + 1) for k in range(K + 1)]
    elif kernel_type == 'highpass':
        coeffs = [0.0 if k < 1 else (-1) ** k for k in range(K + 1)]
    elif kernel_type == 'custom':
        coeffs = [1.0 for _ in range(K + 1)]
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    wavelet_operator = sum(c * T for c, T in zip(coeffs, cheb_polys))

    return wavelet_operator

