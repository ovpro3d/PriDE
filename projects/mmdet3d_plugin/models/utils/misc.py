import torch
import torch.nn.functional as F
import tempfile
import kornia

SIGMOID_MAX = 9.21
LOGIT_MAX = 0.99999


def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:])
    return memory * prev_exist

def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape

        view_shape = [1 for _ in range(len(feat_shape))]
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)

        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX)
    return torch.sigmoid(tensor)


def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def cartesian(anchor, pc_range, use_sigmoid=True):
    if use_sigmoid:
        xyz = safe_sigmoid(anchor[..., :3])  # (Na, 3)
    else:
        xyz = anchor[..., :3].clamp(min=1e-6, max=1 - 1e-6)
    xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)  # (Na, 3)

    return xyz


def reverse_cartesian(xyz, pc_range, use_sigmoid=True):
    xxx = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    yyy = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    zzz = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    unitxyz = torch.stack([xxx, yyy, zzz], dim=-1)
    if use_sigmoid:
        anchor = safe_inverse_sigmoid(unitxyz)
    else:
        anchor = unitxyz.clamp(min=1e-6, max=1 - 1e-6)
    return anchor


def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]

    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]

    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)

    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]


def quat_mul(q, p):
    """
    q: (..., 4)  (w,x,y,z)
    p: (..., 4)
    return: (...,4)
    """
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = p.unbind(-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def transform_reference_points(reference_points, egopose, vel=None, time_diff=None, reverse=False, translation=True):
    if vel is not None:
        assert time_diff is not None
        reference_points[..., :2] += vel * time_diff[:, None, None]
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0

    expand_dims = reference_points.dim() - 2
    new_shape = matrix.shape[:1] + (1, ) * expand_dims + matrix.shape[1:]
    matrix_expanded = matrix.view(*new_shape)
    reference_points = (matrix_expanded @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]

    if vel is not None:
        vel = torch.matmul(
            matrix.unsqueeze(1)[..., :2, :2], vel[..., None]
        ).squeeze(-1)  # (B, K, 2)
    return reference_points, vel


def transform_superellipsoids(sqs, T, vel=None, time_diff=None):
    """
    Args:
        sqs: (B, Q, N, 13), (x, y, z, sx, sy, sz, q1, q2, q3, q4, opa, e1, e2)
        T: (B, M, 4, 4)
        vel: (B, Q, N, 2), (vx, vy)
        time_diff: (B, M) or None
    Returns:
        sqs_new: (B, M, Q, N, 15)
    """
    B, Q, N, D = sqs.shape
    _, M, _, _ = T.shape

    mean = sqs[..., 0:3]   # (B,Q,N,3)
    scales = sqs[..., 3:6]
    quat = sqs[..., 6:10]  # (B,Q,N,4)
    opa = sqs[..., 10:11]
    uv = sqs[..., 11:13]

    if vel is not None:
        assert time_diff is not None
        time_diff = time_diff[:, :, None, None, None].to(sqs)  # (B, M, 1, 1, 1)
        mean = mean[:, None].repeat(1, M, 1, 1, 1)
        mean[..., :2] += vel[:, None, ..., :2] * time_diff        # (B, M, Q, N, 3)
    else:
        mean = mean[:, None].repeat(1, M, 1, 1, 1)

    mean_h = torch.cat([mean, torch.ones_like(mean[..., :1])], dim=-1)  # (B,M,Q,N,4)
    T_expanded = T[:, :, None, None, :, :]  # (B, M, 1, 1, 4, 4)
    mean_new = (T_expanded @ mean_h.unsqueeze(-1)).squeeze(-1)[..., :3]  # (B, M, Q, N, 3)

    R_T = T_expanded[..., :3, :3]  # (B, M, 1, 1, 3, 3)
    q_T = kornia.geometry.conversions.rotation_matrix_to_quaternion(R_T)  # (B, M, 1, 1, 4)
    # quat_new = kornia.geometry.conversions.quaternion_multiply(q_T, quat[:, None])  # (B, M, Q, N, 4)
    quat_new = quat_mul(q_T, quat[:, None])  # (B,M,Q,N,4)

    scales = scales[:, None].expand(B, M, Q, N, 3)
    opa = opa[:, None].expand(B, M, Q, N, 1)
    uv = uv[:, None].expand(B, M, Q, N, 2)

    sqs_new = torch.cat([
        mean_new,
        scales,
        quat_new,
        opa,
        uv,
    ], dim=-1)  # (B,M,Q,N,13)

    return sqs_new


class DumpConfig:
    def __init__(self):
        self.enabled = False
        self.out_dir = tempfile.mkdtemp()
        self.stage_count = 0
        self.frame_count = 0


DUMP = DumpConfig()