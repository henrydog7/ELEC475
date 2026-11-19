import torch.nn.functional as F


def response_kd_loss(z_s, z_t, y, alpha=0.5, beta=0.5, tau=4.0, ignore_index=255):
    """Combine label cross-entropy and temperature-softened teacher responses."""
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if z_s.shape != z_t.shape or y.shape != z_s.shape[:1] + z_s.shape[2:]:
        raise ValueError("Expected matching logits (N, C, H, W) and labels (N, H, W).")

    valid = y != ignore_index
    if not valid.any():
        raise ValueError("Response KD requires at least one non-ignored target pixel.")

    label_loss = F.cross_entropy(z_s, y, ignore_index=ignore_index)
    teacher_probabilities = F.softmax(z_t / tau, dim=1)
    student_log_probabilities = F.log_softmax(z_s / tau, dim=1)
    pixel_loss = -(teacher_probabilities * student_log_probabilities).sum(dim=1)
    response_loss = pixel_loss[valid].mean()
    return alpha * label_loss + beta * response_loss


def feature_kd_loss(f_s, f_t):
    """Compare projected student and teacher features using cosine distance."""
    if f_s.ndim != 4 or f_t.ndim != 4 or f_s.shape[:2] != f_t.shape[:2]:
        raise ValueError("Expected feature maps with matching batch and channel dimensions.")

    f_s = F.interpolate(f_s, size=f_t.shape[-2:], mode="bilinear", align_corners=False)
    return (1 - F.cosine_similarity(f_s, f_t, dim=1)).mean()
