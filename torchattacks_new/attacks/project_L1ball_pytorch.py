import torch


def project_l1ball(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    This algorithm project vectors v onto the l1ball  \sum |w_i| <= z.

    :param v: A torch tensor, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-l1ball: \sum |w_i| <= z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_l1ball_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.

        :param v: NxD torch tensor; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-l1ball: \sum w_i <= z.
        :return: w: result of the projection
        """
        z = torch.tensor(z)
        if z.dim() == 0:
            with torch.no_grad():
                shape = v.shape
                if shape[1] == 1:
                    w = v.clone().detach()
                    w[:] = torch.where(v.abs() > z, z * v.sign(), v)
                    return w

                mu = torch.sort(v.abs(), dim=1)[0]
                mu = torch.flip(mu, dims=(1,))
                cum_sum = torch.cumsum(mu, dim=1)
                j = torch.unsqueeze(torch.arange(1, shape[1] + 1, dtype=mu.dtype, device=mu.device), 0)
                rho = torch.sum(mu * j - cum_sum + z > 0.0, dim=1, keepdim=True) - 1.
                max_nn = cum_sum[torch.arange(shape[0]), rho[:, 0].long()]
                theta = torch.clamp((torch.unsqueeze(max_nn, -1) - z) / (rho.type(max_nn.dtype) + 1), min=0.0)
                w = torch.clamp(v.abs() - theta, min=0.0) * v.sign()
                return w
        else:
            with torch.no_grad():
                shape = v.shape
                N, D = shape[0], shape[1]

                # 处理单元素特例（每行独立处理）
                if D == 1:
                    # 向量化处理：每行应用对应的z[i]
                    z = z.view(-1, 1)  # (N,) -> (N,1)
                    w = torch.where(v.abs() > z, z * v.sign(), v)
                    return w

                # Step 1: 对每行的绝对值降序排列
                mu = torch.sort(v.abs(), dim=1)[0]  # 升序排列
                mu = torch.flip(mu, dims=(1,))  # 翻转得到降序 (N,D)

                # Step 2: 计算累积和
                cum_sum = torch.cumsum(mu, dim=1)  # (N,D)

                # Step 3: 生成列索引 [1,2,...,D]
                j = torch.arange(1, D + 1, dtype=mu.dtype, device=mu.device).unsqueeze(0)  # (1,D)

                # Step 4: 计算每行的rho（临界点）
                z_expanded = z.unsqueeze(1)  # (N,) -> (N,1)
                mask = (mu * j - cum_sum + z_expanded) > 0.0  # (N,D)
                rho = torch.sum(mask, dim=1, keepdim=True) - 1  # (N,1)

                # Step 5: 计算max_nn和theta
                rho_long = rho.squeeze(1).long()  # (N,)
                max_nn = cum_sum[torch.arange(N), rho_long]  # (N,)

                # 计算theta: (max_nn - z) / (rho + 1)
                theta_numerator = (max_nn - z)  # (N,)
                theta_denominator = (rho.squeeze(1) + 1).to(max_nn.dtype)  # (N,)
                theta = theta_numerator / theta_denominator  # (N,)
                theta = torch.clamp(theta.unsqueeze(1), min=0.0)  # (N,1)

                # Step 6: 应用投影
                w_abs = torch.clamp(v.abs() - theta, min=0.0)  # (N,D)
                w = w_abs * v.sign()  # 恢复符号

                return w


    with torch.no_grad():
        shape = v.shape

        if len(shape) == 1:
            return _project_l1ball_2d(torch.unsqueeze(v, 0), z)[0, :]
        else:
            axis = axis % len(shape)
            t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
            tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
            v_t = v.permute(t_shape)
            v_t_shape = v_t.shape
            v_t_unroll = torch.reshape(v_t, (-1, v_t_shape[-1]))

            w_t = _project_l1ball_2d(v_t_unroll, z)

            w_t_reroll = torch.reshape(w_t, v_t_shape)
            return w_t_reroll.permute(tt_shape)
