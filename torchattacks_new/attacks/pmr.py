import torch
import torch.nn as nn

from ..attack import Attack


class PMR(Attack):
    r"""

    Distance Measure : L1+L2

    Arguments:
        model (nn.Module): model to attack.
        k (float): sample rate. (Default: 0.618)
        lr (float): larger values converge faster to less accurate results. (Default: 0.01)
        binary_search_steps (int): number of times to adjust the k with binary search. (Default: 5)
        max_iterations (int): number of iterations to perform gradient descent. (Default: 100)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PMR(model,lr=0.01, k=0.618,max_iterations=100)
        >>> adv_images = attack(images, labels)
"""


    def __init__(
        self,
        model,
        k=0.618,
        lr=0.01,
        binary_search_steps=5,
        max_iterations=100,
    ):
        super().__init__("PMR", model)
        self.k = k
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        x = nn.Parameter(images.clone())
        x_k = nn.Parameter(images)
        y_k = x_k.clone()
        if self.targeted:
            labels = self.get_target_label(images, labels)

        #采样获取参数ka和lambda

        outputs = self.get_logits(x)
        kas = outputs.max(dim=1).values-outputs.min(dim=1).values
        #self.kappa = kas.clone().detach()*2
        y_one_hot = torch.eye(outputs.shape[1]).to(self.device)[labels]
        batch_size = x.shape[0]
        lambdas = torch.zeros(batch_size, device=self.device)
        loss = self.PMR_loss(outputs, y_one_hot,  lambdas, lambdas)
        loss.backward()
        s=x.grad.abs().view(batch_size, -1)
        lambdas = torch.kthvalue(s,int((x[0].numel()*self.k)),dim=1).values

        final_adv_images = images.clone()
        o_best = [1e10] * batch_size
        o_best = torch.Tensor(o_best).to(self.device)

        # Start outer circle
        for outer_step in range(self.binary_search_steps):
            self.global_step = 0
            lr = self.lr
            #Start inner circle
            for iteration in range(self.max_iterations):
                x_k.requires_grad = True
                self.global_step += 1
                # reset gradient
                if x_k.grad is not None:
                    x_k.grad.detach_()
                    x_k.grad.zero_()
                output = self.get_logits(x_k)
                L2_loss = self.L2_loss(x_k, images)
                cost = self.PMR_loss(output, y_one_hot, L2_loss, lambdas)
                # Gradient step
                g = torch.autograd.grad(
                    cost, x_k, retain_graph=False, create_graph=False
                )[0]
                x_k =x_k.detach() - g * lr
                #计算近端算子
                Thr = (lambdas * lr).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                y_nest = torch.where(x_k - images > Thr, x_k - Thr,torch.where(x_k < -Thr, x_k + Thr,images))
                #计算NAG动量
                x_k = y_nest + (y_nest - y_k)*self.global_step/(self.global_step+3)
                y_k = y_nest
                #向[0,1]box投影
                x_k = torch.clamp(x_k,min=0,max=1).detach()
                # 如果达到ka值下限，且reg_loss值有下降，则更新之
                L1_loss = self.L1_loss(x_k, images)
                L2_loss = self.L2_loss(x_k, images)
                loss = self.F_loss(output, y_one_hot)
                F_value = loss+lambdas*(L1_loss+L2_loss)
                idx = (loss<-0.1*kas)&(F_value<o_best)
                o_best[idx] = (F_value)[idx]
                final_adv_images[idx] = x_k[idx]
                # Ploynomial decay of learning rate
                lr = self.lr * (1 - self.global_step / self.max_iterations) ** 0.5
            #如果o_best不是10e10，则说明ka没有达到下限值，调整k值为0.6倍。如果所有的o_best都达标，break.
            idx = (o_best == 1e10)
            print(f"{idx.sum().item()}张图片未达标")
            if idx.sum().item() == 0:
                break
            else:
                self.k *= 0.6
                lambdas[idx] = torch.kthvalue(s[idx], int(images[0].numel() * self.k), dim=1).values
        return final_adv_images

    def L1_loss(self, x1, x2):
        Flatten = nn.Flatten()
        L1_loss = torch.abs(Flatten(x1) - Flatten(x2)).sum(dim=1)
        # L1_loss = L1.sum()
        return L1_loss

    def L2_loss(self, x1, x2):
        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()
        L2_loss = MSELoss(Flatten(x1), Flatten(x2)).sum(dim=1)
        # L2_loss = L2.sum()
        return L2_loss

    def PMR_loss(self, output, one_hot_labels, L2_loss, lams):

        # Not same as CW's f function
        other = torch.max(
            (1 - one_hot_labels) * output - (one_hot_labels * 1e4), dim=1
        )[0]
        real = torch.max(one_hot_labels * output, dim=1)[0]

        if self.targeted:
            F_loss = other - real
        else:
            F_loss = real - other
            #F_loss = torch.clamp((real - other), min=-self.kappa)

        loss = torch.sum(F_loss)+ torch.sum(L2_loss * lams)
        return loss
    def F_loss(self, output, one_hot_labels):

        # Not same as CW's f function
        other = torch.max(
            (1 - one_hot_labels) * output - (one_hot_labels * 1e4), dim=1
        )[0]
        real = torch.max(one_hot_labels * output, dim=1)[0]

        if self.targeted:
            F_loss = other - real
        else:
            F_loss = real - other
            #F_loss = torch.clamp((real - other), min=-self.kappa)
        return F_loss
