import torch
import torch.nn as nn

from ..attack import Attack
from .project_L1ball_pytorch import project_l1ball

class MIFGSML1MASK(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSML1MASK(model, eps=8/255, steps=10, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=100000 / 255, alpha=2 / 255, steps=10, decay=1.0):
        super().__init__("MIFGSML1MASK", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels,c):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        c = c.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for step in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad
            if step==0:
                mask = torch.zeros_like(grad)
                for i in range(len(c)):
                    mask[i] = torch.where(grad[i].abs() > torch.kthvalue(grad[i].abs().view(-1),k=int(3*299*299*(1-(self.eps*c[i]*255/(12.75*10*270000))**0.5)))[0], 1, 0)

            adv_images = adv_images.detach() + self.alpha*c.view(-1,1,1,1) * grad.sign()*mask
            noises=adv_images - images
            delta = project_l1ball(noises.view( noises.size(0),-1), z=self.eps*c, axis=-1)
            delta =delta.view(noises.size(0),noises.size(1),noises.size(2),noises.size(3))
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()


        return adv_images
