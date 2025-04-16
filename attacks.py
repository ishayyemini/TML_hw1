import torch
import torch.nn as nn


class PGDAttack:
    """
    White-box L_inf PGD attack using the cross-entropy loss
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps]. The attack optionally 
        performs random initialization and early stopping, depending on the 
        self.rand_init and self.early_stop flags.
        """
        x_adv = x.clone().detach().to(x.device)
        if self.rand_init:
            x_adv = x_adv + torch.zeros_like(x).uniform_(-self.eps, self.eps).to(x.device)

        for _ in range(self.n):
            x_adv.requires_grad_(True)
            self.model.zero_grad()

            outputs = self.model(x_adv)
            preds = torch.argmax(outputs, dim=1)

            if self.early_stop:
                if (targeted and torch.all(preds == y)) or ((not targeted) and torch.all(preds != y)):
                    break

            loss = self.loss_func(outputs, y)
            for loss_i in range(loss.size(0)):
                loss[loss_i].backward(retain_graph=True)

            with torch.no_grad():
                grad = x_adv.grad.sign()

                if self.early_stop:
                    which_update = ~(preds == y) if targeted else preds == y
                    grad = grad * which_update.view(-1, 1, 1, 1)

                if targeted:
                    x_adv -= self.alpha * grad
                else:
                    x_adv += self.alpha * grad

                x_adv = torch.clip(x_adv, x - self.eps, x + self.eps)
                x_adv = torch.clip(x_adv, 0, 1)

        return x_adv.detach()


class NESBBoxPGDAttack:
    """
    Query-based black-box L_inf PGD attack using the cross-entropy loss, 
    where gradients are estimated using Natural Evolutionary Strategies 
    (NES).
    """

    def __init__(self, model, eps=8 / 255., n=50, alpha=1 / 255., momentum=0.,
                 k=200, sigma=1 / 255., rand_init=True, early_stop=True):
        """
        Parameters:
        - model: model to attack
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - momentum: a value in [0., 1.) controlling the "weight" of
             historical gradients estimating gradients at each iteration
        - k: the model is queries 2*k times at each iteration via 
              antithetic sampling to approximate the gradients
        - sigma: the std of the Gaussian noise used for querying
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.model = model
        self.eps = eps
        self.n = n
        self.alpha = alpha
        self.momentum = momentum
        self.k = k
        self.sigma = sigma
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def _nes_estimate_grad(self, x, y):
        g = torch.zeros_like(x).to(x.device)

        for i in range(self.k):
            u = torch.randn_like(x).to(x.device)

            with torch.no_grad():
                output_plus = self.model(x + self.sigma * u)
                loss_plus = self.loss_func(output_plus, y)
                g += loss_plus.view(-1, 1, 1, 1) * u

                output_minus = self.model(x - self.sigma * u)
                loss_minus = self.loss_func(output_minus, y)
                g -= loss_minus.view(-1, 1, 1, 1) * u

        return g / (2 * self.k * self.sigma)

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns:
        1- The adversarially perturbed samples, which lie in the ranges [0, 1] 
            and [x-eps, x+eps].
        2- A vector with dimensionality len(x) containing the number of queries for
            each sample in x.
        """
        x_adv = x.clone().detach().to(x.device)
        if self.rand_init:
            x_adv = x_adv + torch.zeros_like(x).uniform_(-self.eps, self.eps).to(x.device)

        num_queries = torch.zeros_like(y).to(x.device)
        grad = torch.zeros_like(x).to(x.device)

        with torch.no_grad():
            for _ in range(self.n):
                outputs = self.model(x_adv)
                preds = torch.argmax(outputs, dim=1)

                if self.early_stop:
                    if (targeted and torch.all(preds == y)) or ((not targeted) and torch.all(preds != y)):
                        break

                new_grad = self._nes_estimate_grad(x_adv, y)
                grad = self.momentum * grad + (1 - self.momentum) * new_grad
                sign = torch.sign(grad)

                which_update = ~(preds == y) if targeted else preds == y
                if self.early_stop:
                    sign = sign * which_update.view(-1, 1, 1, 1)

                if targeted:
                    x_adv -= self.alpha * sign
                else:
                    x_adv += self.alpha * sign

                x_adv = torch.clip(x_adv, x - self.eps, x + self.eps)
                x_adv = torch.clip(x_adv, 0, 1)

                num_queries += which_update * 2 * self.k

        return x_adv, num_queries


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return sum((1 / len(self.models)) * model(x) for model in self.models)

class PGDEnsembleAttack:
    """
    White-box L_inf PGD attack against an ensemble of models using the 
    cross-entropy loss
    """

    def __init__(self, models, eps=8 / 255., n=50, alpha=1 / 255.,
                 rand_init=True, early_stop=True):
        """
        Parameters:
        - models (a sequence): an ensemble of models to attack (i.e., the
              attack aims to decrease their expected loss)
        - eps: attack's maximum norm
        - n: max # attack iterations
        - alpha: PGD's step size at each iteration
        - rand_init: a flag denoting whether to randomly initialize
              adversarial samples in the range [x-eps, x+eps]
        - early_stop: a flag denoting whether to stop perturbing a 
              sample once the attack goal is met. If the goal is met
              for all samples in the batch, then the attack returns
              early, before completing all the iterations.
        """
        self.models = models
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def execute(self, x, y, targeted=False):
        """
        Executes the attack on a batch of samples x. y contains the true labels 
        in case of untargeted attacks, and the target labels in case of targeted 
        attacks. The method returns the adversarially perturbed samples, which
        lie in the ranges [0, 1] and [x-eps, x+eps].
        """
        model = EnsembleModel(self.models)
        pgd = PGDAttack(model, self.eps, self.n, self.alpha, self.rand_init, self.early_stop)
        return pgd.execute(x, y, targeted)
