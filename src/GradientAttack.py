"""
created by milin zhang
"""
import torch
import torch.nn as nn
import numpy as np

class GradientAttack():
    """
    abstract class for all gradient-based attack
    """
    def __init__(self,loss_fn,epsilon,clip_min,clip_max,target) -> None:
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.target = target

    def adv_gen(self,*args,**kwds):
        raise NotImplementedError("adv_gen method must be implemented in sub-classes")
    
    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)

    def clip(self,xadv,x,norm=torch.inf):
        """
        clip function for all iterative methods
        """
        if norm == torch.inf:
            if self.clip_min is not None:
                lb = torch.clamp(x-self.epsilon,min=self.clip_min) # lower bound
            else:
                lb = x-self.epsilon
            xadv = torch.max(xadv,lb)
            if self.clip_max is not None:
                ub = torch.clamp(x+self.epsilon,max=self.clip_max) # upper bound
            else:
                ub = x+self.epsilon
            xadv = torch.min(xadv,ub)
        elif norm == 2: # projection 
            d = np.prod([*x.shape[1:]])
            delta = xadv - x
            batchsize = delta.size(0)
            deltanorm = torch.norm(delta.view(batchsize,-1),p=norm,dim=1)
            scale = np.sqrt(d)*self.epsilon/deltanorm
            scale[deltanorm<=(np.sqrt(d*self.epsilon))] = 1
            delta = (delta.transpose(0,-1)*scale).transpose(0,-1).contiguous()
            xadv = x + delta
            if self.clip_min is not None and self.clip_max is not None:
                xadv = torch.clamp(xadv,self.clip_min,self.clip_max)
        else:
            raise NotImplementedError
        return xadv.detach()
    
    def normalize(self,x,p):
        """
        normalize function for all methods
        """
        if p == torch.inf:
            x = x.sign()
        else:
            batch_size = x.size(0)
            norm = torch.norm(x.view(batch_size, -1), p, 1)
            x = (x.transpose(0,-1)/norm).transpose(0,-1).contiguous()
        return x

    def is_adv(self,model,x,y):
        with torch.no_grad():
            pred = model(x)
        if len(pred.shape) == 1:
            pred = pred[None,:]
        pred = torch.argmax(pred,dim=1)
        if self.target:
            return pred == y
        else:
            return pred != y

class FGSM(GradientAttack):
    """
    Fast Gradient Sign Method: https://arxiv.org/abs/1412.6572
    """
    def __init__(self,loss_fn=nn.CrossEntropyLoss(),epsilon=0.3,clip_min=0.,clip_max=1.,target=False) -> None:
        super(FGSM,self).__init__(loss_fn,epsilon,clip_min,clip_max,target)

    def adv_gen(self, model, x, target, epsilon = None):
        if epsilon is not None:
            self.epsilon = epsilon
        x.requires_grad = True
        y_pred = model(x)
        loss = self.loss_fn(y_pred,target)
        if self.target:
            loss = -loss
        model.zero_grad()
        loss.backward()
        xsign = x.grad.data.sign()
        xadv = x + self.epsilon * xsign
        if self.clip_min is not None and self.clip_max is not None:
            xadv = torch.clamp(xadv,self.clip_min,self.clip_max)
        return xadv


class BIM(GradientAttack):
    """
    Basic Iterative Method: https://arxiv.org/abs/1607.02533
    """
    def __init__(self,loss_fn=nn.CrossEntropyLoss(),epsilon=0.3,clip_min=0.,clip_max=1.,target=False) -> None:
        super(BIM,self).__init__(loss_fn, epsilon, clip_min, clip_max, target)
        
    def adv_gen(self, model, x, target, iters = 20, alpha = 0.1, epsilon = None):
        if epsilon is not None:
            self.epsilon = epsilon
        if iters == 0:
            iters = int(min(self.epsilon*255+4,1.25*self.epsilon*255)) # the scale of the original paper is 255
        if alpha is None:
            alpha = 1/iters
        with torch.no_grad():
            y_pred = model(x)
        
        xadv = x
        for i in range(iters):
            idx_mask = self.is_adv(model,xadv,target)
            if torch.sum(idx_mask.float()) == xadv.shape[0]: # early stop
                break
            x_temp, xadv_temp, target_temp = x[~idx_mask], xadv[~idx_mask].detach().clone(), target[~idx_mask]
            xadv_temp.requires_grad = True
            y_pred = model(xadv_temp)
            if len(y_pred.shape) == 1:
                y_pred = y_pred[None,:]
            loss = self.loss_fn(y_pred,target_temp)
            if self.target:
                loss = -loss
            model.zero_grad()
            loss.backward()
            xsign = xadv_temp.grad.data.sign()
            xadv_temp = xadv_temp + alpha*self.epsilon*xsign
            xadv_temp = self.clip(xadv_temp,x_temp)
            xadv[~idx_mask] = xadv_temp
        return xadv


class PGD(GradientAttack):
    """
    Projected Gradient Descent: https://arxiv.org/abs/1706.06083
    """
    def __init__(self,loss_fn=nn.CrossEntropyLoss(),epsilon=0.3,clip_min=0.,clip_max=1.,ord=torch.inf,target=False) -> None:
        super(PGD,self).__init__(loss_fn, epsilon, clip_min, clip_max, target)
        self.ord = ord

    def adv_gen(self, model, x, target, iters = 20, alpha = 0.1, epsilon = None):
        if epsilon is not None:
            self.epsilon = epsilon
        if alpha is None:
            alpha = 1/iters
        delta = torch.rand_like(x)*self.epsilon*2 - self.epsilon # uniform distribution [-eps, eps]
        xadv = self.clip(x + delta,x,self.ord)
        # xadv = x
        for i in range(iters):
            idx_mask = self.is_adv(model,xadv,target)
            if torch.sum(idx_mask.float()) == xadv.shape[0]: # early stop
                break
            x_temp, xadv_temp, target_temp = x[~idx_mask], xadv[~idx_mask].detach().clone(), target[~idx_mask]
            xadv_temp.requires_grad = True
            y_pred = model(xadv_temp)
            if len(y_pred.shape) == 1:
                y_pred = y_pred[None,:]
            loss = self.loss_fn(y_pred,target_temp)
            if self.target:
                loss = -loss
            model.zero_grad()
            loss.backward()
            g = self.normalize(xadv_temp.grad.data,self.ord)
            xadv_temp = xadv_temp + alpha*g
            xadv_temp = self.clip(xadv_temp,x_temp,self.ord)
            xadv[~idx_mask] = xadv_temp
        return xadv

class MI_FGSM(GradientAttack):
    """
    Momentum Iterative Method: https://arxiv.org/abs/1710.06081
    """
    def __init__(self,loss_fn=nn.CrossEntropyLoss(),epsilon=0.3,clip_min=0.,clip_max=1.,mu=1.,target=False) -> None:
        super(MI_FGSM,self).__init__(loss_fn, epsilon, clip_min, clip_max,target)
        self.mu = mu

    def adv_gen(self, model, x, target, iters = 20, norm = 1, alpha = 0.1, epsilon = None):
        # TODO: implement the targeted attack
        if epsilon is not None:
            self.epsilon = epsilon
        if iters == 0:
            iters = 2
        if alpha is None:
            alpha = 1/iters
        g = torch.zeros_like(x)
        xadv = x
        for i in range(iters):
            idx_mask = self.is_adv(model,xadv,target)
            if torch.sum(idx_mask.float()) == xadv.shape[0]: # early stop
                break
            x_temp, xadv_temp, target_temp = x[~idx_mask], xadv[~idx_mask].detach().clone(), target[~idx_mask]
            xadv_temp.requires_grad = True
            y_pred = model(xadv_temp)
            if len(y_pred.shape) == 1:
                y_pred = y_pred[None,:]
            loss = self.loss_fn(y_pred,target_temp)
            if self.target:
                loss = -loss
            model.zero_grad()
            loss.backward()

            g[~idx_mask] = self.mu*g[~idx_mask] + self.normalize(xadv_temp.grad.data,norm)
            xadv_temp = xadv_temp + alpha*self.epsilon*g[~idx_mask].sign()
            xadv_temp = self.clip(xadv_temp,x_temp)
            xadv[~idx_mask] = xadv_temp
        return xadv

