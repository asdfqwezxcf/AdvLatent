import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def cw_loss(inputs,targets):
    nclasses = inputs.shape[-1]
    device = inputs.device
    targets = F.one_hot(targets,nclasses).to(device)
    real = torch.sum(targets*inputs,dim=-1)
    other,_ = torch.max((1-targets)*inputs - targets*10000,dim=-1)
    loss = real - other
    loss = -loss
    return loss.mean()

class Adpt():
    """
    Adaptive attack
    """
    def __init__(self,epsilon=0.3,clip_min=0.,clip_max=1.,ord=torch.inf,target=False,scale=0.3) -> None:
        self.ord = ord
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.target = target
        self.scale = scale

    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)
        
    def loss_fn(self,latent_adv,y_pred,latent,y):
        dist_loss = nn.MSELoss()
        acc_loss = cw_loss
        
        return (1-self.scale)*acc_loss(y_pred,y)+self.scale*dist_loss(latent_adv,latent)
    
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
            _, pred = model(x)
        if len(pred.shape) == 1:
            pred = pred[None,:]
        pred = torch.argmax(pred,dim=1)
        if self.target:
            return pred == y
        else:
            return pred != y
    
    def adv_gen(self, model, x, target, iters = 100, alpha = 0.1):
        if alpha is None:
            alpha = 1/iters
        delta = torch.rand_like(x)*self.epsilon*2 - self.epsilon # uniform distribution [-eps, eps]
        xadv = self.clip(x + delta,x,self.ord)
        with torch.no_grad():
            latent,_ = model(x)

        for i in range(iters):
            idx_mask = self.is_adv(model,xadv,target)
            if torch.sum(idx_mask.float()) == xadv.shape[0]: # early stop
                break
            x_temp, latent_temp, xadv_temp, target_temp = x[~idx_mask], latent[~idx_mask], xadv[~idx_mask].detach().clone(), target[~idx_mask]
            xadv_temp.requires_grad = True
            latent_adv, y_pred = model(xadv_temp)
            if len(y_pred.shape) == 1:
                y_pred = y_pred[None,:]
            loss = self.loss_fn(latent_adv,y_pred,latent_temp,target_temp)
            if self.target:
                loss = -loss
            model.zero_grad()
            loss.backward()
            g = self.normalize(xadv_temp.grad.data,self.ord)
            xadv_temp = xadv_temp + alpha*g
            xadv_temp = self.clip(xadv_temp,x_temp,self.ord)
            xadv[~idx_mask] = xadv_temp
        return xadv
