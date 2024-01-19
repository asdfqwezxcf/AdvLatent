import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NES():
    """
    NES Gradient Estimate: https://arxiv.org/abs/1804.08598
    note that we use broadcasting for gradient estimation which increases the memory usage
    use a small batch size and sample size for memory limited devices
    """
    def __init__(self,epsilon=0.1,clip_min=None,clip_max=None,target=False,
                norm=torch.inf,sigma=0.001,samplesize=50,iterations=100,lr=0.01) -> None:
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.is_targeted = target
        self.norm = norm
        self.sigma = sigma
        self.samplesize_half = samplesize//2
        self.iterations = iterations
        self.lr = lr
        
    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)

    def adv_gen(self,model,x,y):
        x_adv = x # b, c, l, w
        for i in range(self.iterations):
            success = self.is_adv(model,x_adv,y)
            success = success.reshape(-1)
            # early stop
            if torch.sum(success.float()) == x_adv.shape[0]:
                break
            # mask success samples
            x_adv_temp = x_adv[~success]
            x_temp = x[~success]
            y_temp = y[~success]
            grad = self.gradiant_estimate(model,x_adv_temp,y_temp)
            
            if self.norm == torch.inf:
                x_adv_temp = x_adv_temp + self.lr*grad.sign()
                x_adv_temp = self.clip(x_temp,x_adv_temp)
                x_adv[~success] = x_adv_temp
            else:
                # TODO: add l2 norm
                raise NotImplementedError        
        return x_adv
    
    def is_adv(self,model,x,y):
        with torch.no_grad():
            pred = model(x)
        if len(pred.shape) == 1:
            pred = pred[None,:]
        pred = torch.argmax(pred,dim=1)
        if self.is_targeted:
            return pred == y
        else:
            return pred != y
        
    def clip(self,x,x_adv):
        if self.norm == torch.inf:
            if self.clip_min is not None:
                lb = torch.clamp(x-self.epsilon,min=self.clip_min) # lower bound
            else:
                lb = x-self.epsilon
            x_adv = torch.max(x_adv,lb)
            if self.clip_max is not None:
                ub = torch.clamp(x+self.epsilon,max=self.clip_max) # upper bound
            else:
                ub = x+self.epsilon
            x_adv = torch.min(x_adv,ub)
        else:
            # TODO: add clip on l2 ball
            raise NotImplementedError
        return x_adv

    def gradiant_estimate(self,model,x,y):
        batchsize = x.shape[0]
        device = x.get_device()
        c = x.shape[1] # channel 
        l = x.shape[2] # length
        w = x.shape[3] # width
        mu = torch.randn(batchsize,self.samplesize_half,c,l,w) # batchsize, samplesize/2, channel, length, width
        mu = torch.cat((-mu,mu),1) # batchsize, samplesize, channel, length, width
        if device >= 0:
            mu = mu.to('cuda:%d'%device)
        x = x[:,None] # expand dimension for broadcasting
        x_pert = x + self.sigma*mu # batchsize, samplesize, channel, length, width
        x_pert = x_pert.reshape(-1,c,l,w) # batch*sample, c, l, w
        with torch.no_grad():
            y_pred = model(x_pert)
        # gather P(y|x+sigma*mu)
        y_expand = y.repeat(self.samplesize_half*2,1).T.flatten() # batch*sample
        loss = torch.gather(y_pred,1,y_expand[:,None])
        if self.is_targeted:
            loss = -loss
        loss = loss.reshape(batchsize,-1,1,1,1) # batch, sample, 1, 1, 1
        g = -(loss*mu).mean(1)/self.sigma # b, c, l, w
        return g
    
class Nattack():
    """
    Nattack: https://arxiv.org/abs/1905.00441
    note that we use broadcasting for computing mu which increases the memory usage
    use a small batch size and sample size for memory limited devices
    """
    def __init__(self,epsilon=0.1,clip_min=None,clip_max=None,target=False,
                norm=torch.inf,sigma=0.1,lr=0.02,samplesize=100,iterations=50) -> None:
        """
        initialize the hyperparameter for nattack
        arguments:
            epsilon: perturbation budget
            clip_min: lower bound of adversarial samples
            clip_max: upper bound of adversarial samples
            target: specify targeted attack or not
            norm: specify l2 norm or linf norm for attacks
            sigma: standard deviation for the distribution
            lr: learning rate
            samplesize: for each input, we use a mini batch of perturbation to get the mu
                        note that this is different from the batchsize which discribe the number of input x
            iterations: the maximum number of iterations
        """
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.norm = norm
        self.sigma = sigma
        self.lr = lr
        self.samplesize = samplesize
        self.iterations = iterations
        self.is_targeted = target

    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)
    
    def cw_loss(self,inputs,targets,is_targeted=False):
        """
        nattack uses cw loss for optimization
        arguments:
            inputs: a tensor of shape [batchsize*samplesize, nclasses], y_pred
            targets: array, a batch of ground truth or target attack label
            is_targeted: boolean, targeted attack
        return:
            loss: array, a batch of cw loss
        """
        
        nclasses = inputs.shape[-1]
        device = inputs.device
        targets = targets.repeat(self.samplesize,1).T.flatten()
        targets = F.one_hot(targets,nclasses).to(device)
        real = torch.sum(targets*inputs,dim=-1)
        other,_ = torch.max((1-targets)*inputs - targets*10000,dim=-1)
        loss = real - other
        if is_targeted:
            loss = -loss
        return loss

    def clip(self,x,x_adv):
        if self.norm == torch.inf:
            if self.clip_min is not None:
                lb = torch.clamp(x-self.epsilon,min=self.clip_min) # lower bound
            else:
                lb = x-self.epsilon
            x_adv = torch.max(x_adv,lb)
            if self.clip_max is not None:
                ub = torch.clamp(x+self.epsilon,max=self.clip_max) # upper bound
            else:
                ub = x+self.epsilon
            x_adv = torch.min(x_adv,ub)
        else:
            # TODO: add clip on l2 ball
            raise NotImplementedError
        return x_adv
        
    def adv_gen(self,model,x,y,mu=None,shape=None):
        
        batchsize = x.shape[0]
        device = x.device
        # placeholder to store the best xadv and loss value
        x_adv = x
        loss_best = 1e6*(torch.ones_like(y).to(device))
        # by default, the perturbation lies in the space of cifar10 images (3x32x32)
        # for splited model, the perturbation lies in the latent space
        if shape is None:
            shape = (x.shape[1],x.shape[2],x.shape[3])
        # we use random initialization here if mu is not specified
        if mu is None:
            mu = torch.randn(batchsize,1,shape[0],shape[1],shape[2]) # batsh, 1, channel, length, width
        mu = mu.to(device)
        x = x[:,None,:] # unsqueeze x for broadcasting
        for i in range(self.iterations):
            # initialize a set of perturbation from standard normal distribution
            pert = torch.randn(batchsize,self.samplesize,shape[0],shape[1],shape[2]).to(device)
            # z is a linear transformation to make it follow the normal distribution
            z = mu + self.sigma*pert  
            if not pert.shape[-2:] == x.shape[-2:]:
                # bilinear interpolation
                g0_z = F.interpolate(z,x.shape[-2:],mode='bilinear',align_corners=False)
            else:
                # identity mapping
                g0_z = z

            # in the official implementation, there is an arctanh to convert input x to mu space
            # and tanh function 0.5*(torch.tanh(g0_z+arctanh_x)+1) to map the g_z to [0 1]
            # this is because for images there is explicit bound [0 1]
            # but for latent features there is no explicit min max value
            # thus we skip the tanh part and directly add perturbations to images
            # note that since we do not project perturbations
            # when iteration = 1, it becomes a gaussian random perturbation scenario
            x_adv_temp = x + g0_z
            x_adv_temp = self.clip(x,x_adv_temp)
            x_adv_temp = x_adv_temp.reshape(-1,x_adv.shape[1],x_adv.shape[2],x_adv.shape[3]) # batch*sample, channel, length and width
            with torch.no_grad():
                predict = model(x_adv_temp)
            # get loss
            loss = self.cw_loss(predict,y,self.is_targeted)
            loss = loss.reshape(batchsize,self.samplesize)
            # update best xadv and loss
            loss_best_temp,best_idx = torch.min(loss,dim=1)
            update = loss_best_temp < loss_best
            if torch.sum(update.float()) > 0:
                loss_best[update] = loss_best_temp[update]
                offset = torch.arange(0,x_adv_temp.shape[0],self.samplesize).to(device)
                best_idx += offset
                x_adv_temp = torch.index_select(x_adv_temp,0,best_idx)
                x_adv[update] = x_adv_temp[update]
            # early stop
            if torch.sum((loss_best>0).float()) == 0:
                break
            # compute z score
            reward = -0.5*loss
            z_score = (reward - torch.mean(reward,dim=1,keepdim=True))/(torch.std(reward,dim=1,keepdim=True) +1e-7) # batch, sample
            z_score = z_score.reshape(z_score.shape[0],z_score.shape[1],1,1,1) # b, s, 1, 1, 1

            # update mu
            mu = mu + self.lr*torch.mean(z_score*pert,dim=1,keepdim=True)/self.sigma

        return x_adv

class SquareAttack():
    """
    Square Attack: https://arxiv.org/abs/1912.00049
    note here we only have l-inf attack version
    for original implementation, refer to: https://github.com/max-andr/square-attack/tree/master
    """
    def __init__(self,epsilon=0.1,clip_min=None,clip_max=None,target=False,
                 iterations=10000, p_init=1.) -> None:
        
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.is_targeted = target
        self.iterations = iterations
        self.p_init = p_init

    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)
    
    def p_selection(self,it):
        """
        rescale p at different iterations
        """
        it = int(it / self.iterations * 10000)
        
        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it <= 10000:
            p = self.p_init / 512
        else:
            p = self.p_init
        
        return p
    
    def loss_fn(self,logits,targets):
        """
        cw loss (margin loss) for score-based attack
        logits: predicted scores, [batch, nclass]
        targets: ground truth labels, [batch,]
        """
        nclasses = logits.shape[-1]
        device = logits.device
        targets = F.one_hot(targets,nclasses).to(device)
        true = torch.sum(logits*targets,dim=1,keepdim=True) # score of corret class, batch, 1
        other,_ = torch.max((1-targets)*logits - targets*1e6,dim=1,keepdim=True) # score of second highest class
        loss = true - other # batch, 1
        if self.is_targeted:
            loss = -loss
        return loss

    def perturb(self,x,xadv,delta,c,h,w,s):
        anchor_h = np.random.randint(0, h-s)
        anchor_w = np.random.randint(0, w-s)
        delta[:,:,anchor_h:anchor_h+s,anchor_w:anchor_w+s] = self.epsilon*(torch.randint(0,2,(x.shape[0],c,1,1))*2-1).to(delta.device)
        
        xadv = x + delta
        if self.clip_min is not None and self.clip_max is not None:
            xadv = torch.clamp(xadv,self.clip_min,self.clip_max)
            delta = xadv - x
        
        return xadv, delta

    def adv_gen(self,model,x,y):
        b,c,h,w = x.shape
        device = x.device
        n_features = c*h*w
        # vertical init
        rand_select = torch.randint(0,2,(b,c,1,w)).to(device)
        delta_init = rand_select*self.epsilon + (rand_select-1)*self.epsilon # b,c,1,w
        x_adv = x + delta_init
        if self.clip_min is not None and self.clip_max is not None:
            x_adv = torch.clamp(x_adv,self.clip_min,self.clip_max) # b,c,h,w
        
        with torch.no_grad():
            logits = model(x_adv) # b, n
        loss = self.loss_fn(logits,y)

        for i_iter in range(self.iterations):
            idx_to_fool = (loss > 0).reshape(-1)
            if torch.sum(idx_to_fool.float()) == 0: # early stop
                break
            # place holder for update
            x_temp,x_adv_temp,y_temp = x[idx_to_fool],x_adv[idx_to_fool],y[idx_to_fool]
            delta = x_adv_temp - x_temp
            loss_temp = loss[idx_to_fool]
            # parameter for random search
            p = self.p_selection(i_iter)
            s = int(round(np.sqrt(p*n_features/c)))
            s = min(max(s,1),h-1)
            # generate perturbation
            x_adv_new,delta = self.perturb(x_temp,x_adv_temp,delta,c,h,w,s)
            with torch.no_grad():
                logits = model(x_adv_new)
            loss_new = self.loss_fn(logits,y_temp)
            # update loss and xadv
            idx_improved = (loss_new < loss_temp) # b, 1
            loss[idx_to_fool] = (idx_improved.float())*loss_new + (1-idx_improved.float())*loss_temp
            idx_improved = idx_improved.reshape(-1,1,1,1) # for broadcasting
            x_adv[idx_to_fool] = (idx_improved.float())*x_adv_new + (1-idx_improved.float())*x_adv_temp

        return x_adv

