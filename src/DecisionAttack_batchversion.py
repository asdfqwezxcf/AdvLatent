import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_dct
from multiprocessing import Pool, cpu_count

class SignOPT():
    """
    Sign-OPT: https://arxiv.org/abs/1909.10773
    """
    def __init__(self,epsilon=0.1,clip_min=None,clip_max=None,target=False,
                 iterations=20,nsamples=200,alpha=0.2,sigma=0.001,momentum=0.0) -> None:
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.is_targeted = target
        self.iterations = iterations
        self.nsamples = nsamples
        self.alpha_init = alpha
        self.sigma_init = sigma
        self.momentum = momentum

    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)

    def is_adv(self,model,x_adv,y):
        with torch.no_grad():
            pred = model(x_adv)
        if len(pred.shape) == 1:
            pred = pred[None,:]
        pred = torch.argmax(pred,dim=1)
        if self.is_targeted:
            return pred == y
        else:
            return pred != y

    def init_adv_batch(self,model,x,y):
        """
        initialize adversary for multiple instance within finite loops
        """
        theta = torch.randn(x.shape).to(x.device)
        xadv = x + theta # n, c, l, w
        mask = self.is_adv(model,xadv,y)  # n
        for i in range(99):
            if torch.sum(mask) == x.shape[0]:
                break
            theta[~mask] = torch.randn((int(torch.sum(~mask)),*x.shape[1:])).to(x.device)
            xadv = x + theta
            mask = self.is_adv(model,xadv,y)
        
        theta_norm = torch.norm(theta,p=2,dim=(1,2,3),keepdim=True) # m, 1, 1, 1
        theta /= theta_norm # m, c, l, w
        theta_g = self.bin_search_batch(model,x,y,theta,theta_norm) # m
        
        return theta, theta_g, mask

    def bin_search_batch(self,model,x,y,theta,initial_bound):
        """
        binary search processing in batch to find the minimum update
        the implementation is following the origin code
        which is kind of different from the boundary search in HSJA
        """ 
        highs = initial_bound # m, 1, 1, 1
        lows = torch.zeros_like(highs) # m, 1, 1, 1
        shape = tuple(highs.shape)
        while torch.max((highs - lows)/1e-3) > 1:
            lows = lows.reshape(shape) # m, 1, 1, 1
            highs = highs.reshape(shape) # m, 1, 1, 1
            mids = (highs+lows)/2
            blended = x + mids*theta
            sucess = self.is_adv(model,blended,y) # m
            lows = lows.reshape(-1)
            highs = highs.reshape(-1)
            mids = mids.reshape(-1)
            sucess = sucess.float()
            lows = lows*sucess + mids*(1-sucess)
            highs = highs*(1-sucess) + mids*sucess # m
        return highs

    def grad_sign_est(self,model,x,y,theta,initial_bound,sigma):
        """
        take one instance at a time and process in batch for acceleration
        """
        shape = [self.nsamples] + list(x.shape[1:]) # n, c, l, w
        
        u = torch.randn(shape).to(x.device)
        norm_u = torch.norm(u,p=2,dim=(1,2,3),keepdim=True)
        u /= norm_u

        theta_new = theta + sigma*u # broadcasting 1, c, l, w --> n, c, l, w
        theta_norm = torch.norm(theta_new,p=2,dim=(1,2,3),keepdim=True)
        theta_new /= theta_norm
        
        x_adv = x + initial_bound*theta_new
        sign = -1*(self.is_adv(model,x_adv,y)).float().to(x.device) # n
        if self.is_targeted:
            sign = - sign
        sign = sign.view(-1,1,1,1) # expand dim for broadcasting

        sign_grad = (u*sign).mean(dim=0,keepdim=True) # 1, c, l, w
        
        return sign_grad

    def bin_search_local(self,model,x,y,theta,init_bound,threshold):
        xadv = x + theta*init_bound
        # init_bound is a upper bound, find lower bound
        if self.is_adv(model,xadv,y):
            high = init_bound
            low = init_bound*0.99
            xadv = x+theta*low
            while self.is_adv(model,xadv,y):
                low *= 0.99
                xadv = x+theta*low
        # init_bound is a lower bound, find upper bound
        else:
            low = init_bound
            high = init_bound*1.01
            xadv = x+theta*high
            while not self.is_adv(model,xadv,y):
                high *= 1.01
                xadv = x+theta*high
        # do bin search to find proper boundary
        while (high - low) > threshold:
            mid = (high + low) / 2
            xadv = x + theta*mid
            if self.is_adv(model,xadv,y):
                high = mid
            else:
                low = mid
        return high

    def sign_opt(self,model,x,y):
        d = int(np.prod([*x.shape[1:]])) # c * w * h
        theta, theta_g, mask = self.init_adv_batch(model,x,y)
        xadv = x + theta * theta_g # initialize xadv

        xadv_update, x_update, y_update = xadv[mask], x[mask], y[mask]
        theta_update, theta_g_update = theta[mask], theta_g[mask].reshape(-1)

        alpha_update = np.repeat(self.alpha_init,xadv_update.shape[0])
        sigma_update = np.repeat(self.sigma_init,xadv_update.shape[0])

        for i in range(self.iterations):
            # early stop
            within_budget = theta_g_update <= np.sqrt(self.epsilon*d)
            if torch.sum(within_budget.float()) == xadv_update.shape[0]:
                break

            xadv_temp, x_temp, y_temp = xadv_update[~within_budget], x_update[~within_budget], y_update[~within_budget]
            theta_temp, theta_g_temp = theta_update[~within_budget], theta_g_update[~within_budget]
            alpha_temp, sigma_temp = alpha_update[~within_budget], sigma_update[~within_budget]
            sign_grad = self.grad_sign_est(model,x_temp,y_temp,theta_temp,theta_g_temp,sigma_temp) # b, c, l, w
            min_theta, min_theta_g = theta_temp, theta_g_temp
            # increasing grid search for alpha
            for _ in range(10):
                theta_new = theta_temp - alpha * sign_grad
                theta_new /= torch.norm(theta_new)
                theta_g_new = self.bin_search_local(model,x_temp,y_temp,theta_new,min_theta_g,sigma_temp)
                alpha_temp *= 2
                new_mask = theta_g_new < min_theta_g
                min_theta[new_mask] = theta_new[new_mask]
                min_theta_g[new_mask] = theta_g_new[new_mask]
                xadv_temp = x_temp + min_theta *min_theta_g
                if temp_theta_g < min_theta_g:
                    min_theta = temp_theta
                    min_theta_g = temp_theta_g
                    xadv = x + min_theta * min_theta_g # if successfully find min perturbation, update xadv
                else:
                    break
            # decreasing grid search for alpha
            if min_theta_g >= update_theta_g:
                for _ in range(10):
                    alpha *= 0.25
                    temp_theta = update_theta - alpha * sign_grad
                    temp_theta /= torch.norm(temp_theta)
                    temp_theta_g = self.bin_search_local(model,x,y,temp_theta,min_theta_g,sigma)
                    if temp_theta_g < update_theta_g:
                        min_theta = temp_theta
                        min_theta_g = temp_theta_g
                        xadv = x + min_theta * min_theta_g # if successfully find min perturbation, update xadv
                        break
            # early stop should be added here
            distance = torch.norm(x-xadv)
            if self.is_adv(model,xadv,y) and distance**2/d <= self.epsilon:
                break
            # if all failed, reset alpha and decrease sigma
            if alpha <= 1e-4:
                # print('no moving')
                alpha = self.alpha_init
                sigma *= 0.1
                if sigma < 1e-5:
                    break
            # update distortion and distance
            update_theta, update_theta_g = min_theta, min_theta_g

        if distance**2/d > self.epsilon:
            # print('failed to find adversary within budget.')
            xadv = x
        
        else:
            xadv = x

        return xadv

    def adv_gen(self,model,x,y):
        x_advs = []
        for i, xi in enumerate(x):
            x_adv = self.sign_opt(model,xi[None,:],y[i])
            x_advs.append(x_adv)
        x_advs = torch.cat(x_advs,0)
        return x_advs
    
def get_mask(args):
    w,x,y,k = args
    select = np.random.choice(len(x),k,replace=False)
    mask = torch.zeros(w,w)
    mask[x[select],y[select]] = 1
    return mask

class TriAttack():
    """
    Triangle Attack: https://arxiv.org/abs/2112.06569
    """
    def __init__(self,epsilon=0.1,clip_min=None,clip_max=None,target=None,iterations=100,
                 dim_num=5,ratio_mask=0.1) -> None:
        
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.is_targeted = target
        self.iterations = iterations
        self.dim_num = dim_num
        self.ratio_mask = ratio_mask

        try:
            workers = cpu_count()
        except NotImplementedError:
            workers = 1
        self.pool = Pool(workers)

    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)
    
    def is_adv(self,model,x_adv,y):
        with torch.no_grad():
            pred = model(x_adv)
        if len(pred.shape) == 1:
            pred = pred[None,:]
        pred = torch.argmax(pred,dim=1)
        if self.is_targeted:
            return pred == y
        else:
            return pred != y
        
    def init_adv(self,model,x,y):
        # find random adverary
        x_init = torch.rand(x.size()).to(x.device)
        is_adv = self.is_adv(model,x_init,y)
        for i in range(99):
            if torch.sum(is_adv) == x.shape[0]:
                break
            x_init[~is_adv] = torch.rand(x[~is_adv].size()).to(x.device)
            is_adv = self.is_adv(model,x_init,y)
        # find boundary
        low = torch.zeros(torch.sum(is_adv)).to(x.device)
        high = torch.ones(torch.sum(is_adv)).to(x.device)
        while torch.max((high-low)/1e-3) > 1:
            # TODO: add linf search
            low = low.reshape(-1,1,1,1)
            high = high.reshape(-1,1,1,1)
            mid = (high + low) / 2.0
            blended = (1 - mid) * x[is_adv] + mid * x_init[is_adv]
            success = self.is_adv(model,blended,y[is_adv])
            high = high.reshape(-1)
            low = low.reshape(-1)
            mid = mid.reshape(-1)
            low = low*success.float() + mid*(1-success.float())
            high = high*(1-success.float()) + mid*success.float()
        # update x_init
        x_init[is_adv] = (1 - high.reshape(-1,1,1,1)) * x[is_adv] + high.reshape(-1,1,1,1) * x_init[is_adv]
        return x_init, is_adv

    def rotate(self,difference,direction,theta):
        """
        difference: xadv - x, shape: b, c, h, w
        direction: low frequency mask, shape: b, c, h, w
        theta: angle
        """
        alpha = (torch.sum(difference*direction,dim=(1,2,3),keepdim=True) / 
                 torch.sum(difference**2,dim=(1,2,3),keepdim=True)) # b, 1, 1, 1
        orthogonal = direction - alpha * difference # b, c, h, w
        direction_theta = (difference * torch.cos(theta) + 
                           torch.norm(difference,p=2,dim=(1,2,3),keepdim=True) / 
                           torch.norm(orthogonal,p=2,dim=(1,2,3),keepdim=True) *
                           orthogonal * torch.sin(theta))
        direction_theta = (direction_theta / torch.norm(direction_theta,p=2,dim=(1,2,3),keepdim=True) * 
                           torch.norm(difference,p=2,dim=(1,2,3),keepdim=True))
        return direction_theta
    
    def get_orthogonal_1d_in_subspace(self,difference,is_left=True):
        b,c,w,h = difference.shape
        side_length = torch.minimum(w,h)
        zero_mask = torch.zeros(side_length,side_length)
        size_mask = int(side_length*self.ratio_mask)
        if is_left:
            zero_mask[:size_mask,:size_mask] = 1
        else:
            zero_mask[-size_mask:,-size_mask:,] = 1
        to_choose = torch.where(zero_mask == 1)
        x = to_choose[0]
        y = to_choose[1]

        args = []
        for i in range(b*c):
            args.append([side_length,x,y,self.dim_num])
        mask = self.pool.map(get_mask,args)
        mask = torch.Tensor(torch.cat(mask,dim=0)).reshape(difference.shape).to(difference.device)
        mask *= torch.randn_like(mask)
        direction = self.rotate(difference,mask,np.pi/2)
        orthogonal = (direction / torch.norm(direction,p=2,dim=(1,2,3),keepdim=True) * 
                      torch.norm(difference,p=2,dim=(1,2,3),keepdim=True))
        return orthogonal
    
    def get_xadv_2d(self,model,x,xadv,y,axis1,axis2,init_alpha,
                    max_iter=2,plus_lr=0.01,minus_lr=0.005,half_range=0.1):
        alpha = init_alpha # an array of shape b,1,1,1
        upper = np.pi/2+half_range
        lower = np.pi/2-half_range

        d = torch.norm(x-xadv,p=2,dim=(1,2,3),keepdim=True)
        pi = torch.tensor(np.repeat(np.pi,alpha.shape[0])).reshape(-1,1,1,1).to(alpha.device)
        theta = torch.maximum(pi-2*alpha,0)+torch.minimum(pi/16,alpha/2)

        x_hat = torch_dct.idct_2d(xadv)
        right_theta = pi - alpha
        flag = torch.zeros_like(right_theta)

        x_temp = (x + d*(axis1*torch.cos(theta)+
                         axis2*torch.sin(theta))/
                  torch.sin(alpha)*torch.sin(alpha+theta))
        x_temp = torch_dct.idct_2d(x_temp)
        if self.clip_min is not None and self.clip_max is not None:
            x_temp = torch.clamp(x_temp,self.clip_min,self.clip_max)
        with torch.no_grad():
            pred = model(x_temp)
        if len(pred.shape) == 1:
            pred = pred[None,:]
        label = torch.argmax(pred,dim=1)
        
        success = (label != y)
        # if label != original_label:
        flag[success] = 1
        # else:
        alpha[~success] -= minus_lr
        alpha = torch.clamp(alpha,min=lower)
        theta[~success] = torch.maximum(theta[~success], np.pi-2*alpha[~success]+np.pi/64)
        
        x_temp[~success] = (x[~success] + d[~success]*(axis1[~success]*torch.cos(theta[~success])-
                                                       axis2*torch.sin(theta[~success]))/
                            torch.sin(alpha[~success])*torch.sin(alpha[~success]+theta[~success]))
        x_temp[~success] = torch_dct.idct_2d(x_temp[~success])
        if self.clip_min is not None and self.clip_max is not None:
            x_temp = torch.clamp(x_temp,self.clip_min,self.clip_max)
        with torch.no_grad():
            pred = model(x_temp)
        if len(pred.shape) == 1:
            pred = pred[None,:]
        label = torch.argmax(pred,dim=1)
        # if label != orignal_label:
        success_2 = (label != y)
        flag[torch.logical_xor(success_2,success)] = -1
        # else:
        alpha[~success_2] -= minus_lr
        alpha = torch.clamp(alpha,min=lower)

        # perform binary search (only for succeed samples)
        left_theta = theta[success_2]
        right_theta = right_theta[success_2]
        theta_update = (left_theta + right_theta) / 2
        xhat_update, x_update, y_update = x_temp[success_2], x[success_2], y[success_2]
        d_update, axis1_update, axis2_update = d[success_2],axis1[success_2],axis2[success_2]
        alpha_update, flag_update = alpha[success_2], flag[success_2]
        for i in range(max_iter):
            x_temp = (x_update + d_update*(axis1_update*torch.cos(theta_update)+
                                           flag_update*axis2_update*torch.sin(theta_update))/
                      torch.sin(alpha_update)*torch.sin(alpha_update+theta_update))
            x_temp = torch_dct.idct_2d(x_temp)
            if self.clip_min is not None and self.clip_max is not None:
                x_temp = torch.clamp(x_temp,self.clip_min,self.clip_max)
            with torch.no_grad():
                pred = model(x_temp)
            if len(pred.shape) == 1:
                pred = pred[None,:]
            label = torch.argmax(pred,dim=1)
            # if label != original_label:
            success_update = label != y_update
            # else:
            alpha_update[~success_update] -= minus_lr
            alpha_update = torch.clamp(alpha_update,min=lower)
            theta_update[~success_update] = torch.maximum(theta_update[~success_update],
                                                          np.pi-2*alpha_update[~success_update]+np.pi/64)
            flag_update[~success_update] = -flag_update[~success_update]

            x_temp[~success_update] = (x_update[~success_update]+
                                       d_update[~success_update]*(axis1_update[~success_update]*torch.cos(theta_update[~success_update])+
                                                                  flag_update[~success_update]*torch.sin(theta_update[~success_update]))/
                                       torch.sin(alpha_update[~success_update])*torch.sin(alpha_update[~success_update]+theta_update[~success_update]))
            x_temp[~success_update] = torch_dct.idct_2d(x_temp[~success_update])
            if self.clip_min is not None and self.clip_max is not None:
                x_temp = torch.clamp(x_temp,self.clip_min,self.clip_max)
            with torch.no_grad():
                pred = model(x_temp)
            if len(pred.shape) == 1:
                pred = pred[None,:]
            label = torch.argmax(pred,dim=1)
            success2_update = (label != y_update)
            # if label != original_labels:
            xhat_update[success2_update] = x_temp[success2_update]
            alpha_update[success2_update] += plus_lr
            # else:
            alpha_update[~success2_update] -= minus_lr
            alpha_update = torch.clamp(alpha_update,lower,upper)

            left_theta = (torch.clamp(np.pi-2*alpha_update[~success2_update],min=0)+
                          torch.clamp(alpha_update[~success2_update],max=np.pi/16))
            right_theta = theta_update[~success2_update]

            # update adv and alpha
            x_hat[success_2] = xhat_update
            alpha[success_2] = alpha_update
            if torch.sum(~success2_update) == 0:
                break # early stop
            # update for next round
            xhat_update, x_update, y_update = xhat_update[~success2_update],x_update[~success2_update],y_update[~success2_update]
            theta_update = (left_theta + right_theta) / 2
            d_update, axis1_update, axis2_update = d_update[~success2_update], axis1_update[~success2_update], axis2_update[~success2_update]
            alpha_update, flag_update = alpha_update[~success2_update], flag_update[~success2_update]

        alpha += plus_lr
        alpha = torch.clamp(alpha,max=upper)
        return x_hat, alpha
        

    def adv_gen(self,model,x,y):
        d = np.prod([*x.shape[1:]])
        x_adv = self.init_adv(model,x,y)
        dist = torch.norm(x-x_adv,p=2,dim=(1,2,3))
        alpha = torch.tensor(np.repeat(np.pi/2,x.shape[0])).reshape(-1,1,1,1).to(x.device)
        for i in range(self.iterations):
            within_budget = dist<=np.sqrt(self.epsilon*d)
            if torch.sum(within_budget) == x.shape[0]:
                break
            x_adv_temp, x_temp, y_temp = x_adv[~within_budget], x[~within_budget], y[~within_budget]
            difference = torch.dct_2d(x_adv_temp-x_temp)
            axis_1 = difference / torch.norm(difference,p=2,dim=(1,2,3),keepdim=True)
            direction = self.get_orthogonal_1d_in_subspace(difference)
            axis_2 = direction / torch.norm(direction,p=2,dim=(1,2,3),keepdim=True)
            x_adv_temp,alpha = self.get_xadv_2d(model,x_temp,x_adv_temp,y_temp,axis_1,axis_2,alpha)
            dist_temp = torch.norm(x_temp-x_adv_temp,p=2,dim=(1,2,3))
            x_adv[~within_budget] = x_adv_temp
            dist[~within_budget] = dist_temp
        
        return x_adv
