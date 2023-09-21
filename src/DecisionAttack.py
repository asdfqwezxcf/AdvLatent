import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count
import torch_dct
import numpy as np

def get_sample_idx(args):
    m,k,select_prob,shape = args
    select_indices = np.random.choice(m, k, replace=False, p=select_prob)
    factor = np.zeros([m])
    factor[select_indices] = 1
    return factor.reshape(shape)

class Evolutionary():
    """
    Evolutionary Attack: https://arxiv.org/abs/1904.04433
    this implementation is based on: 
    https://github.com/thu-ml/ares/blob/main/pytorch_ares/pytorch_ares/attack_torch/evolutionary.py
    to accelerate the process, we adopt the original implementation to batch operations
    it now could process multiple instance in parallel
    """
    def __init__(self,epsilon=0.1,clip_min=None,clip_max=None,target=False,iterations=1000,
                sigma=0.03,mu=0.1,cc=0.01,ccov=0.001,max_len=30,sub=2) -> None:
        self.epsilon=epsilon
        self.clip_min=clip_min
        self.clip_max=clip_max
        self.is_targeted=target
        self.iterations=iterations
        self.sigma=sigma
        self.mu=mu
        self.cc=cc
        self.ccov=ccov
        self.maxlen=max_len
        self.sub=sub

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

    def init_adv_batch(self,model,x,y):
        """
        initialize adversary within finite loops
        """
        x_init = torch.rand(x.size()).to(x.device)
        success = self.is_adv(model,x_init,y)
        for i in range(99):
            if torch.sum(success.float()) == x.shape[0]:
                break # early break
            x_temp = x[~success]
            x_init[~success] = torch.rand(x_temp.size()).to(x.device)
            success = self.is_adv(model,x_init,y)
        
        return x_init, success
            
    def evolutionary_batch(self,model,x,y):
        """
        note that in original implementation it search on a smaller dimension 
        and use bilinear upsampling to map to image domain
        but here we directly search on the input space 
        beacuse the split bottleneck model has a way smaller dimension on the latent space
        """
        b,c,w,l = x.shape
        mindist = 1e10*torch.ones(b).to(x.device)
        pert_shape = (c,w//self.sub,l//self.sub) # perturbation on subspace
        input_shape = (c,w,l)
        m = np.prod(pert_shape)
        k = int(m / 20)
        xadv, init_success = self.init_adv_batch(model,x,y)
        best_xadv = xadv
        # updatable buffer
        xadv_update, x_update, y_update = xadv[init_success], x[init_success], y[init_success]
        mindist_update, best_xadv_update = mindist[init_success], best_xadv[init_success]
        
        n_success = int(torch.sum(init_success))
        stats_update = np.zeros(n_success)
        stats_counter = np.zeros(n_success)
        evo_path_update = np.zeros((n_success, *pert_shape))
        diag_cov_update = np.ones((n_success, *pert_shape))
        mu_update = np.repeat(self.mu,n_success)

        for i in range(self.iterations):
            # compute distance for the previous round
            distance = x_update - xadv_update
            dist_norm = torch.norm(distance,p=2,dim=(1,2,3))
            sigma = 0.01*dist_norm # adaptive sigma
            # update minimum distance and corresponding xadv
            idx_update = (dist_norm < mindist_update).reshape(-1)
            mindist_update[idx_update] = dist_norm[idx_update]
            best_xadv_update[idx_update] = xadv_update[idx_update]
            # check if the perturbation budget is satisfied
            within_budget = (mindist_update <= np.sqrt(self.epsilon*m)).reshape(-1)
            if torch.sum(within_budget.float()) == xadv_update.shape[0]:
                break
            # copy samples that didnt satisfy distortion constraint to temp buffer
            mask_temp = ~within_budget
            mask_temp_cpu = mask_temp.to('cpu')
            xadv_temp, x_temp, y_temp = xadv_update[mask_temp], x_update[mask_temp], y_update[mask_temp]
            diag_cov_temp = diag_cov_update[mask_temp_cpu]
            evo_path_temp = evo_path_update[mask_temp_cpu]
            nsamples_temp = xadv_temp.shape[0]
            stats_temp = stats_update[mask_temp_cpu]
            stats_counter_temp = stats_counter[mask_temp_cpu]
            mu_temp = mu_update[mask_temp_cpu]
            sigma = sigma[mask_temp].reshape(-1,1,1,1)
            sigma_cpu = sigma.cpu().numpy()
            # find perturbation
            select_probs = diag_cov_temp.reshape(nsamples_temp,m) / np.sum(diag_cov_temp,axis=(1,2,3)).reshape(nsamples_temp,1)
            
            args = []
            for select_prob in select_probs:
                args.append([m,k,select_prob,pert_shape])
            factors = self.pool.map(get_sample_idx,args) # multiprocessing for acceleration
            # factors = []
            # for select_prob in select_probs:
            #     select_indices = np.random.choice(m, k, replace=False, p=select_prob)
            #     factor = np.zeros([m])
            #     factor[select_indices] = 1
            #     factors.append(factor.reshape(pert_shape))
            factors = torch.Tensor(np.stack(factors,axis=0)).to(x.device)
            pert = np.random.normal(0.0, 1.0, (nsamples_temp,*pert_shape))*np.sqrt(diag_cov_temp)*sigma_cpu # sample from N(0,sigma^2*C)
            pert_large = torch.Tensor(pert).to(x.device)
            pert_large = pert_large*factors*torch.Tensor(np.sqrt(diag_cov_temp)).to(x.device)
            # upscale to input space
            pert_large = F.interpolate(pert_large,input_shape[1:],mode='bilinear')
            mu_large = torch.Tensor(mu_temp).to(xadv_temp.device).reshape(-1,1,1,1)
            biased = xadv_temp+mu_large*distance[mask_temp]
            candidate = biased+sigma*dist_norm[mask_temp].reshape(nsamples_temp,1,1,1)*pert_large/torch.norm(pert_large,p=2,dim=(1,2,3),keepdim=True)
            candidate = x_temp-(x_temp-candidate)/torch.norm(x_temp-candidate,p=2,dim=(1,2,3),keepdim=True)*torch.norm(x_temp-biased,p=2,dim=(1,2,3),keepdim=True)
            if self.clip_min is not None and self.clip_max is not None:
                candidate = torch.clamp(candidate, self.clip_min, self.clip_max)
            is_adv = self.is_adv(model,candidate,y_temp)
            is_adv_cpu = is_adv.to('cpu')
            xadv_temp[is_adv] = candidate[is_adv]
            evo_path_temp[is_adv_cpu] = (1-self.cc)*evo_path_temp[is_adv_cpu]+np.sqrt((2*self.cc)-(self.cc**2))*pert[is_adv_cpu]/sigma_cpu[is_adv_cpu]
            diag_cov_temp[is_adv_cpu] = (1-self.ccov)*diag_cov_temp[is_adv_cpu]+self.ccov*(evo_path_temp[is_adv_cpu]**2)
            stats_temp[is_adv_cpu] += 1
            stats_counter_temp += 1
            update_mu_mask = stats_counter_temp==self.maxlen
            mu_temp[update_mu_mask] *= np.exp(stats_temp[update_mu_mask]/self.maxlen - 0.2)

            # copy values from temp to the update buffer for next round
            mu_update[mask_temp_cpu] = mu_temp
            stats_counter[mask_temp_cpu] = stats_counter_temp
            stats_update[mask_temp_cpu] = stats_temp
            evo_path_update[mask_temp_cpu] = evo_path_temp
            diag_cov_update[mask_temp_cpu] = diag_cov_temp
            xadv_update[mask_temp] = xadv_temp
            
        # copy data from update buffer to origin
        mindist[init_success], best_xadv[init_success] = mindist_update, best_xadv_update
        mask = mindist > np.sqrt(self.epsilon*m)
        best_xadv[mask] = x[mask]

        return best_xadv

    def adv_gen(self,model,x,y):
        x_advs = self.evolutionary_batch(model,x,y)
        return x_advs

class HSJA():
    """
    HopSkipJumpAttack: https://arxiv.org/abs/1904.02144
    it process multiple instance at a time
    """
    def __init__(self,epsilon=0.1,clip_min=None,clip_max=None,target=False,iterations=20,
                norm=2,gamma=1.,num_init_est=50,num_max_est=200) -> None:
        self.epsilon=epsilon
        self.clip_min=clip_min
        self.clip_max=clip_max
        self.is_targeted=target
        self.iterations=iterations
        self.norm=norm
        self.gamma=gamma
        self.num_init_est = num_init_est
        self.num_max_est = num_max_est

    def __call__(self, *args, **kwds):
        return self.adv_gen(*args, **kwds)

    def init_adv(self,model,x,y):
        """
        initialize adversary within finite loops
        """
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

    def compute_distance(self,x,xadv):
        if self.norm == 2:
            dist = torch.norm(x-xadv,p=2,dim=(1,2,3))
        else:
            dist = torch.max(torch.abs(x-xadv),dim=(1,2,3))
        return dist

    def gradient_search(self,model,xadv,y,num_est,delta):
        noise_shape = (num_est,*xadv.shape) # n, b, c, w, l
        y = y[None,:].repeat(num_est,1) # n, b
        if self.norm == 2:
            rv = torch.randn(noise_shape)
        else:
            rv = -1 + torch.rand(noise_shape) * 2

        axis = (2,3,4) # c, w, l
        rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=axis, keepdim=True))
        rv = rv.to(xadv.device) 
        if isinstance(delta,torch.Tensor):
            delta = delta.reshape(1,-1,1,1,1)
        perturbed = xadv[None,:] + delta * rv
        if self.clip_min is not None and self.clip_max is not None:
            perturbed = torch.clamp(perturbed, self.clip_min, self.clip_max)
        rv = (perturbed - xadv[None,:]) / delta # n, b, c, w, l

        # query the model.
        perturbed = perturbed.reshape(-1,*noise_shape[2:])
        decisions = self.is_adv(model,perturbed,y.reshape(-1)).float()
        fval = 2.0 * decisions - 1.0
        fval = fval.reshape(num_est, -1) # n, b
        # Baseline subtraction (when fval differs)
        fval_mean = torch.mean(fval,dim=0,keepdim=True) # 1, b
        fval = fval - fval_mean # n, b
        mask1 = fval_mean.reshape(-1) == 1
        mask2 = fval_mean.reshape(-1) == -1
        gradf = torch.mean(fval.reshape(*fval.shape,1,1,1)*rv, dim=0) # b, c, w, l
        gradf[mask1] = torch.mean(rv,dim=0)[mask1]
        gradf[mask2] = torch.mean(rv,dim=0)[mask2]
        
        # Get the gradient direction.
        gradf = gradf / torch.norm(gradf,p=2,dim=(1,2,3),keepdim=True)
        
        return gradf
    
    def boundary_search(self,model,x,xadv,y,theta):
        # Set binary search parameters.
        distance = self.compute_distance(x,xadv)
        if self.norm == 2:
            high = torch.ones(x.shape[0]).to(x.device)
            threshold = torch.tensor(theta).repeat_interleave(x.shape[0]).to(x.device)
        else:
            high = distance
            threshold = torch.minimum(distance*theta,torch.tensor(theta).repeat_interleave(x.shape[0]).to(x.device))
        threshold = torch.where(threshold>1e-5,threshold,1e-5) # the original threshold is too small, the while loop will loop forever

        low = torch.zeros(x.shape[0]).to(x.device)
        while torch.max((high-low)/threshold) > 1:
            # TODO: add linf search
            mid = (high + low) / 2
            mid = mid.reshape(-1,1,1,1) # expand dim for broadcast
            blended = (1 - mid) * x + mid * xadv
            success = self.is_adv(model,blended,y)
            success = success.float()
            low = low*success + mid.reshape(-1)*(1-success)
            high = high*(1-success) + mid.reshape(-1)*success
        high = high.reshape(-1,1,1,1)
        xadv = (1 - high) * x + high * xadv
        return xadv
  
    def stepsize_search(self,model,xadv,grad,y,dist,iter):
        step = dist / np.sqrt(iter+1) # b
        
        while True:
            update = xadv + step.reshape(-1,1,1,1) * grad
            if self.clip_min is not None and self.clip_max is not None:
                update = torch.clamp(update,self.clip_min,self.clip_max)
            success = self.is_adv(model,update,y)
            if torch.sum(success.float()) == xadv.shape[0]:
                break
            step[~success] /= 2.0
            
        return step.reshape(-1,1,1,1)
    
    def hsja(self,model,x,y):
        device = x.get_device()
        b = x.shape[0]
        mindist = 1e10*torch.ones(b).to(device)
        d = int(np.prod([*x.shape[1:]]))
        
        if self.is_targeted:
            xadv = x
            init_success = torch.ones(b,dtype=torch.bool).to(device)
        else:
            xadv,init_success = self.init_adv(model,x,y)
        xadv_best = xadv
        # update buffer
        xadv_update, x_update, y_update = xadv[init_success], x[init_success], y[init_success]
        mindist_update, xadv_best_update = mindist[init_success], xadv_best[init_success]
        
        # initialize theta
        if self.norm == 2:
            theta = self.gamma / (np.sqrt(d) * d)
        else:
            theta = self.gamma / (d ** 2)
            

        for n_iter in range(self.iterations):
            # update previous round 
            dist_update = self.compute_distance(x_update,xadv_update)
            mindist_update[dist_update<mindist_update] = dist_update[dist_update<mindist_update]
            xadv_best_update[dist_update<mindist_update] = xadv_update[dist_update<mindist_update]
            # early stop
            if self.norm == 2:
                within_budget = mindist_update<=np.sqrt(self.epsilon*d)
            else:
                within_budget = mindist_update<=self.epsilon
            if torch.sum(within_budget.float()) == xadv_update.shape[0]:
                break
            # create temp buffer for next round
            xadv_temp, x_temp, y_temp = xadv_update[~within_budget], x_update[~within_budget], y_update[~within_budget]
            # search for boundary
            xadv_temp = self.boundary_search(model,x_temp,xadv_temp,y_temp,theta)
            # set up parameters
            if n_iter == 0:
                delta = 0.1
            else:
                if self.norm == 2:
                    delta = np.sqrt(d) * theta * dist_update[~within_budget]
                else:
                    delta = d * theta * dist_update[~within_budget] # delta is either a number or an array
            # choose number of gradient estimation steps
            num_est = int(min(self.num_init_est*np.sqrt(n_iter+1),self.num_max_est))
            # gradient estimation
            grad = self.gradient_search(model,xadv_temp,y_temp,num_est,delta)
            if self.norm == torch.inf:
                grad = torch.sign(grad)
            # stepsize search
            stepsize = self.stepsize_search(model,xadv_temp,grad,y_temp,dist_update[~within_budget],n_iter)
            # update sample
            xadv_temp = xadv_temp + stepsize*grad
            if self.clip_min is not None and self.clip_max is not None:
                xadv_temp = torch.clamp(xadv_temp,self.clip_min,self.clip_max)
            # update buffer for next round
            xadv_update[~within_budget] = xadv_temp

        xadv_best[init_success] = xadv_best_update
        mindist[init_success] = mindist_update
        mask = mindist > np.sqrt(self.epsilon*d)
        if self.norm == torch.inf:
            mask = mindist > self.epsilon
        xadv_best[mask] = x[mask]

        return xadv_best

    def adv_gen(self,model,x,y):
        x_advs = self.hsja(model,x,y)
        return x_advs

class SignOPT():
    """
    Sign-OPT: https://arxiv.org/abs/1909.10773
    """
    def __init__(self,epsilon=0.1,clip_min=None,clip_max=None,target=False,
                 iterations=50,nsamples=200,alpha=0.2,sigma=0.001,momentum=0.0) -> None:
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

    def init_adv_batch(self,model,x,y,num_init=100):
        """
        take one instance at a time and process initial distortion in batch for acceleration
        """
        shape = [num_init] + list(x.shape[1:]) # n, c, l, w
        theta = torch.randn(shape).to(x.device)
        xadv = x + theta # n, c, l, w
        mask = self.is_adv(model,xadv,y)  # n
        theta = theta[mask] # m, c, l, w
        if len(theta) == 0:
            success = False
            best_theta = torch.zeros_like(x) 
            best_theta_g = 1e10
        else:
            theta_norm = torch.norm(theta,p=2,dim=(1,2,3),keepdim=True) # m, 1, 1, 1
            theta /= theta_norm # m, c, l, w
            theta_g = self.bin_search_batch(model,x,y,theta,theta_norm) # m
            best_theta_g, best_idx = torch.min(theta_g,0)
            best_theta = theta[best_idx] # c, l, w
            success = True
        
        return best_theta[None,:], best_theta_g, success 

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
        d = int(np.prod([*x.shape]))
        update_theta, update_theta_g, success = self.init_adv_batch(model,x,y)
        if success:
            alpha = self.alpha_init
            sigma = self.sigma_init
            xadv = x + update_theta * update_theta_g # initialize xadv
            for i in range(self.iterations):
                sign_grad = self.grad_sign_est(model,x,y,update_theta,update_theta_g,sigma) # 1, c, l, w
                min_theta, min_theta_g = update_theta, update_theta_g
                # increasing grid search for alpha
                for _ in range(10):
                    temp_theta = update_theta - alpha * sign_grad
                    temp_theta /= torch.norm(temp_theta)
                    temp_theta_g = self.bin_search_local(model,x,y,temp_theta,min_theta_g,sigma)
                    alpha *= 2
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
    mask = np.zeros((w,w))
    mask[x[select],y[select]] = 1
    return mask

class TriAttack():
    """
    Triangle Attack: https://arxiv.org/abs/2112.06569
    this implementation processes samples in parallel
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
        direction_theta = (difference * np.cos(theta) + 
                           torch.norm(difference,p=2,dim=(1,2,3),keepdim=True) / 
                           torch.norm(orthogonal,p=2,dim=(1,2,3),keepdim=True) *
                           orthogonal * np.sin(theta))
        direction_theta = (direction_theta / torch.norm(direction_theta,p=2,dim=(1,2,3),keepdim=True) * 
                           torch.norm(difference,p=2,dim=(1,2,3),keepdim=True))
        return direction_theta
    
    def get_orthogonal_1d_in_subspace(self,difference,is_left=True):
        b,c,w,h = difference.shape
        side_length = min(w,h)
        zero_mask = torch.zeros(w,h)
        size_mask = int(round(side_length*self.ratio_mask))
        if is_left:
            zero_mask[:size_mask,:size_mask] = 1
        else:
            zero_mask[-size_mask:,-size_mask:,] = 1
        to_choose = torch.where(zero_mask == 1)
        x = to_choose[0]
        y = to_choose[1]
        dim_num = min(len(x),self.dim_num)
        args = []
        for i in range(b*c):
            args.append([side_length,x,y,dim_num])
        mask = self.pool.map(get_mask,args)
        mask = torch.tensor(np.stack(mask,axis=0)).reshape(difference.shape).to(difference.device)
        mask *= torch.randn_like(mask)
        direction = self.rotate(difference,mask,np.pi/2)
        orthogonal = (direction / torch.norm(direction,p=2,dim=(1,2,3),keepdim=True) * 
                      torch.norm(difference,p=2,dim=(1,2,3),keepdim=True))
        return orthogonal
    
    def get_xadv_2d(self,model,x,xadv,y,axis1,axis2,init_alpha,
                    max_iter=2,plus_lr=0.1,minus_lr=0.005,half_range=0.1):
        alpha = init_alpha # an array of shape b,1,1,1
        x = torch_dct.dct_2d(x)
        xadv = torch_dct.dct_2d(xadv)
        upper = np.pi/2+half_range
        lower = np.pi/2-half_range

        d = torch.norm(xadv-x,p=2,dim=(1,2,3),keepdim=True)
        pi = torch.tensor(np.repeat(np.pi,alpha.shape[0])).reshape(-1,1,1,1).to(alpha.device)
        theta = torch.clamp(pi-2*alpha,min=0)+torch.minimum(pi/16,alpha/2)

        x_hat = torch_dct.idct_2d(xadv).float()
        right_theta = pi - alpha
        flag = torch.zeros_like(right_theta)

        x_temp = (x + d*(axis1*torch.cos(theta)+
                         axis2*torch.sin(theta))/
                  torch.sin(alpha)*torch.sin(alpha+theta)).float()
        
        x_temp = torch_dct.idct_2d(x_temp).float()
        if self.clip_min is not None and self.clip_max is not None:
            x_temp = torch.clamp(x_temp,self.clip_min,self.clip_max)
        with torch.no_grad():
            pred = model(x_temp)
        if len(pred.shape) == 1:
            pred = pred[None,:]
        label = torch.argmax(pred,dim=1)
        
        success_temp = (label != y)
        if torch.sum(success_temp) > 0:
            flag[success_temp] = 1
        if torch.sum(~success_temp) > 0: # if there exist samples not succeeded
            alpha[~success_temp] -= minus_lr
            alpha = torch.clamp(alpha,min=lower)
            theta[~success_temp] = torch.maximum(theta[~success_temp], np.pi-2*alpha[~success_temp]+np.pi/64)
            
            x_temp[~success_temp] = (x[~success_temp] + d[~success_temp]*(axis1[~success_temp]*torch.cos(theta[~success_temp])-
                                                                          axis2[~success_temp]*torch.sin(theta[~success_temp]))/
                                torch.sin(alpha[~success_temp])*torch.sin(alpha[~success_temp]+theta[~success_temp])).float()
            x_temp[~success_temp] = torch_dct.idct_2d(x_temp[~success_temp]).float()
            if self.clip_min is not None and self.clip_max is not None:
                x_temp = torch.clamp(x_temp,self.clip_min,self.clip_max)
            with torch.no_grad():
                pred = model(x_temp)
            if len(pred.shape) == 1:
                pred = pred[None,:]
            label = torch.argmax(pred,dim=1)

        success_update = (label != y)
        if torch.sum(~success_update) > 0: # update alpha for samples inside the boundary
            alpha[~success_update] -= minus_lr
            alpha = torch.clamp(alpha,min=lower)
        # perform binary search for adversarial samples
        if torch.sum(success_update) > 0:
            x_hat[success_update] = x_temp[success_update]
            flag[torch.logical_xor(success_temp,success_update)] = -1

            left_theta = theta[success_update]
            right_theta = right_theta[success_update]
            theta_update = (left_theta + right_theta) / 2
            x_hat_update, x_update, y_update = x_hat[success_update], x[success_update], y[success_update]
            d_update, axis1_update, axis2_update = d[success_update],axis1[success_update],axis2[success_update]
            alpha_update, flag_update = alpha[success_update], flag[success_update]
            success_prev = torch.zeros_like(y_update,dtype=torch.bool)
            for i in range(max_iter):
                x_temp_update = (x_update + d_update*(axis1_update*torch.cos(theta_update)+
                                                      flag_update*axis2_update*torch.sin(theta_update))/
                                 torch.sin(alpha_update)*torch.sin(alpha_update+theta_update)).float()
                x_temp_update = torch_dct.idct_2d(x_temp_update).float()
                if self.clip_min is not None and self.clip_max is not None:
                    x_temp_update = torch.clamp(x_temp_update,self.clip_min,self.clip_max)
                with torch.no_grad():
                    pred = model(x_temp_update)
                if len(pred.shape) == 1:
                    pred = pred[None,:]
                label = torch.argmax(pred,dim=1)
                
                success_temp = label != y_update
                if torch.sum(~success_temp) != 0:
                    alpha_update[~success_temp] -= minus_lr
                    alpha_update = torch.clamp(alpha_update,min=lower)
                    theta_update[~success_temp] = torch.maximum(theta_update[~success_temp],
                                                                np.pi-2*alpha_update[~success_temp]+np.pi/64)
                    flag_update[~success_temp] = -flag_update[~success_temp]

                    x_temp_update[~success_temp] = (x_update[~success_temp]+
                                                    d_update[~success_temp]*(axis1_update[~success_temp]*torch.cos(theta_update[~success_temp])+
                                                                                flag_update[~success_temp]*torch.sin(theta_update[~success_temp]))/
                                                    torch.sin(alpha_update[~success_temp])*torch.sin(alpha_update[~success_temp]+theta_update[~success_temp])).float()
                    x_temp_update[~success_temp] = torch_dct.idct_2d(x_temp_update[~success_temp]).float()
                    if self.clip_min is not None and self.clip_max is not None:
                        x_temp_update = torch.clamp(x_temp_update,self.clip_min,self.clip_max)
                    with torch.no_grad():
                        pred = model(x_temp_update)
                    if len(pred.shape) == 1:
                        pred = pred[None,:]
                    label = torch.argmax(pred,dim=1)
                
                success_update_2 = (label != y_update)
                # update alpha for succeeded samples
                mask = torch.logical_xor(success_prev,success_update_2)
                if torch.sum(mask) > 0:
                    alpha_update[mask] += plus_lr
                    alpha_update = torch.clamp(alpha_update,max=upper)
                if torch.sum(~success_update_2) == 0:
                    break # early stop
                # update alpha for unsucceeded samples
                alpha_update[~success_update_2] -= minus_lr
                alpha_update = torch.clamp(alpha_update,min=lower)

                left_theta[~success_update_2] = (torch.clamp(np.pi-2*alpha_update[~success_update_2],min=0)+
                                                 torch.clamp(alpha_update[~success_update_2],max=np.pi/16))
                right_theta[~success_update_2] = theta_update[~success_update_2]

                # update for next round
                success_prev = success_update_2
                theta_update = (left_theta + right_theta) / 2
            # update for un succeeded samples for next outer iteration
            if torch.sum(~success_update_2) > 0:
                alpha_update[~success_update_2] += plus_lr
                alpha_update = torch.clamp(alpha_update,max=upper)
            if torch.sum(success_update_2) > 0:
                x_hat_update[success_update_2] = x_temp_update[success_update_2]
            x_hat[success_update] = x_hat_update
            alpha[success_update] = alpha_update
            
        return x_hat, alpha
        

    def adv_gen(self,model,x,y):
        d = np.prod([*x.shape[1:]])
        x_adv, is_adv = self.init_adv(model,x,y)
        x,y,x_adv_update = x[is_adv],y[is_adv],x_adv[is_adv]
        dist = torch.norm(x-x_adv_update,p=2,dim=(1,2,3))
        within_budget = dist<=np.sqrt(self.epsilon*d)
        alpha = torch.tensor(np.repeat(np.pi/2,x.shape[0])).reshape(-1,1,1,1).to(x.device)
        for i in range(self.iterations):
            if torch.sum(within_budget) == x.shape[0]:
                break
            x_adv_temp, x_temp, y_temp = x_adv_update[~within_budget], x[~within_budget], y[~within_budget]
            difference = torch_dct.dct_2d(x_adv_temp-x_temp)
            axis_1 = difference / torch.norm(difference,p=2,dim=(1,2,3),keepdim=True)
            direction = self.get_orthogonal_1d_in_subspace(difference)
            axis_2 = direction / torch.norm(direction,p=2,dim=(1,2,3),keepdim=True)
            x_adv_temp,alpha[~within_budget] = self.get_xadv_2d(model,x_temp,x_adv_temp,y_temp,axis_1,axis_2,alpha[~within_budget])
            x_adv_update[~within_budget] = x_adv_temp
            dist = torch.norm(x-x_adv_update,p=2,dim=(1,2,3))
            within_budget = dist<=np.sqrt(self.epsilon*d)

        x_adv_update[~within_budget] = x[~within_budget]
        x_adv[is_adv] = x_adv_update
        return x_adv
