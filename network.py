"""
© 2021 This work is licensed under a CC-BY-NC-SA license.
Title:
**Authors:** Cristiano Capone
"""

import numpy as np
from tqdm import trange

def sigm ( x, dv ):
	if dv < 1 / 30:
		return x > 0
	y = x / dv
	out = 1.5*(1. / (1. + np.exp (-y*3. )) - .5)
	return out

def gaussian(x, mu, sig):
	return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

class GOAL_GRADED:

    def __init__ (self, par):
        # This are the network size N, input I, output O and max temporal span T
        self.N, self.I, self.O, self.T = par['shape']
        net_shape = (self.N, self.T)

        self.dt = par['dt']#1. / self.T;
        self.itau_m = self.dt / par['tau_m']
        self.tau_m = par['tau_m']
        self.tau_m = np.logspace( np.log(par['tau_m']*.5) ,np.log(par['tau_m']*2.),self.N)


        self.itau_s = np.exp (-self.dt / par['tau_s'])
        self.itau_ro = np.exp (-self.dt / par['tau_ro'])
        self.dv = par['dv']

        # This is the network connectivity matrix
        self.J = np.random.normal (0., par['sigma_rec'], size = (self.N, self.N))

        # This is the network input, teach and output matrices
        self.Jin = np.random.normal (0., par['sigma_input'], size = (self.N, self.I))
        self.Jteach = np.random.normal (0., par['sigma_teach'], size = (self.N, self.O))
        self.Jout = np.random.normal (0.0, par['sigma_output'], size = (self.O,self.N))
        self.JoutV = np.random.normal (0.0, par['sigma_output'], size = (1,self.N))
        self.Jsigmaout = np.random.normal (0., par['sigma_output'], size = (self.O,self.N))

        self.dJfilt = np.zeros(np.shape(self.J))
        self.dJfilt_out = np.zeros(np.shape(self.Jout))
        self.dJfilt_sigma_out = np.zeros(np.shape(self.Jsigmaout))

        self.dJoutV_filt = np.zeros(np.shape(self.JoutV))
        self.dJoutV_aggregate = np.zeros(np.shape(self.JoutV))


        self.dJout_aggregate = np.zeros(np.shape(self.Jout))
        self.dJout_sigma_aggregate = np.zeros(np.shape(self.Jsigmaout))
        self.dJ_aggregate = np.zeros(np.shape(self.J))
        self.dh_aggregate = np.zeros(self.N,)

        self.S_filt_tot = np.zeros(self.N,)

        self.value=0
        self.r=0
        self.r_old=0

        # Remove self-connections
        np.fill_diagonal (self.J, 0.)

        self.name = 'model'

        # Impose reset after spike
        self.s_inh = -par['s_inh']
        self.Jreset = np.diag (np.ones (self.N) * self.s_inh)

        # This is the external field
        h = par['h']

        assert type (h) in (np.ndarray, float, int)
        self.h = h if isinstance (h, np.ndarray) else np.ones (self.N) * h

        # Membrane potential
        self.H = np.ones (self.N) * par['Vo']
        self.Vo = par['Vo']

        # These are the spikes train and the filtered spikes train
        self.S = np.zeros (self.N)
        self.S_hat = np.zeros (self.N)
        self.dH = np.zeros (self.N)

        # This is the single-time output buffer
        self.state_out = np.zeros (self.N)
        self.state_out_p = np.zeros (self.N)

        # Here we save the params dictionary
        self.par = par

    def _sigm (self, x, dv = None):
        if dv is None:
            dv = self.dv

        if dv < 1 / 30:
            return x > 0
        y = x / dv

        out = np.zeros (x.shape)
        mask = x > 0
        out [mask] = 1. / (1. + np.exp (-y [mask]))
        out [~mask] = np.exp (y [~mask]) / (1. + np.exp (y [~mask]))

        return out

    def _dsigm (self, x, dv = None):
        return self._sigm (x, dv = dv) * (1. - self._sigm (x, dv = dv))


    def step_ (self, inp, if_spk = False,if_tanh = False):
        itau_m = self.dt/self.tau_m
        itau_s = self.itau_s
        
        if if_spk:
        
            self.S_hat   = self.S_hat   * itau_s + self.S   * (1. - itau_s)

        else:
            self.S_hat   = np.copy(self.S)
        
        self.dH   = self.dH  * np.exp( - itau_m) + np.tanh( (1.-np.exp( -itau_m)) * self.S_hat ) 
        
        if if_spk:
            self.H   = self.H   *  np.exp( - itau_m) + (1.-np.exp( -itau_m)) * (self.J @ self.S_hat   + self.Jin @ inp + self.h)\
                                                          + self.Jreset @ self.S
            self.S   = self._sigm (self.H  , dv = self.dv) - 0.5 > 0.
        else:
            if if_tanh:
                self.H   = self.H   * np.exp( - itau_m) + (1.-np.exp( -itau_m)) *  np.tanh(self.J @ self.S_hat   + self.Jin @ inp)
            else:
                self.H   = self.H   * np.exp( - itau_m) + (1.-np.exp( -itau_m)) * (self.J @ self.S_hat   + self.Jin @ inp + self.h)
     
            self.S   = np.copy(self.H)
      
        self.y = self.Jout@self.S
        self.sigma = np.sqrt( np.clip(self.Jsigmaout @self.S ,0.05**2,0.5**2))

    def policy (self,if_derive_var=False):        
        self.mu = np.tanh(self.y)
        if if_derive_var:
            self.act_variance = np.copy(self.sigma)
        self.action = self.mu + np.random.normal (0., self.act_variance, size = ( self.O,))
        
    def reset (self, init = None):
        self.S   = init if init else np.zeros (self.N)
        self.S_hat   = self.S   * self.itau_s if init else np.zeros (self.N)
        self.state_out  *= 0
        self.state_out_p  *= 0
        self.H  *= 0.
        self.H  += self.Vo
        self.dJfilt *= 0
        self.dJfilt_out *= 0
        self.value=0
        self.value_old=0
        self.dJoutV_filt *= 0
        self.dJoutV_aggregate *= 0

        self.dJfilt_sigma_out *= 0

    def update_J (self,l2_out=0.001):      

        self.Jout = self.adam_out.step (self.Jout, self.dJout_aggregate - l2_out*self.Jout )
 
        self.dJ_aggregate=0
        self.dh_aggregate=0
        self.dJout_aggregate=0
        self.dJout_sigma_aggregate=0
        self.S_filt_tot=0


    def learn_error (self, r ):
            
            alpha_J = 1-self.gamma
            dmu = (self.action - self.mu)*(1.-self.mu**2)

            self.dJout_aggregate += r*self.dJfilt_out
            
            dJ_out = np.zeros((self.O,self.N))

            out_dim=self.out_dim
            dJ_out[0:out_dim,:] = np.outer(dmu, self.S.T)
      
            self.dJfilt_out = self.dJfilt_out*(1-alpha_J) + dJ_out
            self.grad = np.copy(r*self.dJfilt_out)
            self.dJout_aggregate += (r*self.dJfilt_out) 
            self.dh = 0
