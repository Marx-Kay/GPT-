"""Continuous-time CGM-inspired reformulation: coefficient and HJB building blocks.
State (age, financial wealth, permanent income, transitory log-income).
This module defines a CHANGED model, not an exact embedding of annual iid shocks.
See continuous_time_design.md for retirement, terminal and constraint conditions.
No trained lifecycle policy is shipped by this module.
"""
from dataclasses import dataclass
import math
import torch
from ito import ito_generator

@dataclass(frozen=True)
class Parameters:
    gamma: float=10.
    discount_rate: float=-math.log(.96)
    riskfree: float=math.log(1.02)
    equity_premium: float=math.log(1.06)-math.log(1.02)
    stock_vol: float=math.sqrt(math.log(1+(.157/1.06)**2))
    permanent_vol: float=math.sqrt(.0106)
    transitory_variance: float=.0738
    ou_speed: float=-math.log(.01)  # NEW assumption: one-year autocorrelation .01.
    stock_income_correlation: float=0.
    replacement: float=.68212
    retirement: float=66.
    terminal_age: float=100.

def coefficients(s, consumption, equity_share, p=Parameters()):
    a,w,perm,z=s.unbind(-1)
    working=a<p.retirement
    growth=(.16818+2*(-.0323371/10)*a+3*(.0019704/100)*a.square())
    # Log P drift = f'(a), so level drift includes the Ito correction.
    pvol=torch.where(working,p.permanent_vol,0.)
    gp=torch.where(working,growth+.5*p.permanent_vol**2,0.)
    income=torch.where(working,perm*z.exp(),p.replacement*perm)
    drift=torch.stack([torch.ones_like(a),(p.riskfree+equity_share*p.equity_premium)*w+income-consumption,
                       gp*perm,torch.where(working,-p.ou_speed*z,0.)],dim=-1)
    zero=torch.zeros_like(a)
    stock=torch.stack([zero,equity_share*p.stock_vol*w,p.stock_income_correlation*pvol*perm,zero],dim=-1)
    persistent=torch.stack([zero,zero,math.sqrt(1-p.stock_income_correlation**2)*pvol*perm,zero],dim=-1)
    temporary=torch.stack([zero,zero,zero,torch.where(working,math.sqrt(2*p.ou_speed*p.transitory_variance),0.)],dim=-1)
    return drift,torch.stack([stock,persistent,temporary],dim=-1)

def hjb_residual(value,state,consumption,equity_share,hazard,p=Parameters()):
    """Use one-sided retirement domains; hazard is conditional annual -log(p_survive)."""
    drift,diffusion=coefficients(state,consumption,equity_share,p)
    v=value(state)
    return consumption.pow(1-p.gamma)/(1-p.gamma)-(p.discount_rate+hazard)*v+ito_generator(value,state,drift,diffusion)
