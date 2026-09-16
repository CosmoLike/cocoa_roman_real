import warnings
import os
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message=".*column is deprecated.*",
    module=r"sacc\.sacc"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*invalid value encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*overflow encountered*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Function not smooth or differentiabl*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Hartlap correction*"
)
import functools, iminuit, copy, argparse, random, time 
import emcee, itertools
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model
from getdist import IniFile
from schwimmbad import MPIPool
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog='EXAMPLE_EMUL_PROFILE3')
parser.add_argument("--nstw",
                    dest="nstw",
                    help="Number of likelihood evaluations (steps) per temperature per walker",
                    type=int,
                    nargs='?',
                    const=1,
                    default=200)
parser.add_argument("--root",
                    dest="root",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="./projects/example/")
parser.add_argument("--outroot",
                    dest="outroot",
                    help="Name of the Output File",
                    nargs='?',
                    const=1,
                    default="test.dat")
parser.add_argument("--profile",
                    dest="profile",
                    help="Which Parameter to Profile",
                    type=int,
                    nargs='?',
                    const=1,
                    default=1)
parser.add_argument("--factor",
                    dest="factor",
                    help="Factor that set the bounds (multiple of cov matrix)",
                    type=float,
                    nargs='?',
                    const=1.0,
                    default=3.0)
parser.add_argument("--numpts",
                    dest="numpts",
                    help="Number of Points to Compute Minimum",
                    type=int,
                    nargs='?',
                    const=1,
                    default=20)
parser.add_argument("--minfile",
                    dest="minfile",
                    help="Minimization Result",
                    nargs='?',
                    const=1)
parser.add_argument("--cov",
                    dest="cov",
                    help="Chain Covariance Matrix",
                    nargs='?',
                    const=1,
                    default=None)
args, unknown = parser.parse_known_args()
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
yaml_string=r"""
likelihood:
  roman_real.cosmic_shear:
    use_emulator: 1
    path: ./external_modules/data/roman_real
    data_file: example1.dataset # that assumes lens = source
prior:
  # These priors are meant to prevent the sampler to wander far off training
  g1: "lambda As_1e9: stats.norm.logpdf(As_1e9, loc=2.35, scale=1.6)"
  g2: "lambda ns: stats.norm.logpdf(ns, loc=0.96, scale=0.05)"
  g3: "lambda H0: stats.norm.logpdf(H0, loc=70, scale=10.0)"
  g4: "lambda omegab: stats.norm.logpdf(omegab, loc=0.045, scale=0.012)"
  g5: "lambda omegam: stats.norm.logpdf(omegam, loc=0.3 , scale=0.25)"
  g6: "lambda w0pwa: stats.norm.logpdf(w0pwa, loc=-1.0 , scale=1.44)"
  g7: "lambda w: stats.norm.logpdf(w, loc=-1.0, scale=1.44)"
  g8: "lambda roman_A1_1: stats.norm.logpdf(roman_A1_1, loc=0, scale=2.5)"
  g9: "lambda roman_A1_2: stats.norm.logpdf(roman_A1_2, loc=-1.7, scale=2.5)" 
params:
  As_1e9:
    prior:
      min: 0.5
      max: 5
    ref:
      dist: norm
      loc: 2.1
      scale: 0.65
    proposal: 0.4
    latex: 10^9 A_\mathrm{s}
    renames: A
  ns:
    prior:
      min: 0.87
      max: 1.07
    ref:
      dist: norm
      loc: 0.96605
      scale: 0.01
    proposal: 0.01
    latex: n_\mathrm{s}
  H0:
    prior:
      min: 55
      max: 91
    ref:
      dist: norm
      loc: 67.32
      scale: 5
    proposal: 3
    latex: H_0
  omegab:
    prior:
      min: 0.03
      max: 0.07
    ref:
      dist: norm
      loc: 0.0495
      scale: 0.004
    proposal: 0.004
    latex: \Omega_\mathrm{b}
  omegam:
    prior:
      min: 0.1
      max: 0.9
    ref:
      dist: norm
      loc: 0.316
      scale: 0.02
    proposal: 0.02
    latex: \Omega_\mathrm{m}
  w0pwa:
    prior:
      min: -5 
      max: -0.01
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}+w_{a,\mathrm{DE}}
  w:
    prior:
      min: -3
      max: -0.01
    ref:
      dist: norm
      loc: -0.99
      scale: 0.05
    proposal: 0.05
    latex: w_{0,\mathrm{DE}}
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # Nuisance parameters below (it overwrites the default settings)
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  roman_DZ_S1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^1
  roman_DZ_S2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^2
  roman_DZ_S3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^3
  roman_DZ_S4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^4
  roman_DZ_S5:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^5
  roman_DZ_S6:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^6
  roman_DZ_S7:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^7
  roman_DZ_S8:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.002
    ref:
      dist: norm
      loc: 0.0
      scale: 0.002
    proposal: 0.002
    latex: \Delta z_\mathrm{s,roman}^8
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # Intrinsic alignment
  roman_A1_1:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: 0.7
      scale: 0.5
    proposal: 0.5
    latex: A_\mathrm{1-IA,roman}^1
  roman_A1_2:
    prior:
      min: -5
      max:  5
    ref:
      dist: norm
      loc: -1.7
      scale: 0.5
    proposal: 0.5
  roman_A2_1:
    value: 0.0
    latex: A_\mathrm{2-IA,Roman}^1
  roman_A2_2:
    value: 0.0
    latex: A_\mathrm{2-IA,Roman}^2
  roman_BTA_1:
    value: 0.0
    latex: A_\mathrm{BTA-IA,Roman}^1
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  roman_M1:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^1
  roman_M2:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^2
  roman_M3:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^3
  roman_M4:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^4
  roman_M5:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^5
  roman_M6:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^6
  roman_M7:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^7
  roman_M8:
    prior:
      dist: norm
      loc: 0.0
      scale: 0.005
    ref:
      dist: norm
      loc: 0.0
      scale: 0.005
    proposal: 0.005
    latex: m_\mathrm{roman}^8
theory:
  emul_cosmic_shear:
    path: ./cobaya/cobaya/theories/
    stop_at_error: True
    extra_args: 
      device: 'cuda'
      file:  ['./projects/roman_real/emulators/nla_cosmic_shear/w0wa_hmcode2020_cs_cnn.pt']
      extra: ['./projects/roman_real/emulators/nla_cosmic_shear/w0wa_hmcode2020_cs_cnn.h5']
      ord: [['As_1e9','ns','H0','omegab', 'omegam', 'w', 'w0pwa',
             'roman_DZ_S1','roman_DZ_S2','roman_DZ_S3','roman_DZ_S4',
             'roman_DZ_S5','roman_DZ_S6','roman_DZ_S7','roman_DZ_S8',
             'roman_A1_1','roman_A1_2']]
      fast_params: [['roman_M1','roman_M2','roman_M3','roman_M4',
                     'roman_M5','roman_M6','roman_M7','roman_M8']]
      extrapar: [{'MLA': 'CNN', 
                  'INT_DIM_RES': 256, 
                  'CNN_DIM': 576, 
                  'KERNEL_DIM': 21,
                  'OUTPUT_DIM': 1080}]
"""
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
model = get_model(yaml_load(yaml_string))
def chi2(p):
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    if np.any(np.isinf(p)) or  np.any(np.isnan(p)):
      raise ValueError(f"At least one parameter value was infinite (CoCoa) param = {p}")
    point = dict(zip(model.parameterization.sampled_params(), p))
    res1 = model.logprior(point,make_finite=False)
    if np.isinf(res1) or  np.any(np.isnan(res1)):
      return 1e20
    res2 = model.loglike(point,
                         make_finite=False,
                         cached=False,
                         return_derived=False)
    if np.isinf(res2) or  np.any(np.isnan(res2)):
      return 1e20
    return -2.0*(res1+res2)
def chi2v2(p):
    p = [float(v) for v in p.values()] if isinstance(p, dict) else p
    point = dict(zip(model.parameterization.sampled_params(), p))
    logposterior = model.logposterior(point, as_dict=True)
    chi2likes=-2*np.array(list(logposterior["loglikes"].values()))
    chi2prior=-2*np.atleast_1d(model.logprior(point,make_finite=False))
    return np.concatenate((chi2likes, chi2prior))
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def min_chi2(x0,
             cov, 
             fixed=-1, 
             nstw=200,
             nwalkers=5,
             pool=None):

    def mychi2(params, *args):
        z, fixed, T = args
        params = np.array(params, dtype='float64')
        if fixed > -1:
            params = np.insert(params, fixed, z)
        return chi2(p=params)/T
    if fixed > -1:
        z      = x0[fixed]
        x0     = np.delete(x0, (fixed))
        args = (z, fixed, 1.0)
        cov = np.delete(cov, (fixed), axis=0)
        cov = np.delete(cov, (fixed), axis=1)
    else:
        args = (0.0, -2.0, 1.0)

    def logprob(params, *args):
        res = mychi2(params, *args)
        if (res > 1.e19 or np.isinf(res) or  np.isnan(res)):
          return -np.inf
        else:
          return -0.5*res
    
    class GaussianStep:
       def __init__(self, stepsize=0.2):
           self.cov = stepsize*cov
       def __call__(self, x):
           return np.random.multivariate_normal(x, self.cov, size=1)
    
    ndim        = int(x0.shape[0])
    nwalkers    = int(nwalkers)
    nstw        = int(nstw)
    if fixed == -1:
      temperature = np.array([1.0, 0.25, 0.1, 0.005, 0.001], dtype='float64')
    else:
      temperature = np.array([0.3, 0.1, 0.005, 0.001], dtype='float64')
    stepsz      = temperature/3.0

    partial_samples = [x0]
    partial = [mychi2(x0, *args)]

    for i in range(len(temperature)):
        x = [] # Initial point
        for j in range(nwalkers):
            x.append(GaussianStep(stepsize=stepsz[i])(x0)[0,:]) 
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, 
                                        ndim=ndim, 
                                        log_prob_fn=logprob, 
                                        args=(args[0], args[1], temperature[i]),
                                        moves=[(emcee.moves.DEMove(), 0.8),
                                               (emcee.moves.DESnookerMove(), 0.2)],
                                        pool=pool)
        sampler.run_mcmc(np.array(x,dtype='float64'), 
                         nstw, 
                         skip_initial_state_check=True)
        samples = sampler.get_chain(flat=True, discard=0)
        j = np.argmin(-1.0*np.array(sampler.get_log_prob(flat=True)))
        partial_samples.append(samples[j])
        partial.append(mychi2(samples[j], *args))
        x0 = copy.deepcopy(samples[j])
        sampler.reset()
    # min chi2 from the entire emcee runs
    j = np.argmin(np.array(partial))
    return partial_samples[j]
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def prf(x0, nstw, cov, fixed=-1, nwalkers=5, pool=None):
    res =  min_chi2(x0=np.array(x0, dtype='float64'), 
                    fixed=fixed,
                    cov=cov, 
                    nstw=nstw, 
                    nwalkers=nwalkers,
                    pool=pool)
    return res
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        dim      = model.prior.d()     
        nwalkers = max(3*dim, pool.comm.Get_size())
        nstw = args.nstw

        # 1st: load the cov. matrix --------------------------------------------
        if args.cov is None:
          cov = model.prior.covmat(ignore_external=False) # cov from prior
          factor = min(1.0, args.factor)
        else:
          cov = np.loadtxt(args.root+args.cov)[0:model.prior.d(),0:model.prior.d()]
          factor = args.factor
        sigma = np.sqrt(np.diag(cov))

        # 2nd: Get minimum --------------------------------------------------
        if args.minfile is not None: # load minimum from running MCMC
          x0 = np.loadtxt(args.minfile)
          chi20 = x0[-1]
          x0 = x0[0:model.prior.d()]
        else: # Compute the minimum (slow)
          (x0, results) = model.get_valid_point(max_tries=1000, 
                                     ignore_fixed_ref=False,
                                     logposterior_as_dict=True)
          res = np.array(list(prf(x0=x0, 
                                  nstw=int(5.*nstw/4.), 
                                  nwalkers=nwalkers,
                                  pool=pool,
                                  cov=cov,
                                  fixed=-1)), dtype="object")
          x0 = np.array(res, dtype='float64')[0:model.prior.d()]
          chi20 = chi2(x0)
          print(f"Global Min: params = {x0}, and chi2 = {chi20}")

        # Test consistency of the min and profile codes
        if (abs(chi2(x0)-chi20)>0.02):
          raise ValueError("Inconsistency Min and Profile setups")

        # 3rd: Set the parameter profile range ---------------------------------
        start = np.zeros(model.prior.d(), dtype='float64')
        stop  = np.zeros(model.prior.d(), dtype='float64')
        start = x0 - factor*sigma
        stop  = x0 + factor*sigma
        
        # We need to respect the YAML priors
        bounds0 = model.prior.bounds(confidence=0.999999)
        for i in range(model.prior.d()):
            if (start[i] < bounds0[i][0]):
              start[i] = bounds0[i][0]
            if (stop[i] > bounds0[i][1]):
              stop[i] = bounds0[i][1]

        half_range = (stop[args.profile] - start[args.profile]) / 2.0
       
        numpts = args.numpts-1 if args.numpts%2 == 1 else args.numpts 
      
        param  = np.linspace(start = x0[args.profile] - half_range,
                             stop  = x0[args.profile] + half_range,
                             num = numpts)
        numpts=numpts+1
        param = np.insert(param, numpts//2, x0[args.profile])
        
        # 4th Print to the terminal ---------------------------------------------
        names = list(model.parameterization.sampled_params().keys()) # Cobaya Call
        print(f"nstw (evals/Temp/walkers)={args.nstw}, "
              f" param={names[args.profile]}\n"
              f"profile param values = {param}")
        
        # 5th: Set the vectors that will hold the final result -----------------
        xf = np.tile(x0, (numpts, 1))
        xf[:,args.profile] = param

        chi2res = np.zeros(numpts)  
        chi2res[numpts//2] = chi20
        
        # 5th: run from midpoint to right --------------------------------------
        tmp = np.array(xf[numpts//2,:], dtype='float64')
        for i in range(numpts//2+1,numpts): 
            tmp[args.profile] = param[i]
            res = prf(tmp, 
                      fixed=args.profile,
                      nstw=int(nstw), 
                      nwalkers=nwalkers,
                      pool=pool,
                      cov=cov)
            xf[i,:] = np.insert(res, args.profile, param[i])
            tmp = np.array(xf[i,:],dtype='float64')
            chi2res[i] = chi2(xf[i,:])
            print(f"Partial ({i+1}/{numpts}): params={tmp}, and chi2={chi2res[i]}")
        
        # 6th: run from midpoint to left ---------------------------------------
        tmp = np.array(xf[numpts//2,:], dtype='float64')
        for i in range(numpts//2-1, -1, -1):
            tmp[args.profile] = param[i]
            res = prf(tmp, 
                      fixed=args.profile,
                      nstw=int(nstw), 
                      nwalkers=nwalkers,
                      pool=pool,
                      cov=cov)
            xf[i,:] = np.insert(res, args.profile, param[i])
            tmp = np.array(xf[i,:],dtype='float64')
            chi2res[i] = chi2(xf[i,:])
            print(f"Partial ({i+1}/{numpts}): params={tmp}, and chi2={chi2res[i]}")
        
        # 8th Append derived parameters ----------------------------------------
        xf = np.column_stack((xf, 
                              np.array([chi2v2(d) for d in xf], dtype='float64')))

        # 9th Save output file -------------------------------------------------    
        os.makedirs(os.path.dirname(f"{args.root}chains/"),exist_ok=True)
        hd = [names[args.profile],"chi2"] + names
        hd = hd + list(model.info()['likelihood'].keys()) + ["prior"]
        np.savetxt(f"{args.root}chains/{args.outroot}.{names[args.profile]}.txt",
                   np.concatenate([np.c_[param, chi2res],xf], axis=1),
                   fmt="%.9e",
                   header=f"nstw={args.nstw}, param={names[args.profile]}\n"+' '.join(hd),
                   comments="# ")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------