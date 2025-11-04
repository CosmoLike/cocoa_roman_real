# Roman Real Space 3x2pt Analysis

## Medium Tier Survey Version Specific
- survey area: 2415 square degrees
- n_eff: 41.3 per square arcmin

## Files included:
- Covariance matrix (cov_dc1)
- Data vector (challenge1.modelvector)
- Redshift distribution (challenge1.nz)
- Default mask (dc1.mask)
- Dataset file (dc1.dataset)
- Example mcmc yaml file (EXAMPLE_MCMC.yaml)

## Relevant general details:
 - 8 tomographic bins
 - theta_min = 2.5, theta_max = 250, ntheta = 15

## Data vector:
Each data vector contains two columns:
  - column 0: index i;
  - column 1: value of data vector for corresponding probe/bin combination

The data vectors are ordered with the first block corresponding to xi+, followed by xi-, gammat, w

## Covariance matrix details:
The covariance matrix was computed with [CosmoCov](https://github.com/CosmoLike/CosmoCov/tree/master) and follows the format described there with:
 - column 0, 1: matrix indices
 - column 2, 3: the corresponding bin-averaged angular separations (in radians) of the element
 - column 4, 5, 6, 7: the tomographic bins involved
 - column 8, 9: the Gaussian part and the non-Gaussian part of the element. The total value is the sum of the two.

For galaxy-galaxy lensing, we exclude bin combos where the z_min(lens bin) > z_max(source bin), i.e. where lens is fully behind source. This results in the exclusion of bin combos (l,s) = [(6,0)(7,0),(7,1)]. These have already been excluded in the provided matrix and data vector.

## Default mask:

The matrix, as given, is not positive definite. To make the matrix positive definite, you must make a cut on small-theta scales for galaxy-galaxy lensing and galaxy-clustering components. We find that a galaxy clustering cut at scales of 1.5 Mpc/h is sufficient for this purpose. Using the default mask provided will automatically apply this cut by masking out the corresponding points.

## Redshift distribution:

This data challenge uses 8 tomographic bins.  The first column gives the redshift with the following 8 columns giving the number density in that redshift bin for tomographic bin 1, 2, 3 . . .
