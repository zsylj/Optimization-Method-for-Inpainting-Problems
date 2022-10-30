# Optimization-Method-for-Inpainting-Problems
This is a repo for Image Inpainting Problem
the mathematical model can be modeled as:

0.5 * || K*u - f ||^2_2 + tau | D * u |_1

where:

u: Vectorized Ground Truth

K: Mask, discard pixels randomly

f: Measurement, or Masked Image

D: Difference Matrix, Can be used for An-isotropic TV(ATV)


file name pattern:

AA_BB_Inpainting_CC_DD

AA: Optimization Method

BB: Regularization Term

CC: color or grayscale image

DD: single image or multiple image(may require more RAM)

Optimization Method Including:

ISTA

FISTA

SALSA

Chambolle:

  https://link.springer.com/article/10.1023/B:JMIV.0000011325.36760.1e
  
Split Bregman:

  https://epubs.siam.org/doi/10.1137/080725891
  
GAP:

  https://arxiv.org/pdf/1511.03890.pdf  
  
ADMM:

  https://web.stanford.edu/class/ee364b/lectures/admm_slides.pdf
  
  https://arxiv.org/pdf/1108.1587.pdf
