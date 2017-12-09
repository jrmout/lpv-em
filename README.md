# lpv-em
This repository provides MATLAB code to estimate stable dynamical systems (DS) from data. It was tested only in MATLAB 2016a but it should be compatible with any modern MATLAB version.
It uses <a href="https://yalmip.github.io/">YALMIP</a> as optimization interface and includes the <a href="https://github.com/sqlp/sedumi">sedumi</a> solver as submodule, but you can use any other solver of your choice supported by YALMIP (check <a href="https://yalmip.github.io/allsolvers/">here</a> for a list of supported solvers). 
For nonconvex problems it relies on <a href="https://de.mathworks.com/matlabcentral/fileexchange/43643-fminsdp">FMINSDP</a>.
If you use this code for your research please cite [this paper](http://proceedings.mlr.press/v78/medina17a/medina17a.pdf)
```
J. Medina and A. Billard, 'Learning Stable Task Sequences from Demonstration with Linear Parameter Varying Systems and Hidden Markov Models'. In Conference on Robot Learning (CoRL), Mountain View, U.S.A., 2017.' 
```
You can watch the talk explaining this algorithm [here](https://youtu.be/xfyK03MEZ9Q?t=1h47m5s). 

To run the code first init and update the respective submodules. In the terminal, go to your lpv_em folder
```
$ cd your_lpv_em_folder
```
then
```
$ git submodule update --init --recursive
```
After this, to test the LPV-EM algorithm in the MATLAB command window run
```
$ demo_gui_mix_lds_inv_max
```
and draw some trajectories on it. After each trajectory is drawn, the optimal model parameters will be recomputed. If you want to simulate the encoded dynamics enable the 'Simulate' button and click somewhere on the figure. During simulation you can perturb the system with your mouse to see how it adapts. You can watch a video of this demo running <a href="https://www.youtube.com/watch?v=ojAhun_1_uQ">here</a> and [here](https://youtu.be/xfyK03MEZ9Q?t=1h47m5s). It looks like this
<a href="https://www.youtube.com/watch?v=92Xg3rFH8ag&t=22s">![Exemplary LPV-EM output](plot/lpv_em.png)</a>

An extension of this model to multiple attractors reached in a sequence can be found <a href="https://www.youtube.com/watch?v=92Xg3rFH8ag&t=22s">here</a>.

The repository also provides other alternative solutions:
- 'demo_gui_mix_lds.m' considers a standard mixture of linear dynamical systems and tries to solve the estimation problem relying on nonconvex problems. As a result it might get stuck into local minima. 
- 'demo_gui_mix_inv_lds.m' considers a mixture of inverse linear dynamical systems and therefore remains convex. However, the inverse formulation might produce inaccurate results and the LPV-EM algorithm is preferable. 
- 'demo_gui_lds.m' considers a single linear dynamical system.
- 'demo_gui_inv_lds.m' considers an inverse linear dynamical system. 
- demos without gui: 'demo_lds.m' (1 single DS), 'demo_inv_lds.m' (1 single inverse DS), 'demo_mix_lds.m' (mixture DS), 'demo_mix_inv_lds.m' (mixture inverse DS), 'demo_mix_lds_inv_max.m' (LPV-EM algorithm)
