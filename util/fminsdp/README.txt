This is fminsdp -  a code for solving non-linear optimization problems 
with matrix inequality constraints.

To install fminsdp, simply run the file "install.m". This will
add the current folder and the subfolder "examples" to your Matlab 
path.

After running "install.m", please try out the examples found
in the subfolder "examples" and have a look at the user manual found
in the folder "docs". Note that Ipopt and PENLAB, used in example 4 and 5, 
must be downloaded and installed separately (Binaries for
Ipopt can be found here: http://www.coin-or.org/Binaries/Ipopt/. Penlab
can be obtained here: http://web.mat.bham.ac.uk/kocvara/penlab/).

Four subfolders are contained in this folder:

docs        - A brief user manual
examples    - Example codes demonstrating use of fminsdp
interfaces  - Code for interfacing with NLP-solvers KNITRO, SNOPT, Ipopt, MMA/GCMMA and PENLAB
utilities   - A couple of handy functions 


If you have comments, suggestions or bug reports, please 
feel free to contact the author via e-mail:

carl-johan.thore@liu.se



Version history

--- 1 September 2016

*  "Feasibility mode" available for the ldl-method. Tested with fmincon and gcmma 
   (performance in terms of number e.g. number of iterations not so impressive)
*  Updated documentations

--- 19 Aug 2016

*  Now implements three different methods for handling matrix constraints:
   the original Cholesky-method, PENlab and the "ldl-factorization" method. 
   Choose by setting options.method = 'cholesky', 'penlab', or 'ldl'
*  New examples demonstrating the use of PENLab- and ldl-methods
*  Improvments in the PENLab interface
*  Bug fixes and minor improvements to the MMA/GCMMA interface
*  option max_cpu_time now available for fmincon, mma/gcmma and ipopt

--- 29 Sep 2015

*  Minor changes to ipopt-interface to handle the case when there are no non-linear constraints

--- 15 Oct 2014

*  Interface to NLPSDP-solver PENLab (v1.04)

*  Call to symbol_fact moved inside fminsdp. This means that the user need only
   provide the sparsity pattern of the matrix constraints, not their symbolic
   Cholesky factorizations.

--- 21 Aug 2014

* The constraint matrices in the output struct, i.e. output.A{i}, were constructed 
  in the wrong way. When running "eig(output.A{1})" in example1.m,
  several negative eigenvalues showed up as a consequence.
  This problem is now fixed. 

--- 15 Aug 2014

* Changed random initialization for the Matlab-function "eigs" used for 
  generating initial guesses for the auxiliary variables. (opts.v0 is now
  set to a vector of ones of appropriate size; see documentation for eigs)
* Interface to NLP-solvers MMA and GCMMA (versions September 2007). 
  Less developed and tested than the other interfaces so far.
* Fixed bug that caused the Jacobian of the non-linear inequality constraints
  to have wrong size when options.c>0
* Updates in ipopt_main.m for better treatment (less memory required) of sparsity 
  patterns for the Hessian of the Lagrangian
* Added options lb_linear, lb_cineq and ub_cineq for NLP-solver Ipopt

--- 19 Feb 2014

* Minor update in sdpoptionset
* Minor change in input checks in ipopt_main
* Lagrange multipliers can now be passed to Ipopt
* MaxIter now defaults to 3000 for ipopt
