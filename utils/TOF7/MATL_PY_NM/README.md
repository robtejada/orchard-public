# Coefficients for 7th-order Theory of Figures calculation

This archive contains machine-readable tables of coefficients of the ToF7
equations in several formats and code for loading them into programs written in
C++, Matlab, or Python. The original work is described in
<https://arxiv.org/abs/####.####>, where the background theory and the equations
that make use of these coefficients are explained. The purpose of this archive is
to make it easier for interested researchers to incorporate a ToF7 calculation in
their work, as loading the long list of necessary coefficients is a cumbersome and
error-prone process.

Please contact Nadine.Nettelmann@dlr.de or nmovshov@ucsc.edu with any questions or
if additional information is needed.

## Machine-readable tables

These are tables of coefficients in several formats:

* `tab_*.dat` -- These are flat ascii files containing the coefficients in blocks
  of space-delimited columns. Refer to the paper for instructions on how to
  interpret the numeric values in these files. (The other provided formats below
  were generated from these files, so that floating-point values in the binary
  formats are equivalent to the ascii values in these files. In other words, there
  is no precision benefit or penalty for using one or another format.)

* `tof7_coeffs.json` -- A structured ascii file that can be loaded by programs
  that support a json format. For example,

  ```python
  >>> import json
  >>> with open('tof7_coeffs.json','r') as f:
  ...     C7 = json.load(f)
  >>> type(C7)
  <class 'dict'>
  ```

  In a Python program the contents of the json file are saved in a hierarchical
  dict.

  ```python
  >>> C7.keys()
  dict_keys(['A', 'f'])  
  ```

  The dict `C7['A']` holds coefficients used in the equations for the $s_n$
  functions (see eq. A10 in the paper) and `C7['f']` holds the coefficients for
  calculating the $f_n$ functions (see eq. A9 in the paper). These dicts are
  further divided so that each block in the raw `.dat` tables, corresponding to a
  single terms in a single equation, is saved as a numeric array (a list of lists
  in python). For example:

  ```python
  >>> C7['A']['A0']['S0']
  ```
are the powers and coefficients found in the second block of `tab_Sn.dat`, the
ones used in the $S_0$ term of the $A_2$ equation, in the set of implicit
non-linear equations for the $s_n$ shape functions.

* `tof7_coeffs.mat` -- Binary file containing a Matlab struct variable.

    ```MATLAB
    >> load tof7_coeffs.mat % populates workspace with struct C7
    >> C7 = 
  struct with fields:
    A: [1×1 struct]
    f: [1×1 struct]
    ```

  Matlab structs are also hierarchical data structures, equivalent to python dicts
  but the syntax for accessing fields is slightly different. For example,

  ```MATLAB
  >> C7.A.A2.S0
  ```

  is the Matlab array holding the powers and coefficients of the $S_0$ term in the
  $A_2$ equation for $s_n$, and

  ```MATLAB
  >> C7.f.f2
  ```

  holds the powers of $s_n$ and coefficients needed to calculate $f_2$ for the
  integrand of $S_2$ (the second block of `tab_fn.dat`).

## Scripts

Also included in the archive are several scripts used in parsing and serializing
the raw `tab_*.dat` files. The output of these scripts are the formatted data
files above, so the real utility of the scripts is to serve as template code in
case there is a need to load the raw tables into new programs that are unable to
use the provided serialized formats.

* `read_tabs.m` -- reads the `tab_*.dat` files and creates a hierarchical Matlab
  struct. Comments in the code will help understand how to read the blocks in the
  `.dat` files.

* `pythonize_c7.m` -- converts the hierarchical Matlab struct to a Python dict and
  optionally save as a structured json file.