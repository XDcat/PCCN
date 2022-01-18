#TRP TOPOLOGY for Swiss-PdbViewer 3.5
 based on GROMOS96 parameters kindly provided by Wilfred van Gunsteren.  
 Ref: W.F. van Gunsteren et al. (1996) in Biomolecular simulation:
      the GROMOS96 manual and user guide. Vdf Hochschulverlag ETHZ
      (http://igc.ethz.ch/gromos)
-----------------------------------------------------------------------
//NAME    TYPE      CHARGE-B1  AT   GRO-43A1  SIMPEL  
 N        N         -0.2800     5   -0.2800    0.0000 
 H        H          0.2800    18    0.2800    0.0000 
 CA       CH1        0.0000    12    0.0000    0.0000 
 CB       CH2        0.0000 *  13    0.0000    0.0000 
 CG       C         -0.1400 *  11   -0.1400    0.0000 
 CD1      C         -0.1000 *  11   -0.1000    0.0000 
 HD1      HC         0.1000 *  17    0.1000    0.0000 
 CD2      C          0.0000 *  11    0.0000    0.0000 
 NE1      NR        -0.0500 *   8   -0.0500    0.0000 
 HE1      H          0.1900 *  18    0.1900    0.0000 
 CE2      C          0.0000 *  11    0.0000    0.0000 
 CE3      C         -0.1000 *  11   -0.1000    0.0000 
 HE3      HC         0.1000 *  17    0.1000    0.0000 
 CZ2      C         -0.1000 *  11   -0.1000    0.0000 
 HZ2      HC         0.1000 *  17    0.1000    0.0000 
 CZ3      C         -0.1000 *  11   -0.1000    0.0000 
 HZ3      HC         0.1000 *  17    0.1000    0.0000 
 CH2      C         -0.1000 *  11   -0.1000    0.0000 
 HH2      HC         0.1000 *  17    0.1000    0.0000 
 C        C          0.3800    11    0.3800    0.0000 
 O        O         -0.3800     1   -0.3800    0.0000 
//BOND
---------
 N        H            2
 N        CA          20
 CA       C           26
 C        O            4
 C       +N            9
 CA       CB          26
 CB       CG          26
 CG       CD1          9
 CG       CD2         15
 CD1      HD1          3
 CD1      NE1          9
 CD2      CE2         15
 CD2      CE3         15
 NE1      HE1          2
 NE1      CE2          9
 CE2      CZ2         15
 CE3      HE3          3
 CE3      CZ3         15
 CZ2      HZ2          3
 CZ2      CH2         15
 CZ3      HZ3          3
 CZ3      CH2         15
 CH2      HH2          3
//ANGLE
-----------------------
-C        N        H           31
 H        N        CA          17
-C        N        CA          30
 N        CA       C           12
 CA       C       +N           18
 CA       C        O           29
 O        C       +N           32
 N        CA       CB          12
 C        CA       CB          12
 CA       CB       CG          14
 CB       CG       CD1         36
 CB       CG       CD2         36
 CD1      CG       CD2          6
 CG       CD1      HD1         35
 HD1      CD1      NE1         35
 CG       CD1      NE1          6
 CG       CD2      CE2          6
 CD1      NE1      CE2          6
 CD1      NE1      HE1         35
 HE1      NE1      CE2         35
 NE1      CE2      CD2          6
 CG       CD2      CE3         38
 NE1      CE2      CZ2         38
 CD2      CE2      CZ2         26
 CE2      CD2      CE3         26
 CD2      CE3      HE3         24
 HE3      CE3      CZ3         24
 CD2      CE3      CZ3         26
 CE2      CZ2      HZ2         24
 HZ2      CZ2      CH2         24
 CE2      CZ2      CH2         26
 CE3      CZ3      HZ3         24
 HZ3      CZ3      CH2         24
 CE3      CZ3      CH2         26
 CZ2      CH2      HH2         24
 HH2      CH2      CZ3         24
 CZ2      CH2      CZ3         26
//IMPROPER
-----------------------
 N       -C        CA       H            1
 C        CA      +N        O            1
 CA       N        C        CB           2
 CG       CD1      CD2      CB           1
 CD2      CG       CD1      NE1          1
 CD1      CG       CD2      CE2          1
 CG       CD1      NE1      CE2          1
 CG       CD2      CE2      NE1          1
 CD1      NE1      CE2      CD2          1
 CD1      CG       NE1      HD1          1
 NE1      CD1      CE2      HE1          1
 CD2      CE2      CE3      CG           1
 CE2      CD2      CZ2      NE1          1
 CE3      CD2      CE2      CZ2          1
 CD2      CE2      CZ2      CH2          1
 CE2      CD2      CE3      CZ3          1
 CE2      CZ2      CH2      CZ3          1
 CD2      CE3      CZ3      CH2          1
 CE3      CZ3      CH2      CZ2          1
 CE3      CD2      CZ3      HE3          1
 CZ2      CE2      CH2      HZ2          1
 CZ3      CE3      CH2      HZ3          1
 CH2      CZ2      CZ3      HH2          1
//TORSION
-----------------------
-CA      -C        N        CA           4
-C        N        CA       C           19
 N        CA       C       +N           20
 N        CA       CB       CG          17
 CA       CB       CG       CD2         20
