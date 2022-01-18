#TYR TOPOLOGY for Swiss-PdbViewer 3.5
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
 CG       C          0.0000 *  11    0.0000    0.0000  
 CD1      C         -0.1000 *  11   -0.1000    0.0000  
 HD1      HC         0.1000 *  17    0.1000    0.0000  
 CD2      C         -0.1000 *  11   -0.1000    0.0000  
 HD2      HC         0.1000 *  17    0.1000    0.0000  
 CE1      C         -0.1000 *  11   -0.1000    0.0000  
 HE1      HC         0.1000 *  17    0.1000    0.0000  
 CE2      C         -0.1000 *  11   -0.1000    0.0000  
 HE2      HC         0.1000 *  17    0.1000    0.0000  
 CZ       C          0.1500 *  11    0.1500    0.0000  
 OH       OA        -0.5480 *   3   -0.5480    0.0000  
 HH       H          0.3980    18    0.3980    0.0000  
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
 CG       CD1         15
 CG       CD2         15
 CD1      HD1          3
 CD1      CE1         15
 CD2      HD2          3
 CD2      CE2         15
 CE1      HE1          3
 CE1      CZ          15
 CE2      HE2          3
 CE2      CZ          15
 CZ       OH          12
 OH       HH           1
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
 CB       CG       CD1         26
 CB       CG       CD2         26
 CD1      CG       CD2         26
 CG       CD1      HD1         24
 HD1      CD1      CE1         24
 CG       CD1      CE1         26
 CG       CD2      HD2         24
 HD2      CD2      CE2         24
 CG       CD2      CE2         26
 CD1      CE1      HE1         24
 HE1      CE1      CZ          24
 CD1      CE1      CZ          26
 CD2      CE2      HE2         24
 HE2      CE2      CZ          24
 CD2      CE2      CZ          26
 CE1      CZ       CE2         26
 CE1      CZ       OH          26
 CE2      CZ       OH          26
 CZ       OH       HH          11
//IMPROPER
-----------------------
 N       -C        CA       H            1
 C        CA      +N        O            1
 CA       N        C        CB           2
 CG       CD1      CD2      CB           1
 CD2      CG       CD1      CE1          1
 CD1      CG       CD2      CE2          1
 CG       CD1      CE1      CZ           1
 CG       CD2      CE2      CZ           1
 CD1      CE1      CZ       CE2          1
 CD2      CE2      CZ       CE1          1
 CD1      CG       CE1      HD1          1
 CD2      CG       CE2      HD2          1
 CE1      CZ       CD1      HE1          1
 CE2      CZ       CD2      HE2          1
 CZ       CE1      CE2      OH           1
//TORSION
-----------------------
-CA      -C        N        CA           4
-C        N        CA       C           19
 N        CA       C       +N           20
 N        CA       CB       CG          17
 CA       CB       CG       CD1         20
 CE1      CZ       OH       HH           2
