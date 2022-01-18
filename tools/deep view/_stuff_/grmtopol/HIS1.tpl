#HIS1 TOPOLOGY for Swiss-PdbViewer 3.5
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
 ND1      NR         0.0000 *   8    0.0000    0.0000
 HD1      H          0.1900 *  18    0.1900    0.0000
 CD2      CR1        0.1300 *  16    0.1300    0.0000
 CE1      CR1        0.2600 *  16    0.2600    0.0000
 NE2      NR        -0.5800 *   8   -0.5800    0.0000
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
 CG       ND1          9
 CG       CD2          9
 ND1      HD1          2
 ND1      CE1          9
 CD2      NE2          9
 CE1      NE2          9
 NE2     -C           32
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
 CB       CG       ND1         36
 CB       CG       CD2         36
 ND1      CG       CD2          6
 CG       ND1      HD1         35
 CG       ND1      CE1          6
 HD1      ND1      CE1         35
 CG       CD2      NE2          6
 ND1      CE1      NE2          6
 CD2      NE2      CE1          6
 CD2      NE2     -C           33
 CE1      NE2     -C           33
 NE2     -C       -O           16
 NE2     -C       -NE2         16
 NE2     -C       -OE1         16
 NE2     -C       -OD2         16
//IMPROPER
-----------------------
 N       -C        CA       H            1
 C        CA      +N        O            1
 CA       N        C        CB           2
 CG       ND1      CD2      CB           1
 CD2      CG       ND1      CE1          1
 ND1      CG       CD2      NE2          1
 CG       ND1      CE1      NE2          1
 CG       CD2      NE2      CE1          1
 CD2      NE2      CE1      ND1          1
 ND1      CG       CE1      HD1          1
//TORSION
-----------------------
-CA      -C        N        CA           4
-C        N        CA       C           19
 N        CA       C       +N           20
 N        CA       CB       CG          17
 CA       CB       CG       ND1         20
 CD2      NE2     -C       -O           18
