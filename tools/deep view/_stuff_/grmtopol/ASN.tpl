#ASN TOPOLOGY for Swiss-PdbViewer 3.5
 based on GROMOS96 parameters kindly provided by Wilfred van Gunsteren.  
 Ref: W.F. van Gunsteren et al. (1996) in Biomolecular simulation:
      the GROMOS96 manual and user guide. Vdf Hochschulverlag ETHZ
      (http://igc.ethz.ch/gromos)
-----------------------------------------------------------------------
//NAME    TYPE      CHARGE-B1  AT   GRO-43A1  SIMPEL    PARSE
 N        N         -0.2800     5   -0.2800    0.0000   -0.4000
 H        H          0.2800    18    0.2800    0.0000    0.4000
 CA       CH1        0.0000    12    0.0000    0.0000    0.0000
 CB       CH2        0.0000    13    0.0000    0.0000    0.0000
 CG       C          0.3800    11    0.3800    0.0000    0.5500
 OD1      O         -0.3800     1   -0.3800    0.0000   -0.5500
 ND2      NT        -0.8300     6   -0.8300    0.0000   -0.7800
 HD21     H          0.4150    18    0.4150    0.0000    0.3900
 HD22     H          0.4150    18    0.4150    0.0000    0.3900
 C        C          0.3800    11    0.3800    0.0000    0.5500
 O        O         -0.3800     1   -0.3800    0.0000   -0.5500
//BOND
---------
 N        H            2
 N        CA          20
 CA       C           26
 C        O            4
 C       +N            9
 CA       CB          26
 CB       CG          26
 CG       OD1          4
 CG       ND2          8
 ND2      HD21         2
 ND2      HD22         2
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
 CB       CG       OD1         29
 CB       CG       ND2         18
 OD1      CG       ND2         32
 CG       ND2      HD21        22
 CG       ND2      HD22        22
 HD21     ND2      HD22        23
//IMPROPER
-----------------------
 N       -C        CA       H            1
 C        CA      +N        O            1
 CA       N        C        CB           2
 CG       OD1      ND2      CB           1
 ND2      HD21     HD22     CG           1
//TORSION
-----------------------
-CA      -C        N        CA           4
-C        N        CA       C           19
 N        CA       C       +N           20
 N        CA       CB       CG          17
 CA       CB       CG       ND2         20
 CB       CG       ND2      HD21         4
