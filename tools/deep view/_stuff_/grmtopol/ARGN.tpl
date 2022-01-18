#ARGN TOPOLOGY for Swiss-PdbViewer 3.5
 based on GROMOS96 parameters kindly provided by Wilfred van Gunsteren.  
 Ref: W.F. van Gunsteren et al. (1996) in Biomolecular simulation:
      the GROMOS96 manual and user guide. Vdf Hochschulverlag ETHZ
      (http://igc.ethz.ch/gromos)
-----------------------------------------------------------------------
//NAME    TYPE      CHARGE-B1  AT   GRO-43A1  SIMPEL   
 N        N         -0.2800     5   -0.2800    0.0000  
 H        H          0.2800    18    0.2800    0.0000  
 CA       CH1        0.0000    12    0.0000    0.0000  
 CB       CH2        0.0000    13    0.0000    0.0000  
 CG       CH2        0.0000    13    0.0000    0.0000  
 CD       CH2        0.0000    13    0.0000    0.0000  
 NE       NE        -0.2800    10   -0.2800    0.0000  
 HE       H          0.2800    18    0.2800    0.0000  
 CZ       C          0.1500    11    0.1500    0.0000  
 NH1      NE        -0.5480    10   -0.5480    0.0000  
 HH1      H          0.3980    18    0.3980    0.0000  
 NH2      NZ        -0.8300     9   -0.8300    0.0000  
 HH21     H          0.4150    18    0.4150    0.0000  
 HH22     H          0.4150    18    0.415     0.0000  
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
 CG       CD          26
 CD       NE          20
 NE       HE           2
 NE       CZ          10
 CZ       NH1         10
 CZ       NH2         10
 NH1      HH1          2
 NH2      HH21         2
 NH2      HH22         2
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
 CB       CG       CD          14
 CG       CD       NE          12
 CD       NE       HE          19
 HE       NE       CZ          22
 CD       NE       CZ          32
 NE       CZ       NH1         27
 NE       CZ       NH2         27
 NH1      CZ       NH2         27
 CZ       NH1      HH1         22
 CZ       NH2      HH21        22
 CZ       NH2      HH22        22
 HH21     NH2      HH22        23
//IMPROPER
-----------------------
 N       -C        CA       H            1
 C        CA      +N        O            1
 CA       N        C        CB           2
 NE       CD       CZ       HE           1
 CZ       NH1      NH2      NE           1
 NH2      HH21     HH22     CZ           1
//TORSION
-----------------------
-CA      -C        N        CA           4
-C        N        CA       C           19
 N        CA       C       +N           20
 N        CA       CB       CG          17
 CA       CB       CG       CD          17
 CB       CG       CD       NE          17
 CG       CD       NE       CZ          19
 CD       NE       CZ       NH1          4
 NE       CZ       NH1      HH1          4
 NE       CZ       NH2      HH21         4
