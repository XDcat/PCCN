#LYS TOPOLOGY for Swiss-PdbViewer 3.5
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
 CE       CH2        0.0000    13    0.0000    0.0000 
 NZ       NT        -0.8300     6   -0.8300    0.0000 
 HZ1      H          0.4150    18    0.4150    0.0000 
 HZ2      H          0.4150    18    0.4150    0.0000 
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
 CD       CE          26
 CE       NZ          20
 NZ       HZ1          2
 NZ       HZ2          2
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
 CG       CD       CE          14
 CD       CE       NZ          14
 CE       NZ       HZ1         10
 CE       NZ       HZ2         10
 HZ1      NZ       HZ2          9
//IMPROPER
-----------------------
 N       -C        CA       H            1
 C        CA      +N        O            1
 CA       N        C        CB           2
//TORSION
-----------------------
-CA      -C        N        CA           4
-C        N        CA       C           19
 N        CA       C       +N           20
 N        CA       CB       CG          17
 CA       CB       CG       CD          17
 CB       CG       CD       CE          17
 CG       CD       CE       NZ          17
 CD       CE       NZ       HZ1         14
