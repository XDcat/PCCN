#SEP TOPOLOGY for Swiss-PdbViewer 3.5
 Warning this parameter comes from Serine with phosphate group
 from PDG. The Proton of the Phospahte has been united with the
 three O?Ps of the phosphate, net charge -1. 
 These parameters may not be callibrated properly.    
-----------------------------------------------------------------------
//NAME    TYPE      CHARGE-B1  AT   GRO-43A1  SIMPEL  
 N        N         -0.2800     5   -0.2800    0.0000 
 H        H          0.2800    18    0.2800    0.0000 
 CA       CH1        0.0000    12    0.0000    0.0000 
 CB       CH2        0.1500    13    0.1500    0.0000 
 OG       O4P       -0.3600     3   -0.3600    0.0000 
 P        P          1.0800    27    0.6300    0.0000
 O1P      O1P       -0.2900     2   -0.4740   -0.3330
 O2P      O2P       -0.2900     2   -0.4730   -0.3330
 O3P      O3P       -0.2900     2   -0.4730   -0.3330
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
 CB       OG          17
 OG       P           27
 P        O1P         23
 P        O2P         23
 P        O3P         23
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
 CA       CB       OG          14
 CB       OG       P           25
 OG       P        O1P         13
 OG       P        O2P         13
 OG       P        O3P         13
 O1P      P        O2P         28
 O2P      P        O3P         28
 O1P      P        O3P         28
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
 N        CA       CB       OG          17
 O3P      P        OG       CB          11
 O3P      P        OG       CB           9
 P        OG       CB       CA          14
