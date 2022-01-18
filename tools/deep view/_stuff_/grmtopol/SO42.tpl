#SO42-  TOPOLOGY for Swiss-PdbViewer 3.5
 based on GROMOS96 parameters kindly provided by Wilfred van Gunsteren.  
 Ref: W.F. van Gunsteren et al. (1996) in Biomolecular simulation:
      the GROMOS96 manual and user guide. Vdf Hochschulverlag ETHZ
      (http://igc.ethz.ch/gromos)
-----------------------------------------------------------------------
//NAME    TYPE      CHARGE-B1  AT   GRO-43A1  SIMPEL  
 IS       S          1.4400    20    0.5400    0.0000 
 IO1      OM        -0.3600     2   -0.6350   -0.5000 
 IO2      OM        -0.3600     2   -0.6350   -0.5000 
 IO3      OM        -0.3600     2   -0.6350   -0.5000 
 IO4      OM        -0.3600     2   -0.6350   -0.5000 
//BOND
---------
 IS       IO1         24
 IS       IO2         24
 IS       IO3         24
 IS       IO4         24
//ANGLE
-----------------------
 IO1      IS       IO2         12
 IO1      IS       IO3         12
 IO1      IS       IO4         12
 IO2      IS       IO3         12
 IO2      IS       IO4         12
 IO3      IS       IO4         12
//IMPROPER
-----------------------
//TORSION
-----------------------
