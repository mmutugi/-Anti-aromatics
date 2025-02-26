#!/usr/bin/env python3

import pyscf

mol = pyscf.gto.M( output = 'Mole' ,unit = "Angstrom",verbose = 4, basis = "sto-3g", spin = 2, atom =

        [('C',1.01575272635155,-0.00000001277915,0.00000011971704),
  ('C',0.00000011307059,1.01575370679373,-0.00000029889446),
  ('C', -1.01575237942000,-0.00000015398183,0.00000035590132),
  ('C', 0.00000095569805,-1.01575310784758,-0.00000017949877),
  ('H', 2.09371923119101,-0.00000126376536,0.00000067124784),
  ('H',-0.00000072792838,2.09372059198303,-0.00000058631836),
  ('H', -2.09372110538908,0.00000110659851,0.00000055970058),
  ('H', 0.00000118642625,-2.0937208670013, -0.00000064185520)])

# Basic UHF
mf = pyscf.scf.UHF(mol).run()
mf.dump_scf_summary()
print(f'UHF = {mf.e_tot}')

# Coupled cluster on top of UHF reference
mycc = pyscf.cc.CCSD(mf).run()
et_correction = mycc.ccsd_t()
print(f'UCCSD = {mycc.e_tot} | UCCSD(T) = {mycc.e_tot + et_correction}')

# UMP2
mp2 = mf.MP2().run()
print(f'UMP2 = {mp2.e_tot}')