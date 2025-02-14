#!/usr/bin/env python3

from pyscf import gto, scf
from pyblock2._pyscf.mcscf import sort_orbitals
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes, MPOAlgorithmTypes
import scipy.stats
import numpy as np
import os

atoms = '''
  C          -5.73145150963860      1.57627838105768     -0.00000921349277
  C          -4.86844675231667      2.68762312712735     -0.00002953980194
  C          -3.56528207868998      3.21858476791439     -0.00001954635435
  C          -2.17139954399532      3.02689504921304      0.00001678650836
  C          -1.05985229542323      2.16380968729628      0.00003480504931
  C          -0.52869687960549      0.86104555826866      0.00001941612897
  C          -5.92303569171562      0.18221189398057      0.00002014699207
  C          -5.39191626919101     -1.12068721984431      0.00003085999043
  C          -4.28029924910402     -1.98367774440291      0.00001572413530
  C          -2.88654169853931     -2.17544380114287     -0.00001333178762
  C          -1.58327707557522     -1.64432671358718     -0.00002732913110
  C          -0.72041042084029     -0.53316120146896     -0.00001030841882
  H          -5.52407865713236      3.55226579396499     -0.00006443338712
  H          -3.70056400685787      4.29521336485916     -0.00004753208511
  H          -1.75038559958518      4.02698260717779      0.00001813046439
  H          -0.19529092247120      2.81949876577456      0.00006079492007
  H           0.27964291117843     -0.95421179693235     -0.00002566109014
  H           0.54787075469918      0.99652536832461      0.00003549015617
  H          -2.75110411296838     -3.25200936029059     -0.00003902592004
  H          -0.92774927417810     -2.50897518860421     -0.00005637824573
  H          -6.25650762449024     -1.77632004195935      0.00006083841545
  H          -4.70138472725839     -2.98374151330937      0.00000441222650
  H          -6.73152692624161      1.99742823959431     -0.00001184150047
  H          -6.99962235005951      0.04655197698873      0.00003673622819
'''

mol = gto.M(atom=atoms, basis='6-31G', verbose=4, symmetry=False, max_memory=8000) # mem in MB
mf = scf.RHF(mol).run(conv_tol=1E-14)

num_threads = 128
scratch = os.path.join('/scratch/mm207/block2/DMRG/c12', os.environ['SLURM_JOB_ID'])

# [Part A] split localization
mf.mo_coeff, mf.mo_occ, mf.mo_energy, nactorb, nactelec = sort_orbitals(mol, mf.mo_coeff, mf.mo_occ, mf.mo_energy,
    nactorb=mol.nao, nactelec=mol.nelectron, do_loc=True, split_high=1.0, split_low=1.0)
ncas, n_elec, spin, ecore, h1e, g2e, orb_sym = itg.get_rhf_integrals(mf, ncore=0, ncas=None, g2e_symm=8)
driver = DMRGDriver(scratch=scratch, symm_type=SymmetryTypes.SU2, n_threads=num_threads, stack_mem=100 << 30)
driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, reorder='fiedler', algo_type=MPOAlgorithmTypes.Conventional, iprint=1)

# [Part B] forward schedule
bond_dims = [250] * 8 + [500] * 8 + [1000] * 4 + [1500] * 4 + [2000] * 4 + [2500] * 8
noises = [1E-5] * (len(bond_dims) - 6) + [0] * 6
thrds = [1E-7] * (len(bond_dims) - 4) + [1E-8] * 4
ket = driver.get_random_mps(tag="KET", bond_dim=bond_dims[0], nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=len(bond_dims), bond_dims=bond_dims, noises=noises, thrds=thrds,
    twosite_to_onesite=len(bond_dims) - 4, iprint=2)
print('DMRG energy = %20.15f' % energy)

# [Part C] reverse schedule
bond_dims = [2000] * 4 + [1800] * 4 + [1600] * 4 + [1400] * 4 + [1200] * 4 + [1000] * 4
noises = [0] * len(bond_dims)
thrds = [1E-8] * len(bond_dims)
ket2 = driver.copy_mps(ket, tag='KET2')
ket2 = driver.adjust_mps(ket2, dot=2)[0]
energy = driver.dmrg(mpo, ket2, n_sweeps=len(bond_dims), bond_dims=bond_dims, noises=noises, thrds=thrds, tol=0, iprint=2)

# [Part D] extrapolation
ds, dws, eners = driver.get_dmrg_results()
print('BOND DIMS         = ', ds[3::4])
print('Discarded Weights = ', dws[3::4])
print('Energies          = ', eners[3::4, 0])
reg = scipy.stats.linregress(dws[3::4], eners[3::4, 0])
emin, emax = min(eners[3::4, 0]), max(eners[3::4, 0])
print('DMRG energy (extrapolated) = %20.15f +/- %15.10f' % (reg.intercept, abs(reg.intercept - emin) / 5))

# [Part E] plot extrapolation
import matplotlib.pyplot as plt
from matplotlib import ticker
de = emax - emin
x_reg = np.array([0, dws[-1] + dws[3]])
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1E}"))
plt.plot(x_reg, reg.intercept + reg.slope * x_reg, '--', linewidth=1, color='#426A5A')
plt.plot(dws[3::4], eners[3::4, 0], ' ', marker='s', mfc='white', mec='#426A5A', color='#426A5A', markersize=5)
plt.text(dws[3] * 0.25, emax + de * 0.1, "$E(M=\\infty) = %.6f \\pm %.6f \\mathrm{\\ Hartree}$" %
    (reg.intercept, abs(reg.intercept - emin) / 5), color='#426A5A', fontsize=12)
plt.text(dws[3] * 0.25, emax - de * 0.0, "$R^2 = %.6f$" % (reg.rvalue ** 2), color='#426A5A', fontsize=12)
plt.xlim((0, dws[-1] + dws[3]))
plt.ylim((reg.intercept - de * 0.1, emax + de * 0.2))
plt.xlabel("Largest Discarded Weight")
plt.ylabel("Sweep Energy (Hartree)")
plt.subplots_adjust(left=0.17, bottom=0.1, right=0.95, top=0.95)
plt.savefig('test.png', dpi=600)
