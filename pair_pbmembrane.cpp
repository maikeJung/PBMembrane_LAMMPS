/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Hongyan Yuan (URI), June, 2015
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_pbmembrane.h"
#include "math_extra.h"
#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "integrate.h"
#include "citeme.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

static const char cite_pair_pbmembrane[] =
  "pair pbmembrane command:\n\n";

/* ---------------------------------------------------------------------- */

Pairpbmembrane::Pairpbmembrane(LAMMPS *lmp) : Pair(lmp)
{
  if (lmp->citeme) lmp->citeme->add(cite_pair_pbmembrane);

  single_enable = 0;
  writedata = 1;
}

/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

Pairpbmembrane::~Pairpbmembrane()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(cut);
    memory->destroy(zeta);
    memory->destroy(mu);
    memory->destroy(beta);

  }
}

/* ---------------------------------------------------------------------- */

void Pairpbmembrane::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double evdwl,one_eng,rsq,forcelj,factor_lj;
  double fforce[3],ttor[3],rtor[3],r12[3];
  double a1[3][3],a2[3][3];
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *iquat,*jquat;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  AtomVecEllipsoid::Bonus *bonus = avec->bonus;
  int *ellipsoid = atom->ellipsoid;
  double **x = atom->x;
  double **f = atom->f;
  double **tor = atom->torque;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    iquat = bonus[ellipsoid[i]].quat;
    MathExtra::quat_to_mat_trans(iquat,a1);
      
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      // r12 = center to center vector

      r12[0] = x[j][0]-x[i][0];
      r12[1] = x[j][1]-x[i][1];
      r12[2] = x[j][2]-x[i][2];
      rsq = MathExtra::dot3(r12,r12);
      jtype = type[j];

      // compute if less than cutoff

      if (rsq < cutsq[itype][jtype]) {

        jquat = bonus[ellipsoid[j]].quat;
        MathExtra::quat_to_mat_trans(jquat,a2);  
        one_eng = pbmembrane_analytic(i,j,a1,a2,r12,rsq,fforce,ttor,rtor);

        fforce[0] *= factor_lj;
        fforce[1] *= factor_lj;
        fforce[2] *= factor_lj;
        ttor[0] *= factor_lj;
        ttor[1] *= factor_lj;
        ttor[2] *= factor_lj;

        f[i][0] += fforce[0];
        f[i][1] += fforce[1];
        f[i][2] += fforce[2];
        tor[i][0] += ttor[0];
        tor[i][1] += ttor[1];
        tor[i][2] += ttor[2];

        if (newton_pair || j < nlocal) {
          rtor[0] *= factor_lj;
          rtor[1] *= factor_lj;
          rtor[2] *= factor_lj;
          f[j][0] -= fforce[0];
          f[j][1] -= fforce[1];
          f[j][2] -= fforce[2];
          tor[j][0] += rtor[0];
          tor[j][1] += rtor[1];
          tor[j][2] += rtor[2];
        }

        if (eflag) evdwl = factor_lj*one_eng;

        if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,
                                 evdwl,0.0,fforce[0],fforce[1],fforce[2],
                                 -r12[0],-r12[1],-r12[2]);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
 
  
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void Pairpbmembrane::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(zeta,n+1,n+1,"pair:zeta");
  memory->create(mu,n+1,n+1,"pair:mu");
  memory->create(beta,n+1,n+1,"pair:beta");
  
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void Pairpbmembrane::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");
 
  cut_global = force->numeric(FLERR,arg[0]);

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void Pairpbmembrane::coeff(int narg, char **arg)
{
  if (narg < 8 || narg > 8)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);
  double cut_one = force->numeric(FLERR,arg[4]);
  double zeta_one = force->numeric(FLERR,arg[5]);
  double mu_one = force->numeric(FLERR,arg[6]);
  double beta_one = force->numeric(FLERR,arg[7]);

  
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      zeta[i][j] = zeta_one;
      mu[i][j] = mu_one;
      beta[i][j] = beta_one;

      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void Pairpbmembrane::init_style()
{
  avec = (AtomVecEllipsoid *) atom->style_match("ellipsoid");
  if (!avec) error->all(FLERR,"Pair pbmembrane requires atom style ellipsoid");

  neighbor->request(this,instance_me);
 
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double Pairpbmembrane::init_one(int i, int j)
{

  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }


  epsilon[j][i] = epsilon[i][j];
  sigma[j][i] = sigma[i][j];
  zeta[j][i] = zeta[i][j];
  mu[j][i] = mu[i][j];
  beta[j][i] =  beta[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void Pairpbmembrane::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++) {
    
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
        fwrite(&zeta[i][j],sizeof(double),1,fp);
        fwrite(&mu[i][j],sizeof(double),1,fp);
        fwrite(&beta[i][j],sizeof(double),1,fp);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void Pairpbmembrane::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  
  
  for (i = 1; i <= atom->ntypes; i++) {
    
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
          fread(&zeta[i][j],sizeof(double),1,fp);
          fread(&mu[i][j],sizeof(double),1,fp);
          fread(&beta[i][j],sizeof(double),1,fp);
          
          
        }
        
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&zeta[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&mu[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&beta[i][j],1,MPI_DOUBLE,0,world);
        
      }
    }
  }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void Pairpbmembrane::write_restart_settings(FILE *fp)
{
  //fwrite(&model_type,sizeof(int),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void Pairpbmembrane::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
      
    //fread(&model_type,sizeof(int),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  
  //MPI_Bcast(&model_type,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void Pairpbmembrane::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g %g %g\n",i,
            epsilon[i][i],sigma[i][i],cut[i][i],zeta[i][i],mu[i][i], beta[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void Pairpbmembrane::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g %g\n",i,j,
              epsilon[i][i],sigma[i][i],cut[i][j],zeta[i][j],mu[i][j], beta[i][j]);
}




/* ----------------------------------------------------------------------
   compute analytic energy, force (fforce), and torque (ttor & rtor)
   based on rotation matrices a 
   if newton is off, rtor is not calculated for ghost atoms
------------------------------------------------------------------------- */

double Pairpbmembrane::pbmembrane_analytic(const int i,const int j,double a1[3][3],
                                       double a2[3][3], double *r12,
                                       const double rsq, double *fforce,
                                       double *ttor, double *rtor)

{ 
    int *type = atom->type;
    int newton_pair = force->newton_pair;
    int nlocal = atom->nlocal;

    double r12hat[3];
    MathExtra::normalize3(r12,r12hat);
    double r = sqrt(rsq);


    double ni1[3],nj1[3],dphi_drhat[3],dUdrhat[3],dUdni1[3],dUdnj1[3],dphi_dni1[3],dphi_dnj1[3] ; 
    double t,t1,t2,t4,cos_t,U,uR,uA,dUdr,dUdphi;
    double pi = 3.141592653589793, pow2_1by6 = 1.122462048309373;

    double energy_well = epsilon[type[i]][type[j]];
    double rmin = pow2_1by6*sigma[type[i]][type[j]];
    double rcut = cut[type[i]][type[j]];        
    double zt = zeta[type[i]][type[j]];
    double muu = mu[type[i]][type[j]];        
    double sint = beta[type[i]][type[j]];
   
    ni1[0]=a1[0][0];
    ni1[1]=a1[0][1];
    ni1[2]=a1[0][2];
    
    nj1[0]=a2[0][0];
    nj1[1]=a2[0][1];
    nj1[2]=a2[0][2];
    
    double ninj   = MathExtra::dot3(ni1,nj1);
    double ni1rhat = MathExtra::dot3(ni1,r12hat);
    double nj1rhat = MathExtra::dot3(nj1,r12hat);
    
    double a  = ninj + (sint-ni1rhat)*(sint+nj1rhat) - 2.0*sint*sint;
    double phi = 1.0 + (a-1.0)*muu;
    
    dphi_drhat[0] = muu*( (sint-ni1rhat)*nj1[0]-ni1[0]*(sint+nj1rhat) );
    dphi_drhat[1] = muu*( (sint-ni1rhat)*nj1[1]-ni1[1]*(sint+nj1rhat) );
    dphi_drhat[2] = muu*( (sint-ni1rhat)*nj1[2]-ni1[2]*(sint+nj1rhat) );
    
    dphi_dni1[0] = muu*(nj1[0]-r12hat[0]*(sint+nj1rhat));
    dphi_dni1[1] = muu*(nj1[1]-r12hat[1]*(sint+nj1rhat));
    dphi_dni1[2] = muu*(nj1[2]-r12hat[2]*(sint+nj1rhat));
    
    dphi_dnj1[0] = muu*(ni1[0]+r12hat[0]*(sint-ni1rhat));
    dphi_dnj1[1] = muu*(ni1[1]+r12hat[1]*(sint-ni1rhat));
    dphi_dnj1[2] = muu*(ni1[2]+r12hat[2]*(sint-ni1rhat));    

    if (r < rmin)
    {
        t = rmin/r;
        t2=t*t;
        t4=t2*t2;
        uR = (t4 - 2.0*t2)*energy_well;
        
        U = uR +(1.0 - phi)*energy_well;
        
        dUdr = 4.0*(t2-t4)/r*energy_well;
      
        dUdphi = -energy_well;
    }
    
    
    else
    {
      
        t = pi/2.0*(r-rmin)/(rcut-rmin);
        cos_t = cos(t);
        t1=cos_t;

        for (int k = 1; k <= 2*zt-2; k++) {t1 *= cos_t;}  // get cos()^(2zt-1)
        
        uA = -energy_well*t1*cos_t; 
        
        U = uA*phi;
        
        dUdr = zt*pi/(rcut-rmin)*(t1)*sin(t)*phi*energy_well;
        
        dUdphi = uA;
             
    }
        
    dUdrhat[0] = dUdphi*dphi_drhat[0];
    dUdrhat[1] = dUdphi*dphi_drhat[1];
    dUdrhat[2] = dUdphi*dphi_drhat[2];
    
    double dUdrhatrhat = MathExtra::dot3(dUdrhat,r12hat);
    
    fforce[0] = dUdr*r12hat[0] + (dUdrhat[0]-dUdrhatrhat*r12hat[0])/r;
    fforce[1] = dUdr*r12hat[1] + (dUdrhat[1]-dUdrhatrhat*r12hat[1])/r;
    fforce[2] = dUdr*r12hat[2] + (dUdrhat[2]-dUdrhatrhat*r12hat[2])/r;
    
    // torque i
    dUdni1[0] = dUdphi*dphi_dni1[0];
    dUdni1[1] = dUdphi*dphi_dni1[1];
    dUdni1[2] = dUdphi*dphi_dni1[2];
    
    MathExtra::cross3(dUdni1,ni1,ttor);  //minus sign is replace by swapping ni1 and dUdni1
    
    if (newton_pair || j < nlocal) {
        
        dUdnj1[0] = dUdphi*dphi_dnj1[0];
        dUdnj1[1] = dUdphi*dphi_dnj1[1];
        dUdnj1[2] = dUdphi*dphi_dnj1[2];
        
        MathExtra::cross3(dUdnj1,nj1,rtor);  //minus sign is replace by swapping ni1 and dUdni1
    }
    
    
  // output energy, force, torque, only for the in_two_particles input file, for checking the model implementation 
  /*
  fprintf(screen,"energy = %f \n",U);
  fprintf(screen,"force_i  = %f %f %f \n",fforce[0],fforce[1],fforce[2]);
  fprintf(screen,"torque_i  = %f %f %f \n",ttor[0],ttor[1],ttor[2]);
  fprintf(screen,"torque_j = %f %f %f \n",rtor[0],rtor[1],rtor[2]);
  */
    
    return U;  
  
}



