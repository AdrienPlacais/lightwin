%************************************************************************
% Matrice Transfert dans une cavité 
% -> Calcul du gain en energie et de la Phase de la part. synchrone
% -> Tentaive pr calculer numeriquement Mtransfer longitudinale (z,delta)
% technique de tracewin normalement
%*************************************************************************

clear all
tic
% ****************
% Pour la cavité *
% ****************

% Constantes 

c=299792458;
mp = 938.27202900; % masse proton en Mev
q=1;
Ncell=2;

% frequence RF
f= 352.2*10^6;
w0 = 2*pi*f;

% phase RF (à rentrer en degrés)
phiRF = 142.089 ;
phi= phiRF *pi/180;

% Energie cinetique (en MeV)
Wc0 = 18.793905 ; betaTwin = 0.1972014;
beta_init = sqrt((1+Wc0/mp)^2-1)/(1+Wc0/mp);
str = [' comparaison beta beta_calculé - betaTwin = '];
disp(str);
comp = beta_init-betaTwin
gamma_init = 1/sqrt(1-beta_init^2);

% coeff multiplicateur sur le champ
k=1.68927*1.000092819734090;     %AP: where does the second factor comes from?
%k=2.77692*1.2688;

% import de la carte de champ et calcul de dE/dz
Field=importdata('spoke.txt');
Ez= k*Field.data(:,4); 
zmat=(Field.data(:,1) - min(Field.data(:,1))) ; %on recupère z en s'assurant qu'on commence a z=0 
step = zmat(10)-zmat(9);
dE_dz = diff(Ez)/step;
clear step
Li=length(dE_dz);
dE_dz(Li+1,1)=dE_dz(Li,1);
clear Li

% *****************************
% Paramètres de la simulation *
% *****************************

%taille de pas initial calculé en fonction du betalambda
facteur = 50;
lambda = c/f;
Pas = beta_init*lambda/(1*2*Ncell*facteur);




% ========================================================================
% Boucle pour calcul du gain en energie et phase pour la synchrone
% ========================================================================
%*******************
% Pas de calcul
% ******************

% valeur maximum de z
Zmax = max(zmat);
%update pas
Imax=fix(Zmax/Pas);
%Pas = Zmax/Imax; 


%*****************************************
% Initialisation particule synchrone
%*****************************************

Wc = Wc0;

z=Pas/2;
Er = 0;
Ei = 0;
time=Pas/(2*beta_init*c);
%phi = phi+(Pas*w0)/(2*beta_init*c);

Z=[0];
Ener=[Wc];
Betas=[beta_init];
beta = beta_init;
gamma = gamma_init;


%*****************************************
% Initialisation matrice transfert
%*****************************************

Mt = eye(2);

%*******************************************
% c'est parti pour la boucle 
% ******************************************


for i=1:(Imax-1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Particule Synchrone
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Position, reference et  energie en entrée
 
    Win = Wc;
    gammain =gamma;
    

 % calcul gain energie 
    Eint = interp1(zmat,Ez,z);
    Er = Er + q*Eint*cos(phi);
    Ei = Ei + q*Eint*sin(phi);
    Wc = q*Eint*cos(phi)*Pas + Win; % NRgie ne sortie
   gamma = gammain + q*Eint*cos(phi)*Pas/mp; % gamma en sortie
   gammas = (gamma+gammain)/2; % gamma synchrone~ gamma 'moyen'
   betas = sqrt(1-1/gammas^2);
   
    %pour calcul de le phase synchrone 
    phis=atan(Ei/Er)*180/pi;
    
   
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calcul matrice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
Matt = interp1(zmat,dE_dz,z)*cos(phi);
Steve = (q*Matt*Pas)/(gammas*betas^2*mp);
Izzy = 1 - (q*(2-betas^2)*Eint*cos(phi)*Pas)/(gammas*betas^2*mp);

AXL = [1 0;Steve Izzy];
SLASH = [1 Pas/(2*gamma^2);0 1];
DUFF = [1 Pas/(2*gammain^2);0 1];

Mt = SLASH*AXL*DUFF *Mt;
 
        
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% passage au pas suivant 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 z=z+Pas;   
 time = time+Pas/(betas*c);
 phi = phi+(Pas*w0)/(betas*c);
 Z= horzcat(Z,z) ;
 Ener=horzcat(Ener,Wc);
 Betas = horzcat(Betas, beta);   %AP: beta not updated in loop?

end
%Energie de sortie de lasynchrone 
Wcout_Synch = Wc;
Beta_synch = beta;   %AP: beta not updated in loop?
Gamma_synch = 1/sqrt(1-Beta_synch^2);
%Phase synchrone
Phase_synch =phis
% Accelerating Voltage 
Vcav = abs((Wc-Wc0)/cos(phis*pi/180))
% phase RF de la synchrone en sorite
phiRF_s_out = phi;

% matrice de transfert
Mt

MTdiff = Mt-[0.90134141 0.38549457;-0.34179967 0.95212012]
toc
