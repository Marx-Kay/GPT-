!     Last change:  FG    4 Aug 2003    2:30 pm
PROGRAM lifecycle
IMPLICIT NONE

!!!!!!!!!!!!!!!!!!!!
! DEFINE VARIABLES !
!!!!!!!!!!!!!!!!!!!!

CHARACTER(LEN=6), DIMENSION(80,1) :: filename
CHARACTER(LEN=1) :: t_out
CHARACTER(LEN=2) :: t_out2
INTEGER :: i1, i2, i3, i4, i5, i6, i7, i8, i9
INTEGER :: index_out
INTEGER :: aux2
INTEGER :: tb, t1, tr, td, tn, tn1,t
parameter (tb=20, tr=66, td=100, tn=td-tb+1)
INTEGER :: na, ncash, n, nc
parameter (na=16, ncash=51, n=5, nc=11)
INTEGER, DIMENSION(1) :: pt
REAL :: maxcash=50.0, mincash=0.25
REAL :: l_maxcash, l_mincash, stepcash
REAL :: stepc, maxc, minc, mpc
REAL :: aa=-2.170042+2.700381, b1=0.16818, b2=-0.0323371/10, b3=0.0019704/100
REAL :: ret_fac = 0.68212
REAL :: smay=0.1, smav=0.1, corr_v=0.0, corr_y=0.0  
REAL :: rho=5.0, delta=0.97, psi=0.5, theta, psi_1, psi_2
REAL :: r=1.015, mu=0.04, sigr=0.2
REAL :: cash_1, riskret0
double precision :: auxVV, u, int_V
INTEGER :: i_net_cash, i_net_cash2, ic1, ic2
REAL :: ttc
REAL, DIMENSION(tn-1,1) :: survprob=0.0 , delta2=0.0
REAL, DIMENSION(n,1) :: grid, weig, gret, ones_n_1=1.0, grid2
REAL, DIMENSION(n,n) :: yp
REAL, DIMENSION(n,n) :: yh
REAL, DIMENSION(n,n,n) :: nweig1
REAL, DIMENSION(tr-tb+1,1) :: f_y=0.0
REAL, DIMENSION(tr-tb,1) :: gy=0.0
REAL, DIMENSION(n,n,tn-1) :: gyp=0.0
REAL, DIMENSION(ncash,1) :: gcash, lgcash
REAL, DIMENSION(na,1) :: ga
REAL, DIMENSION(na,n) :: riskret
REAL, DIMENSION(nc,1) :: gc
double precision, DIMENSION(na,nc) :: auxV
double precision, DIMENSION(na*nc,1) :: vec_V
real, DIMENSION(ncash,1) :: secd
REAL, DIMENSION(ncash,tn) :: C=0.0, V
REAL, DIMENSION(ncash,tn) :: A=1.0

INTEGER :: w, nsim
REAL :: cash, sav, partcount
REAL, DIMENSION(1,1) :: eps_r, simTY, eps_y
parameter (nsim=10000)
REAL, DIMENSION(nsim,1) :: ones_nsim_1=1.0
REAL, DIMENSION(tn,1) :: meanY, meanC, meanW, meanA, meanS, meanB, meanWY, meanalpha, meanGPY
REAL, DIMENSION(tn,1) :: cGPY, meanYs, meanCs, meanWs
REAL, DIMENSION(tn,nsim) :: simPY, simGPY, simY, simC, simW=0.0, simA, simS, simB, simW_Y=0.0, simR

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! APPROXIMATION TO NORMAL DISTRIBUTION  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
grid(1,1) = -2.85697001387280    
grid(2,1) =  -1.35562617997427   
grid(3,1) =    0.00000000000000       
grid(4,1) =    1.35562617997426       
grid(5,1) =    2.85697001387280  

weig(1,1) = 0.01125741132772  
weig(2,1) = 0.22207592200561 
weig(3,1) = 0.53333333333333 
weig(4,1) = 0.22207592200561
weig(5,1) = 0.01125741132772  

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! CONDITIONAL SURVIVAL PROBABILITIES  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
survprob(1,1) = 0.99845
survprob(2,1) = 0.99839
survprob(3,1) = 0.99833
survprob(4,1) = 0.9983
survprob(5,1) = 0.99827
survprob(6,1) = 0.99826
survprob(7,1) = 0.99824
survprob(8,1) = 0.9982
survprob(9,1) = 0.99813
survprob(10,1) = 0.99804
survprob(11,1) = 0.99795
survprob(12,1) = 0.99785
survprob(13,1) = 0.99776
survprob(14,1) = 0.99766
survprob(15,1) = 0.99755
survprob(16,1) = 0.99743
survprob(17,1) = 0.9973
survprob(18,1) = 0.99718
survprob(19,1) = 0.99707
survprob(20,1) = 0.99696
survprob(21,1) = 0.99685
survprob(22,1) = 0.99672
survprob(23,1) = 0.99656
survprob(24,1) = 0.99635
survprob(25,1) = 0.9961
survprob(26,1) = 0.99579
survprob(27,1) = 0.99543
survprob(28,1) = 0.99504
survprob(29,1) = 0.99463
survprob(30,1) = 0.9942
survprob(31,1) = 0.9937
survprob(32,1) = 0.99311
survprob(33,1) = 0.99245
survprob(34,1) = 0.99172
survprob(35,1) = 0.99091
survprob(36,1) = 0.99005
survprob(37,1) = 0.98911
survprob(38,1) = 0.98803
survprob(39,1) = 0.9868
survprob(40,1) = 0.98545
survprob(41,1) = 0.98409
survprob(42,1) = 0.9827
survprob(43,1) = 0.98123
survprob(44,1) = 0.97961
survprob(45,1) = 0.97786
survprob(46,1) = 0.97603
survprob(47,1) = 0.97414
survprob(48,1) = 0.97207
survprob(49,1) = 0.9697
survprob(50,1) = 0.96699
survprob(51,1) = 0.96393
survprob(52,1) = 0.96055
survprob(53,1) = 0.9569
survprob(54,1) = 0.9531
survprob(55,1) = 0.94921
survprob(56,1) = 0.94508
survprob(57,1) = 0.94057
survprob(58,1) = 0.9357
survprob(59,1) = 0.93031
survprob(60,1) = 0.92424
survprob(61,1) = 0.91717
survprob(62,1) = 0.90922
survprob(63,1) = 0.90089
survprob(64,1) = 0.89282
survprob(65,1) = 0.88503
survprob(66,1) = 0.87622
survprob(67,1) = 0.86576
survprob(68,1) = 0.8544
survprob(69,1) = 0.8423
survprob(70,1) = 0.82942
survprob(71,1) = 0.8154
survprob(72,1) = 0.80002
survprob(73,1) = 0.78404
survprob(74,1) = 0.76842
survprob(75,1) = 0.75382
survprob(76,1) = 0.73996
survprob(77,1) = 0.72464
survprob(78,1) = 0.71057
survprob(79,1) = 0.6961
survprob(80,1) = 0.6809

!!!!!!!!!!!!!!!!!!!!!!!!
! OUTPUT FILES - NAMES !
!!!!!!!!!!!!!!!!!!!!!!!!
10 format (I1.1)
20 format (I2.2)
do t=1,tn-1
   if (t<10) then
      write(t_out,10) t
      filename(t,1)= 'year0' // t_out
   else
      write(t_out2,20) t
      filename(t,1)= 'year' // t_out2   
   end if   
end do

!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ADDITIONAL COMPUTATIONS !
!!!!!!!!!!!!!!!!!!!!!!!!!!!
do i1=1,n
   gret(i1,1) = r+mu+grid(i1,1)*sigr
end do 
do i6=1,n
   do i7=1,n
      do i8=1,n
         nweig1(i6,i7,i8) = weig(i6,1)*weig(i7,1)*weig(i8,1)
      end do
   end do
end do

theta = (1.0-rho)/(1.0-1.0/psi)
psi_1 = 1.0-1.0/psi
psi_2 = 1.0/psi_1 

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! GRIDS FOR THE STATE VARIABLES AND FOR PORTFOLIO RULE !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
DO i1=1,na
   ga(i1,1)=(na-i1)/(na-1.0)
END DO

do i5=1,na
   do i8=1,n
      riskret(i5,i8)=r*(1-ga(i5,1))+gret(i8,1)*ga(i5,1)
   end do   !i8
end do    !i5

l_maxcash = log(maxcash)
l_mincash = log(mincash)
stepcash = (l_maxcash-l_mincash)/(ncash-1)
DO i1=1,ncash
   lgcash(i1,1)=l_mincash+(i1-1.0)*stepcash
END DO
DO i1=1,ncash
   gcash(i1,1)=exp(lgcash(i1,1))
END DO

!!!!!!!!!!!!!!!!
! LABOR INCOME !
!!!!!!!!!!!!!!!!
do i1=1,n
   grid2(:,1) = grid(i1,1)*corr_y+grid(:,1)*ones_n_1(:,1)*(1-corr_y**2)**(0.5)
   yh(1:n,i1) = EXP(grid2(:,1)*smay)
end do

do i1=1,n
   grid2(:,1) = grid(i1,1)*corr_v+grid(:,1)*ones_n_1(:,1)*(1-corr_v**2)**(0.5)
   yp(:,i1) = grid2(:,1)*smav
end do

DO i1=tb,tr
   f_y(i1-tb+1,1) = EXP(aa+b1*i1+b2*i1**2+b3*i1**3)
END DO

do i1=tb,tr-1
   gy(i1-tb+1,1) = f_y(i1-tb+2,1)/f_y(i1-tb+1,1)-1.0
   do i2=1,n
      gyp(:,i2,i1-tb+1) = EXP(gy(i1-tb+1,1)*ones_n_1(:,1)+yp(:,i2))
   end do
end do

do i1=tr-tb+1,tn-1
   do i2=1,n
      gyp(:,i2,i1) = EXP(0.0*ones_n_1(:,1))
   end do
end do


!!!!!!!!!!!!!!!!!!!
! TERMINAL PERIOD !
!!!!!!!!!!!!!!!!!!!
do i1=1,ncash
   C(i1,tn) = gcash(i1,1)
end do
A(:,tn) = 0.0
do i1=1,ncash
   V(i1,tn) = C(i1,tn)*((1.0-delta)**(psi/(psi-1.0)))
end do

call spline(gcash(:,1),V(:,tn),ncash,1.0,secd(:,1))

!!!!!!!!!!!!!!!!!!!!!!
! RETIREMENT PERIODS !
!!!!!!!!!!!!!!!!!!!!!!
DO i1= 1,td-tr
   t= tn-i1
   write(*,*) t
   do i3=1,ncash
      if (i3.eq.1) then
          maxc = C(i3,t+1)
          minc = maxc/2
      else
         minc = C(i3-1,t)
         if (i3<10) then
            maxc = minc + (gcash(i3,1) - gcash(i3-1,1))
        else
           mpc = max((C(i3-1,t)-C(i3-9,t))/(gcash(i3-1,1) - gcash(i3-9,1)),0.1)
           maxc = minc + mpc*(gcash(i3,1) - gcash(i3-1,1))      
        end if         
      end if
       stepc=(maxc-minc)/(nc-1)
       DO i9=1,nc
          gc(i9,1)=minc+(i9-1.0)*stepc
       END DO
       do i4=1,nc
         u=(1.0-delta)*(gc(i4,1)**psi_1) !1-1/psi
         sav = gcash(i3,1)-gc(i4,1)
         do i5=1,na
            auxVV=0.0
!            do i6=1,n
!               do i7=1,n
                  do i8=1,n
                     cash_1= riskret(i5,i8)*sav+ret_fac
                     cash_1 = max(min(cash_1,gcash(ncash,1)),gcash(1,1))
                     call sc_splint(gcash(:,1),V(:,t+1),secd(:,1),ncash,cash_1,int_V)
                     auxVV=auxVV+weig(i8,1)*survprob(t,1)*(int_V**(1.0-rho))
                  end do   !i8
!               end do      !i7
!            end do         !i6
            auxV(i5,i4) = (u+delta*(auxVV**(1.0/theta)))**psi_2    !1/(1-1/psi)
         end do    !i5
      end do   !i4
      vec_V = RESHAPE((auxV),(/na*nc,1/))
      V(i3,t) = MAXVAL(vec_V(:,1))
      pt = MAXLOC(vec_V(:,1))
      aux2 = FLOOR((REAL(pt(1))-1)/REAL(na))
      C(i3,t) = gc(aux2+1,1)
      A(i3,t) = ga(pt(1)-na*aux2,1)
    end do    !i3

   call spline(gcash(:,1),V(:,t),ncash,1.0,secd(:,1))

  OPEN(UNIT=1,FILE=filename(t,1),STATUS='replace',ACTION='write')
  do i5=1,ncash
    WRITE(1,*) A(i5,t), gcash(i5,1)
  end do
  do i5=1,ncash
	WRITE(1,*) C(i5,t), gcash(i5,1)
  end do
  do i5=1,ncash
	WRITE(1,*) V(i5,t), gcash(i5,1)
  end do
  CLOSE(UNIT=1)

end do    !i1

!!!!!!!!!!!!!!!!!
! OTHER PERIODS !
!!!!!!!!!!!!!!!!!
DO i1= 1,tr-tb
   t= tr-tb-i1+1
   write(*,*) t
   do i3=1,ncash
      if (i3.eq.1) then
          minc = gcash(i3,1)/5
          maxc = 0.999*gcash(i3,1)
      else
          minc = C(i3-1,t)
         if (i3<10) then
            maxc = minc + (gcash(i3,1) - gcash(i3-1,1))
         else
            mpc = max((C(i3-1,t)-C(i3-9,t))/(gcash(i3-1,1) - gcash(i3-9,1)),0.1)
            maxc = minc + mpc*(gcash(i3,1) - gcash(i3-1,1))      
         end if         
      end if
       stepc=(maxc-minc)/(nc-1)
       DO i9=1,nc
          gc(i9,1)=minc+(i9-1.0)*stepc
       END DO
       do i4=1,nc
         u=(1.0-delta)*(gc(i4,1)**psi_1)       !1-1/psi
         sav = gcash(i3,1)-gc(i4,1)
         do i5=1,na
            auxVV=0.0
            do i6=1,n
               do i8=1,n
                  do i7=1,n
                     cash_1= riskret(i5,i8)*sav/gyp(i6,i8,t)+yh(i7,i8)
                     cash_1 = max(min(cash_1,gcash(ncash,1)),gcash(1,1))                     
                     call sc_splint(gcash(:,1),V(:,t+1),secd(:,1),ncash,cash_1,int_V)
                     auxVV=auxVV+nweig1(i6,i7,i8)*survprob(t,1)*((int_V*gyp(i6,i8,t))**(1.0-rho))
                  end do   !i8
               end do      !i7
            end do         !i6
            auxV(i5,i4) = (u+delta*(auxVV**(1.0/theta)))**psi_2    !1/(1-1/psi)                       
         end do    !i5
      end do   !i4
      vec_V = RESHAPE((auxV),(/na*nc,1/))
      V(i3,t) = MAXVAL(vec_V(:,1))
      pt = MAXLOC(vec_V(:,1))
      aux2 = FLOOR((REAL(pt(1))-1)/REAL(na))
      C(i3,t) = gc(aux2+1,1)
      A(i3,t) = ga(pt(1)-na*aux2,1)
   end do    !i3
 
   call spline(gcash(:,1),V(:,t),ncash,1.0,secd(:,1))
 
  OPEN(UNIT=1,FILE=filename(t,1),STATUS='replace',ACTION='write')
  do i5=1,ncash
    WRITE(1,*) A(i5,t), gcash(i5,1)
  end do
  do i5=1,ncash
	WRITE(1,*) C(i5,t), gcash(i5,1)
  end do
  do i5=1,ncash
	WRITE(1,*) V(i5,t), gcash(i5,1)
  end do
  CLOSE(UNIT=1)

end do    !i1

!!!!!!!!!!!!!!!
! SIMULATIONS !
!!!!!!!!!!!!!!!
do i1=1,nsim/2

call randn(1,eps_y(1,1))
simPY(1,i1) = eps_y(1,1)*smav
simPY(1,nsim/2+i1) = -eps_y(1,1)*smav
simGPY(1,i1) = 1.0
simGPY(1,nsim/2+i1) = 1.0
call randn(1,simTY(1,1))
simY(1,i1) = EXP(simTY(1,1)*smay)
simY(1,nsim/2+i1) = EXP(-simTY(1,1)*smay)

do i2=2,tr-tb
   w = i2+tb-1
   call randn(1,eps_y(1,1))
   simPY(i2,i1) = eps_y(1,1)*smav+simPY(i2-1,i1)
   simPY(i2,nsim/2+i1) = -eps_y(1,1)*smav+simPY(i2-1,i1)
   simGPY(i2,i1) = exp(gy(i2-1,1))*exp(simPY(i2,i1))/exp(simPY(i2-1,i1))  !gy is for next-period's income
   simGPY(i2,nsim/2+i1) = exp(gy(i2-1,1))*exp(simPY(i2,nsim/2+i1))/exp(simPY(i2-1,nsim/2+i1))
   call randn(1,simTY(1,1))
   simY(i2,i1) = EXP(simTY(1,1)*smay)
   simY(i2,nsim/2+i1) = EXP(-simTY(1,1)*smay)
end do

end do  !i1 (nsim)

do t=tr-tb+1,tn
   simY(t,:) = ret_fac
   simGPY(t,:) = 1.0
end do

do t=1,tn
do i1=1,nsim/2
   call randn(1,eps_r(1,1))
   simR(t,i1) = mu + r + sigr*eps_r(1,1)
   simR(t,nsim/2+i1) = mu + r - sigr*eps_r(1,1)
end do   
end do   


simW(:,:)=0.0
do t=1,tn
do i1=1,nsim
   simW_Y(t,i1) = simW(t,i1)/simY(t,i1)
   cash = simW(t,i1)+simY(t,i1)
   CALL ntoil(log(cash),lgcash(:,1),ncash,i_net_cash2) !need equally-spaced grid for ntoil
   ic1 = i_net_cash2
   ic2 = i_net_cash2+1 
   ttc = (cash-gcash(ic1,1))/(gcash(ic2,1)-gcash(ic1,1))
   ttc = MAX(MIN(1.0,ttc),0.0)
   simC(t,i1) = (1-ttc)*C(ic1,t)+ttc*C(ic2,t)
   simA(t,i1) = (1-ttc)*A(ic1,t)+ttc*A(ic2,t)
   simC(t,i1) = MIN(simC(t,i1),0.9999*cash)
   sav = cash-simC(t,i1)
   simS(t,i1) = simA(t,i1)*sav
   simS(t,i1) = MIN(simS(t,i1),sav)
   simB(t,i1) = sav-simS(t,i1)
   if (t<tn) then
      simW(t+1,i1) = (simB(t,i1)*r+simS(t,i1)*simR(t,i1))/simGPY(t+1,i1)
   end if

end do  !i1 (nsim)

end do  !t 

meanC(:,1) = MATMUL(simC(:,:),ones_nsim_1(:,1))/nsim
meanY(:,1) = MATMUL(simY(:,:),ones_nsim_1(:,1))/nsim
meanW(:,1) = MATMUL(simW(:,:),ones_nsim_1(:,1))/nsim
meanS(:,1) = MATMUL(simS(:,:),ones_nsim_1(:,1))/nsim
meanB(:,1) = MATMUL(simB(:,:),ones_nsim_1(:,1))/nsim
meanWY(:,1) = MATMUL(simW_Y(:,:),ones_nsim_1(:,1))/nsim
meanalpha(:,1) = MATMUL(simA(:,:),ones_nsim_1(:,1))/nsim
meanGPY(:,1) = MATMUL(simGPY(:,:),ones_nsim_1(:,1))/nsim

!meanalpha(:,i_ra,i_ies,i_beta,icht) = 0.0
!do i2=1,tsim
!partcount = 0
!   do i1=1,nsim
!   if (simA(i2,i1)>0) then
!       meanalpha(i2,i_ra,i_ies,i_beta,icht) = meanalpha(i2,i_ra,i_ies,i_beta,icht)+simA(i2,i1)
!       partcount = partcount +1 
!   end if   
!   end do  !i1 (nsim)
!   if (partcount >0) then
!       meanalpha(i2,i_ra,i_ies,i_beta,icht) = meanalpha(i2,i_ra,i_ies,i_beta,icht)/partcount
!   else
!       meanalpha(i2,i_ra,i_ies,i_beta,icht) = 1.0
!   end if
!end do !i2 (time within cht)  

OPEN(UNIT=25,FILE='CWY',STATUS='replace',ACTION='write')
do i2=1,tn
   WRITE(25,*) meanC(i2,1), meanW(i2,1), meanY(i2,1), meanGPY(i2,1)
end do
CLOSE(UNIT=25)

cGPY(1,1)=1.0
do i2=2,tn
   cGPY(i2,1) = cGPY(i2-1,1)+(meanGPY(i2,1)-1)
end do
do i2=1,tn
   meanCs(i2,1) = meanC(i2,1)*cGPY(i2,1)
   meanWs(i2,1) = meanW(i2,1)*cGPY(i2,1)
   meanYs(i2,1) = meanY(i2,1)*cGPY(i2,1)
end do

OPEN(UNIT=25,FILE='CWYs',STATUS='replace',ACTION='write')
do i2=1,tn
   WRITE(25,*) meanCs(i2,1), meanWs(i2,1), meanYs(i2,1)
end do
CLOSE(UNIT=25)

OPEN(UNIT=25,FILE='SB',STATUS='replace',ACTION='write')
do i2=1,tn
   WRITE(25,*) meanS(i2,1), meanB(i2,1), meanalpha(i2,1)
end do
CLOSE(UNIT=25)

end program
