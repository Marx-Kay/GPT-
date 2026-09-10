!     Last change:  FG   29 Nov 1999    2:49 pm
SUBROUTINE  randn(n,rand_n)
IMPLICIT NONE

INTEGER, INTENT(IN) :: n
INTEGER :: ind1
real :: v1, v2, rsq, fac, rand_s
REAL, INTENT(OUT) :: rand_n(n,1)

do ind1=1,n
1   call random_number(rand_s)
   v1 = 2.0*rand_s-1.0
   call random_number(rand_s)
   v2 = 2.0*rand_s-1.0
   rsq = v1**2+v2**2
   IF(rsq.ge.1 .OR. rsq.eq.0) GOTO 1
   fac=SQRT(-2.0*LOG(rsq)/rsq)
   rand_n(ind1,1) =v1*fac
end do

RETURN
END subroutine

