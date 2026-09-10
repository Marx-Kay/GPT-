!     Last change:  FG   13 Apr 2001    4:06 pm
subroutine sc_splint(xa,ya,y2a,n,x,y)
!for scalars
INTEGER, INTENT(IN) :: n
REAL, INTENT(IN) :: x
REAL, INTENT(IN) :: xa(n,1), ya(n,1), y2a(n,1)
double precision, INTENT(OUT) :: y
INTEGER :: k, khi, klo
REAL :: a, b, h
       klo = 1
       khi = n
       do WHILE (khi-klo>1)
          k=(khi+klo)/2
          if (xa(k,1)>x) then
            khi=k
          else
            klo=k
          end if
       END do
    h = xa(khi,1)-xa(klo,1)
    a = (xa(khi,1)-x)/h
    b = (x-xa(klo,1))/h
    y = a*ya(klo,1)+b*ya(khi,1)+((a**3-a)*y2a(klo,1)+(b**3-b)*y2a(khi,1))*(h**2)/6.0
    if (y>ya(n,1)) then
           y = ya(n,1)
   else if (y<ya(1,1)) then
       y = ya(1,1)
   end if
!    if (y<ya(1,1)) then
!       y = ya(1,1)
!   end if
end subroutine
