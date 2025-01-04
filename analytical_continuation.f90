module precs
  implicit none
  integer, parameter, public :: prec=8
end module precs

program analytical_continuation
!
!    Makes and analytical continuation of the complex function
!    to arbitrary complex energy mesh
!
!   Written by A. Poteryaev <Alexander.Poteryaev _at_ cpht.polytechnique.fr>
!
  use precs
  
  implicit none

  integer :: npoints = 1000            !  Default number of points on real energy mesh
  real(prec) :: emin = -5              !  Default lower boundary for real energy mesh
  real(prec) :: emax =  5              !  Default upper boundary for real energy mesh
  real(prec) :: eta  = 0.05            !  Default imaginary offset
  
  integer :: i, ios, n_in, n_out
  real(prec) :: e1, e2, ref, imf
  complex(prec) :: fz
  complex(prec), allocatable :: z_in(:), z_out(:), f(:)

  logical :: print_usage = .false.
  logical :: real_in_energy = .false.
  logical :: imag_in_energy = .false.
  logical :: filename_out_exist = .false.
  logical :: filename_in_exist  = .false.
  logical :: emin_exist = .false.
  logical :: emax_exist = .false.
  logical :: eta_exist  = .false.

  character(64) :: argname, filename_in, filename_out

  
  do i = 1, 7
    call getarg( i, argname )
    argname = trim(adjustl(argname))

!   Check the presence of arguments
    
    if( i == 1 .and. len_trim(argname) == 0 )then
      print_usage = .true.
      write(6,*) print_usage
      exit
    end if
    
!   Input file

    if( index(argname,'-if=') == 1 )then
      filename_in = trim(adjustl(argname(5:)))
      open( 10, file=filename_in, form='formatted', status='old',    &
            iostat=ios, position='rewind' )
      
      if( len_trim(filename_in) == 0 .or. ios /= 0 )then
        print_usage = .true.
        exit
      end if
      
      filename_in_exist = .true.
    end if
    
!   Output file

    if( index(argname,'-of=') == 1 )then
      filename_out = trim(adjustl(argname(5:)))
      open( 20, file=filename_out, form='formatted', status='old',    &
            iostat=ios, position='rewind' )
      
      if( len_trim(filename_out) == 0 .or. ios /= 0 )then
        print_usage = .true.
        exit
      end if
      
      filename_out_exist = .true.
    end if
    
!   Form of input energy

    if( index(argname,'-inener=') == 1 )then
      if( index(trim(adjustl(argname(9:))),'real') == 1 )then
        real_in_energy = .true.
      elseif( index(trim(adjustl(argname(9:))),'imag') == 1 )then
        imag_in_energy = .true.
      else
        print_usage = .true.
        exit
      end if
    end if
    
!   Emin for the output energy range
   
    if( index(argname,'-emin=') == 1 )then
      argname = trim(adjustl(argname(7:)))
      read(argname,*,iostat=ios) emin
      
      if( ios /= 0 )then
        print_usage = .true.
        exit
      end if
      
      emin_exist = .true.
    end if
    
!   Emax for the output energy range
   
    if( index(argname,'-emax=') == 1 )then
      argname = trim(adjustl(argname(7:)))
      read(argname,*,iostat=ios) emax
      
      if( ios /= 0 )then
        print_usage = .true.
        exit
      end if
      
      emax_exist = .true.
    end if
    
!   Eta for the output energy range
   
    if( index(argname,'-eta=') == 1 )then
      argname = trim(adjustl(argname(6:)))
      read(argname,*,iostat=ios) eta
      
      if( ios /= 0 )then
        print_usage = .true.
        exit
      end if
   
      eta_exist = .true.
    end if
    
!   Npoints for the output energy range
   
    if( index(argname,'-npoints=') == 1 )then
      argname = trim(adjustl(argname(10:)))
      read(argname,*,iostat=ios) npoints
      
      npoints = abs(npoints)
      
      if( ios /= 0 .or. npoints == 1 )then
        print_usage = .true.
        exit
      end if
    end if
    
  end do

  if( .not.filename_out_exist .and. ( (emin_exist .or. emax_exist) &
  .and. (emin>emax) ) ) print_usage = .true.
  
!   Print help screen

  if( print_usage .or. .not.(filename_in_exist) )then
    write(6,"(A)")
    write(6,"(A)")' Analytical_continuation:  missing or wrong &
    arguments'
    write(6,"(A)")' Usage:'
    write(6,"(A)")'   analytical_continuation -if=filename1 &
    [-inener=real|imag] [-of=filename2] [-emin=emin] [-emax=emax] &
    [-eta=eta] [-npoints=n]'
    write(6,"(A)")'    filename1    - (required) input file with &
    complex energy points, z_i, and function at this point f(z_i).'
    write(6,"(A)")'                   Format of the file depends on &
    the value of -inener option (see below).'
    write(6,"(A)")'                   If -inener option is omitted &
    then filename1 must have 4 columns { Re(z_i) Im(z_i) Re(f(z_i)) &
     Im(f(z_i) }'
    write(6,"(A)")'    -inener=real - (optional) only real part of z_i &
     is given { Re(z_i) Re(f(z_i)) Im(f(z_i) }.'
    write(6,"(A)")'    -inener=imag - (optional) only imaginary part &
    of z_i is given { Im(z_i) Re(f(z_i)) Im(f(z_i) } (e.g. &
    Matsubara axis).'   
    write(6,"(A)")
    write(6,"(A)")'                   The output energy mesh can be &
    defined in two different ways:'
    write(6,"(A)")'    filename2    - (optional) on input contains &
    complex energies, c_i, to which perform analytical continuation &
    { Re(c_i) Im(c_i) }.'
    write(6,"(A)")'                              on output contains &
    c_i and f(c_i) { Re(c_i) Im(c_i) Re(f(c_i)) Im(f(c_i) }.'
    write(6,"(A)")'                   If present, the last of the &
    options will not be taken into account.'
    write(6,"(A)")
    write(6,"(A)")'                   If -of option is omitted next &
    parameters create output energy mesh,'
    write(6,"(A)")'                        c_i = cmplx( emin+de*(i-1), &
    eta ), where de=(emax-emin)/(n-1).'
    write(6,"(A)")'    emin         - (optional, default=-5) lower &
    energy boundary.' 
    write(6,"(A)")'    emax         - (optional, default= 5) upper &
    energy boundary.' 
    write(6,"(A)")'    eta          - (optional) imaginary offset, &
    default value depends on -inener=imag, otherwise = 0.05.'
    write(6,"(A)")'    npoints      - (optional, default=1000) number &
    of points for energy mesh.'
    write(6,"(A)")
    write(6,"(A)")' NB: If on output NaN occurs then reduce the number &
    of the input points.'
    write(6,"(A)")
    stop
  end if  

!   Read input file


  i = 0
  do
    i = i + 1
    read(10,*,iostat=ios) e1
    if( ios /= 0 ) exit
  end do
  n_in = i - 1
  ios  = 0
  rewind(10)
  
  allocate( z_in(n_in), f(n_in) )
  
  do i = 1, n_in 
    if( real_in_energy )then
      read(10,*,iostat=ios) e1, ref, imf
      z_in(i) = cmplx(e1,0,prec)
      f(i)    = cmplx(ref,imf,prec)
    elseif( imag_in_energy )then
      read(10,*,iostat=ios) e1, ref, imf
      z_in(i) = cmplx(0,e1,prec)
      f(i)    = cmplx(ref,imf,prec)
    else
      read(10,*,iostat=ios) e1, e2, ref, imf
      z_in(i) = cmplx(e1,e2,prec)
      f(i)    = cmplx(ref,imf,prec)
    end if
  end do
  
  if( ios /= 0 ) stop
  
!   Make energy mesh for output

  if( filename_out_exist )then
!   Read energy mesh from filename_out

    i = 0
    do
      i = i + 1
      read(20,*,iostat=ios) e1
      if( ios /= 0 ) exit
    end do
    n_out = i - 1
    rewind(20)
    
    allocate( z_out(n_out) )
    
    do i = 1, n_out
      read(20,*,iostat=ios) e1, e2
      z_out(i) = cmplx(e1,e2,prec)
    end do
    rewind(20)
  
  else
!   Make energy mesh using emin, emax, npoints, eta
    
    filename_out = trim(adjustl(filename_in))//'__output'
    open( 20, file=filename_out, form='formatted', position='rewind' )
    
    if( .not.eta_exist .and. imag_in_energy ) eta = imag(z_in(1))
    
    n_out = npoints
    allocate( z_out(n_out) )
  
    e1 = ( emax - emin ) / real( n_out - 1 ,prec)
    do i = 1, n_out
      e2 = emin + e1*(i-1)
      z_out(i) = cmplx(e2,eta,prec)
    end do
    
  end if

!==============================================================================  

!   Calculate Pade coefficients
 
  call pade_coefficients( f, z_in, n_in )
  
  do i = 1, n_out
    call pade( fz, z_out(i), z_in, f, n_in )
    write(20,"(2(1x,f20.12),2(1x,e20.12))") z_out(i), fz
!    write(20,"(2(1x,2f20.12))") z_out(i), fz
  end do

end program analytical_continuation


subroutine pade_coefficients( a, z, n )
!  
!             Computation of Pade coefficients.		    
!    Inputs:							    
!io    a  -  On input is a function for approximation
!            on output contains Pade coefficients		
!i     z  -  complex points in which function a is determined
!i     n  -  size of arrays.					
! 
!r   Remarks: 						
!         (J. of Low Temp. Phys., v29, n3/4, 1977)		       
!						    
  use precs
  
  implicit none
!--->  Passed variables
  integer,       intent(in   ) :: n
  complex(prec), intent(in   ) :: z(n)
  complex(prec), intent(inout) :: a(n)
!--->  Local variables 
  integer :: i, j
  complex(prec), allocatable :: g(:,:)
     
  allocate( g(n,n) )
  
  g(1,:) = a   

  do j = 2,n
    do i = 2,j
      g(i,j) = ( g(i-1,i-1)-g(i-1,j) ) / ( z(j)-z(i-1) ) / g(i-1,j)
    end do
  end do
  
  forall( i = 1:n ) a(i) = g(i,i)
    
  deallocate( g )
    
end subroutine pade_coefficients

subroutine pade( f, z1, z, p, n )
!
!	     Calculation of analytical function	
!    in the arbitrary complex point for a given Pade coefficients 
!								       
!    Inputs:							       
!i     p  -  Pade coefficients				       
!i     z  -  set of points from which analytical continue is performed
!i     n  -  size of arrays				       
!i     z1 -  complex point					       
!    Outputs:							       
!i     f  -  value of function
!
  use precs
  
  implicit none
!---> Passed variables
  integer,       intent(in   ) :: n
  complex(prec), intent(in   ) :: z1
  complex(prec), intent(in   ) :: z(n)
  complex(prec), intent(in   ) :: p(n)
  complex(prec), intent(  out) :: f
!---> Local variables
  integer       :: i, np
  real(prec)    :: cof1, cof2, d_sum
  complex(prec) :: a1, a2, b1, b2, anew, bnew

!  d_sum = 0.d0
!  do i = 1,n-1
!    d_sum = d_sum + abs( (z1-z(i))*p(i+1) )
!  end do  

!  if( d_sum >= 1.d0 )then
!    d_sum = nint(n*d_sum/(n-1)) 
!    d_sum = log10(d_sum)
!    np  = int(d_sum) + 1
!    cof1 = 1.d-250
!    cof2 = 1.d0
!    if( np > 275 ) cof2 = 1.d-275
!    else  
!    d_sum = nint((n-1)/(n*d_sum)) 
!    d_sum = log10(d_sum)
!    np  = -int(d_sum) - 1
!    cof1 = 1.d+250
!    cof2 = 1.d0
!    if( np < -275 ) cof2 = 1.d+275
!  end if

  cof1 = 1
  cof2 = 1

  a1 = cmplx(0,0,prec)
  a2 = cof1*p(1)
  b1 = cmplx(cof1,0,prec)
  b2 = cmplx(cof1,0,prec)
  
  do i = 1,n-1
    anew = a2 + ( z1 - z(i) ) * p(i+1) * a1
    bnew = b2 + ( z1 - z(i) ) * p(i+1) * b1
    a1   = a2
    b1   = b2
    a2   = anew
    b2   = bnew
  end do
    
  a2 = cof2 * a2
  b2 = cof2 * b2
  f  = a2 / b2

end subroutine pade
