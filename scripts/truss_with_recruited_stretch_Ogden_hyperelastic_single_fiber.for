      SUBROUTINE UHYPER_STRETCH(DLAMBDA,AJ,U,U1,U2,U3,U4,TEMP,NOEL,
     1 CMNAME,INCMPFLAG,NUMSTATEV,STATEVOLD,STATEV,NUMFIELDV,FIELDV,
     2 FIELDVINC,NUMPROPS,PROPS,I_ARRAY,NIARRAY,R_ARRAY,NRARRAY,
     3 C_ARRAY,NCARRAY)

      INCLUDE 'ABA_PARAM.INC'

      CHARACTER*80 CMNAME,C_ARRAY(*)

      INTEGER NOEL,INCMPFLAG,NUMSTATEV,NUMFIELDV,NUMPROPS
      INTEGER NIARRAY,NRARRAY,NCARRAY
      INTEGER I

      DOUBLE PRECISION DLAMBDA(*),AJ,U(2),U1(4),U2(4),U3(4),U4(3)
      DOUBLE PRECISION TEMP,STATEVOLD(*),STATEV(*),FIELDV(*)
      DOUBLE PRECISION FIELDVINC(*),PROPS(*),I_ARRAY(*),R_ARRAY(*)

      DOUBLE PRECISION LAM,PS,MUF,AF
      DOUBLE PRECISION WF0,WF1,WF2,WF3,WF4
      DOUBLE PRECISION WM0,WM1,WM2,WM3,WM4
      DOUBLE PRECISION ZERO,ONE,TWO,TINY

      PARAMETER (ZERO=0.0D0, ONE=1.0D0, TWO=2.0D0, TINY=1.0D-14)

C----- initialize outputs
      U(1)=ZERO
      U(2)=ZERO

      DO 10 I=1,4
         U1(I)=ZERO
         U2(I)=ZERO
         U3(I)=ZERO
   10 CONTINUE

      DO 20 I=1,3
         U4(I)=ZERO
   20 CONTINUE

      DO 30 I=1,NUMSTATEV
         STATEV(I)=STATEVOLD(I)
   30 CONTINUE

C----- truss-only: use first principal stretch
      LAM = DLAMBDA(1)

C----- straightness parameter Ps
C----- preferred: assign through FIELDV(1) per truss / element set
      PS = 9.155D0 / (9.155D0 + 1.275D0)
      IF (NUMFIELDV .GE. 1) THEN
         IF (FIELDV(1) .GT. TINY) THEN
            PS = MIN(FIELDV(1), ONE)
         END IF
      END IF

C----- fiber Ogden properties from the input deck
      MUF = 80.8D0
      AF  = 62.3D0
      IF (NUMPROPS .GE. 2) THEN
         IF (PROPS(1) .GT. TINY) MUF = PROPS(1)
         IF (ABS(PROPS(2)) .GT. TINY) AF = PROPS(2)
      END IF

C----- recruited embedded fiber part: psi_f(lambda_t)
      CALL FIBER_PAPER_OGDEN(LAM,PS,MUF,AF,WF0,WF1,WF2,WF3,WF4)

C----- subtract matrix redundancy part: psi_m(lambda)
      CALL MATRIX_PAPER_OGDEN(LAM,WM0,WM1,WM2,WM3,WM4)

C----- modified embedded-fiber energy: psi_tilde = psi_f - psi_m
      U(2)  = WF0 + WM0
      U1(1) = WF1 + WM1
      U2(1) = WF2 + WM2
      U3(1) = WF3 + WM3
      U4(1) = WF4 + WM4

C----- leave transverse branches zero for truss-only use
      U1(2)=ZERO
      U1(3)=ZERO
      U1(4)=ZERO

      U2(2)=ZERO
      U2(3)=ZERO
      U2(4)=ZERO

      U3(2)=ZERO
      U3(3)=ZERO
      U3(4)=ZERO

      U4(2)=ZERO
      U4(3)=ZERO

      U(1)=U(2)

      RETURN
      END


C======================================================================
C  Embedded fiber from the paper:
C  psi_f = 2*mu_f/alpha_f^2 * ( lam_t^alpha_f
C         + 2*lam_t^(-alpha_f/2) - 3 )
C
C  Recruitment:
C    lam_t = 1            , 1 <= lam < 1/Ps
C    lam_t = lam*Ps       , lam >= 1/Ps
C
C  Paper values:
C    mu_f = 80.8 Pa
C    alpha_f = 62.3
C======================================================================
      SUBROUTINE FIBER_PAPER_OGDEN(LAM,PS,MUF,AF,W0,W1,W2,W3,W4)

      DOUBLE PRECISION LAM,PS,MUF,AF,W0,W1,W2,W3,W4
      DOUBLE PRECISION C0
      DOUBLE PRECISION LT,DLT
      DOUBLE PRECISION B1,B2,B3
      DOUBLE PRECISION ZERO,ONE,TWO,THREE,FOUR,SIX

      PARAMETER (ZERO=0.0D0, ONE=1.0D0, TWO=2.0D0,
     1           THREE=3.0D0, FOUR=4.0D0, SIX=6.0D0)

      C0  = TWO*MUF/(AF*AF)

C----- unrecruited region: no mechanical contribution
      IF (LAM .LT. ONE/PS) THEN
         LT  = ONE
         DLT = ZERO

         W0 = C0*(LT**AF + TWO*LT**(-AF/TWO) - THREE)
         W1 = ZERO
         W2 = ZERO
         W3 = ZERO
         W4 = ZERO
         RETURN
      END IF

C----- recruited region
      LT  = LAM*PS
      DLT = PS

      W0 = C0*(LT**AF + TWO*LT**(-AF/TWO) - THREE)

      W1 = C0*( AF*LT**(AF-ONE)
     1        - AF*LT**(-AF/TWO-ONE) ) * DLT

      W2 = C0*( AF*(AF-ONE)*LT**(AF-TWO)
     1        + AF*(AF/TWO+ONE)*LT**(-AF/TWO-TWO) ) * DLT*DLT

      W3 = C0*( AF*(AF-ONE)*(AF-TWO)*LT**(AF-THREE)
     1        - AF*(AF/TWO+ONE)*(AF/TWO+TWO)
     2          *LT**(-AF/TWO-THREE) ) * DLT*DLT*DLT

      W4 = C0*( AF*(AF-ONE)*(AF-TWO)*(AF-THREE)
     1          *LT**(AF-FOUR)
     2        + AF*(AF/TWO+ONE)*(AF/TWO+TWO)*(AF/TWO+THREE)
     3          *LT**(-AF/TWO-FOUR) ) * DLT*DLT*DLT*DLT

      RETURN
      END

C======================================================================
C  Matrix redundancy subtraction from the paper:
C  psi_m = 2*mu_m/alpha_m^2 * ( lam^alpha_m
C         + 2*lam^(-alpha_m/2) - 3 )
C
C  Paper values:
C    mu_m = 353.5 Pa
C    alpha_m = -21.5
C======================================================================
      SUBROUTINE MATRIX_PAPER_OGDEN(LAM,W0,W1,W2,W3,W4)

      DOUBLE PRECISION LAM,W0,W1,W2,W3,W4
      DOUBLE PRECISION MUM,AM,C0
      DOUBLE PRECISION ZERO,ONE,TWO,THREE,FOUR

      PARAMETER (ZERO=0.0D0, ONE=1.0D0, TWO=2.0D0,
     1           THREE=3.0D0, FOUR=4.0D0)

      MUM = 1.0D0
      AM  = 2.0D0
      C0  = TWO*MUM/(AM*AM)

      W0 = C0*(LAM**AM + TWO*LAM**(-AM/TWO) - THREE)

      W1 = C0*( AM*LAM**(AM-ONE)
     1        - AM*LAM**(-AM/TWO-ONE) )

      W2 = C0*( AM*(AM-ONE)*LAM**(AM-TWO)
     1        + AM*(AM/TWO+ONE)*LAM**(-AM/TWO-TWO) )

      W3 = C0*( AM*(AM-ONE)*(AM-TWO)*LAM**(AM-THREE)
     1        - AM*(AM/TWO+ONE)*(AM/TWO+TWO)
     2          *LAM**(-AM/TWO-THREE) )

      W4 = C0*( AM*(AM-ONE)*(AM-TWO)*(AM-THREE)
     1          *LAM**(AM-FOUR)
     2        + AM*(AM/TWO+ONE)*(AM/TWO+TWO)*(AM/TWO+THREE)
     3          *LAM**(-AM/TWO-FOUR) )

      RETURN
      END

