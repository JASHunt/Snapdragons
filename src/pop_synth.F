      program PopSynth
      implicit none
      integer MN,MNSc,MN3D,MNiso,npr
      double precision VUKMS,LUKPC,mtemp
      parameter (MN=9125000)
      parameter (MNiso=34)
      parameter (MNSc=4096)
      parameter (MN3D=1024)
      parameter (VUKMS=207.4d0)
      parameter (LUKPC=100.0d0)
      integer i,niso(0:MNiso-1),Ns,idum,j,k,jindex(0:6200)
     &     ,IsoNo(0:MNiso-1),l,q,inputdata,GenSel,per3,Binary
     &     ,IsoStart(0:MNiso-1),ageindex,np,m,per2,lbegin,lend,ngen
      integer EXTINCT,Schlegel,dummyint,nparts,xIMF
     &     ,Ns1,Ns2,Nsf1,Nsf2
      double precision MiMax,Afac,Afac1,z,PAge,vgen,igen,avgen,aigen
      double precision ugen,bgen,rgen,jgen,hgen,kgen,Afac2
      double precision augen,abgen,argen,ajgen,ahgen,akgen
      double precision exU,exB,exR,exJ,exH,exK
      double precision ChosenLim,frac,Vigen,dummy,dmod,mag2,rd,circ
      double precision Age(0:MN-1),logL_Lo(0:MN-1),logG(0:MN-1),
     &     Mi(0:MN-1),Mc(0:MN-1),logT(0:MN-1),
     &     C_O(0:MN-1),VMU(0:MN-1),
     &     VMB(0:MN-1),VMV(0:MN-1),VMR(0:MN-1),VMI(0:MN-1),
     &     VMJ(0:MN-1),VMH(0:MN-1),VMK(0:MN-1),
     &     mbol(0:MN-1)
      double precision vl_p,vb_p,vlo_p,vlo2,vbo_p,vrots,mtn,Nsf
     &     ,lo_p,bo_p,l2_p,b2_p
      double precision M_hec,period,pmode,logMdot,intIMF,Msmin
      double precision xsun,ysun,zsun,usun,vsun,wsun,vlsr,mtotalneed
      double precision d_p,Milb,mag,mgen,sumgen,R,tss,month
      double precision x_p,y_p,z_p,vx_p,vy_p,vz_p,m_p,ex,ex31,ex148
      integer il,ib,ie1,ie2,ie3,ie4,ie5,ie6,ie7,ie8,ii
      integer il3,ib3,id3,error,jflag
      double precision dl,db,wl1,wl2,wb1,wb2,wd1,wd2,exS,ex3
      double precision dl3,db3,dd3,CAfactor
      double precision leS(0:MNSc-1),beS(0:MNSc-1),le3(0:MN3D-1),
     &     be3(0:MN3D-1),de3(0:99),datS(0:16777215),dat3(0:104857599)
      double precision l_p,b_p,a(6),ao(6),ae(6),p(2),po(2),pe(2), 
     &     ap(4),apo(4),ape(4),Tao(5),Tae(5),Ta(5)
      double precision M_PI,t0,uh,vh,wh,makms,mdkms,makmso,mdkmso
      parameter (M_PI=4.0d0*datan(1.0d0))
      double precision alpha,delta,g,gb,gl,GRVS
      double precision Uo,Vo,Wo,xo_p,yo_p,zo_p,vxo_p,vyo_p,vzo_p,
     &     vro_p,vroto_p,rop,od_p,rp,vr_p,vrot_p,Ue,Ve,We
c      parameter (M_PI=3.1415926535897932384626433d0)
      real ran1
      character*40 FN
      character*25 ONAME
      character*1 check
      character*1 hash

c *** SNAPDRAGONS/PopSynth version 2.1 21/03/16 ***
c *** Most recent addition: Kroupa IMF, not tested fully ***

c *** For support contact jason.hunt.11@ucl.ac.uk ***
c *** For paper see Hunt & Kawata 2015, MNRAS, 450, 2132 ***

c *** For ASCII input file ini/input.dat ***
c *** x,y,z(kpc),vx,vy,vz(km/s),metallicity(z),age(Gyr),mass(solar) ***
c *** Note metallciity in z not [Fe/H]

c *** USER OPTIONS ***
c *** Output to binary (1) or ASCII (0)? ***
      Binary=0
c *** Minimum magnitude necessary ***
c$$$      ChosenLim=11.0d0
      ChosenLim=20.0d0
c$$$      ChosenLim=1000.0d0
      write(6,*) 'Limited to V<',ChosenLim
c *** Add Gaia errors? 
c *** 0=No errors
c *** 1=Gaia - Use 'Months' to set intermediate data releases
c *** 2=TGAS Tycho stars
c *** 3=TGAS Hipparcos stars only
c$$$      error=0
      error=1
c *** mission length in months, 60=5 years ***
c *** 22 months for Gaia DR2 (testing ongoing) ***
      month=22.0d0
c$$$      month=60.0d0
c *** Add Milky Way Extinction (1=yes) ***
      EXTINCT=0
c$$$      EXTINCT=1
c *** Extintion map (1=Schlegel, 0=2d analytic) ***
      Schlegel=1
c *** Choose IMF (Salpeter=1, Kroupa=2) ***
      xIMF=1
c$$$      xIMF=2
c *** Random seed ***
      idum=-1314
c *** Set Sun position / velocity ***
c *** Should be negative along x-axis (Will be fixed to be anywhere at some point)
      xsun=-8.0d0
      ysun=0.0d0
      zsun=0.02d0
      vlsr=1.1d0*VUKMS
c *** vlsr=1.1d0*VUKMS=228.14 km/s ***
c *** 11.1,12.24,7.25 from Schoenrich, Binney & Dehnen ***
      usun=11.1d0
      vsun=12.24d0
      wsun=7.25d0
c *** END USER OPTIONS ***

      if (xIMF.eq.1) then
         write(6,*) 'Using Salpeter IMF'
      else if (xIMF.eq.2) then
         write(6,*) 'Using Kroupa IMF'
      endif
      if (error.eq.0) then
         write(6,*) 'No error added'
      else if (error.eq.1) then
         write(6,*) 'Gaia error added'
      else if (error.eq.2) then
         write(6,*) 'TGAS error added (Tycho stars)'
      else if (error.eq.3) then 
         write(6,*) 'TGAS error added (Hipparcos stars)'
      endif
c *** TEMP particle ***
      open(20,FILE='ini/UBV/IsoNo.txt', STATUS='OLD')
      do i=0,MNiso-1
         read(20,'(I10)') IsoNo(i)
      enddo
      close(20)
      per2=0
      per3=0
      ngen=0
      GenSel=0
c *** Read extinction arrays ***
      if (EXTINCT.eq.1) then
         if (Schlegel.eq.1) then
            open(51,file='ini/ScEx1.bin',status='old'
     &           ,form='unformatted')
            write(6,*) 'Reading Extinction arrays'
            do i=0,16777215
               read(51,end=156) datS(i)
            enddo
 156                dl=(2.0d0*M_PI)/4095.0d0 
            db=M_PI/4095.0d0   
            write(6,*) 'Schlegel extinction array Read'
         else
            open(51,file='ini/2dEx1.bin',status='old'
     &           ,form='unformatted')
            write(6,*) 'Reading Extinction arrays'
            do i=0,1048575
               read(51,END=157) datS(i)
            enddo
             write(6,*) '2D analytical extinction array Read'
 157                  dl=(2.0d0*M_PI)/1023.0d0 
             db=M_PI/1023.0d0   
         endif
         close(51)
         open(51,file='ini/3dEx1.bin',status='old',form='unformatted')
         do i=0,104857599
            read(51,END=158) dat3(i)
         enddo
 158          close(51)
         write(6,*) '3D factor array Read'
         write(6,*) 'Extinction running'
c     0<l<2*pi 
         dl3=(2.0d0*M_PI)/1023.0d0  
c     -pi/2<b<pi/2, with +pi/2 for +ve
         db3=M_PI/1023.0d0    
         dd3=3.47712123/99.0d0 
      else
         write(6,*) '*** No Extinction Added ***'
      endif
c *** Define SSP data ***
      Msmin=0.1d0
c$$$      Msmin=0.15d0   ! consistent with isochrone files
c *** Read in (Padova) Isochrone Files ***   
      j=0
      k=1
      jindex(0)=0
      do i=0,MNiso-1
         write(FN,'(a18,i5.5,a4)')'ini/UBV/output_0.0',IsoNo(i),'.dat'
         open(20,FILE=FN,STATUS='OLD')
c *** 10 headers read and discarded ***
         do l=1,9
            read(20,'(A150)')
         enddo
         hash='#'
c *** Read data ***
         niso(i)=0
         write(ONAME,'(a12,i5.5,a4)')'ini/UBV/V-VI',IsoNo(i),'.dat'
         do l=0,30000
            read(20,*,end=600) check
            if (index(check,hash).ne.1) then
               backspace(20)
               read(20,*) Age(j),Mi(j),Mc(j),logL_Lo(j),logT(j)
     &              ,logG(j),mbol(j),VMU(j),VMB(j),VMV(j),VMR(j)
     &              ,VMI(j),VMJ(j),VMH(j),VMK(j),C_O(j),M_hec
     &              ,period,pmode,logMdot,intIMF
               j=j+1
               niso(i)=niso(i)+1
            else
               read(20,*,end=600) check
               jindex(k)=j
               k=k+1
            endif
         enddo 
 600         close(20)
         IsoStart(i)=j
      enddo
      write(6,*) 'Isochrones Read'
 1940  format(17(1pE13.5),I10,3(1pE13.5))
c *** Open files to read & write ***
c *** open output file (binary or ASCII) ***
      if (Binary.eq.0) then
         open(10,file='output/GeneratedStars.dat'
     &        ,status='unknown')
c         open(15,file='ini/input.dat',status='unknown')
      else
      open(10,file='output/GeneratedStars.bin',status='unknown'
     &     ,form='unformatted')
      endif
c *** Open input file ***
c *** For example file (lbs.dat format)***
c      open(30,file='ini/lbsels-2015-25deglt10.dat',status='unknown')
c      open(30,file='ini/DaisukeInput.dat',status='unknown')


c      open(30,file='ini/l85-270bfullFaure.dat',status='unknown')
c      open(30,file='ini/lbsels.dat',status='unknown')
c 2030 format(16(1pE13.5),I10)
c 2030  format(17(1pE13.5),I10)
c 2031 format(13(1pE13.5))
c *** For simpler file ***
      open(30,file='ini/input.dat'
     &     ,status='unknown')
 2030  format(9(1pE13.5))
c      open(30,file='ini/d001000',status='unknown')
c 2030 format(7(1pE13.5),I10,2(1pE13.5))
      np=MN
      sumgen=0.0d0
      mtotalneed=0.0d0
      nparts=0
c   16 format(9(1pE13.5))  
      do q=0,np-1
         mtemp=0.0d0
c *** For example file (DaisukeInupt.dat) ***
c         read(30,2030,end=603) l2_p,b2_p,dummy,x_p,y_p,z_p,dummy,
c     &        dummy,dummy,vx_p,vy_p,vz_p,dummy,dummy,z,tss,m_p,dummyint
c *** For example file (Faure.dat) ***
c         read(30,2030,end=603) l2_p,b2_p,dummy,x_p,y_p,z_p,dummy,
c     &        dummy,vx_p,vy_p,vz_p,dummy,dummy
c         m_p=1.0e3
c         x_p=x_p+8.0d0
c         PAge=dlog10(ran1(idum)*12000000000.0d0)
ccc         z=(ran1(idum)*0.9d0+0.1d0)*19000.0d0
c         z=(ran1(idum)*0.4d0+0.8d0)*19000.0d0
c         PAge=dlog10(6000000000.0d0)
c         z=19000.0d0
c above is roughly right range, not dispersion
c 1.2589254*19000 (23919)- 0.7943282*19000 (15092)
c *** For example file (old lbs.dat) ***
c         read(30,2030,end=603) dummy,dummy,dummy,x_p,y_p,z_p,dummy,
c     &        dummy,vx_p,vy_p,vz_p,dummy,dummy,z,tss,m_p,dummyint
c *** x,y,z(kpc),vx,vy,vz(km/s),metallicity(z),age(Gyr),mass(solar)
         read(30,*,end=603) x_p,y_p,z_p,vx_p,vy_p,vz_p,z,tss,m_p
cz,tss,m_p
c         write(15,16) x_p,y_p,z_p,vx_p,vy_p,vz_p,z,tss,m_p
         nparts=nparts+1
         if (q.eq.0) then
            xsun=xsun/100.0d0
            ysun=ysun/100.0d0
            zsun=zsun/100.0d0
         endif
c         m_p=m_p*1.0e11/6.8d0
         x_p=x_p/100.0d0 
         y_p=y_p/100.0d0
         z_p=z_p/100.0d0
c For input velocities w.r.t Sun
c         vx_p=(vx_p+usun)/VUKMS
c         vy_p=(vy_p+vlsr+vsun)/VUKMS
c         vz_p=(vz_p+wsun)/VUKMS
c For input velocities w.r.t GC
         vx_p=vx_p/VUKMS
         vy_p=vy_p/VUKMS
         vz_p=vz_p/VUKMS
c         vy_p=vy_p+vlsr
c *** If metallicity in [M/H] (dex) - for marigo, should change ***
c         z=0.019*(10.**z)
c *** From Auriga ***
c         z=z*0.0019d0
c *** Turn to internal for isochrone matching ***
         z=z*1e6
c         write(6,*) z
         if (z.lt.100.0d0) then
            z=100.0d0
         else if (z.gt.30000.0d0) then
            z=30000.0d0
         endif 
         z_p=z_p-zsun
         y_p=y_p-ysun
         x_p=x_p-xsun
         rp=dsqrt((x_p+xsun)**2+(y_p+ysun)**2)
         vr_p=((vx_p*(x_p+xsun)+vy_p*(y_p+ysun)+vz_p
     &        *(z_p+zsun))/rp)*VUKMS
         vrot_p=(-1.0d0*((x_p+xsun)*vy_p-(y_p+ysun)*vx_p)/rp)*VUKMS
         rp=dsqrt((x_p+xsun)**2+(y_p+ysun)**2+(z_p+zsun)**2)
         d_p=dsqrt(x_p**2+y_p**2+z_p**2)
c         write(6,*) 'dp=',d_p,'rp=',rp,xsun,zsun
         dmod=5.0d0*(dlog10(d_p*100000.0d0)-1.0d0)
         PAge=log10(tss*1000000000.0d0)
c *** Find approproate isochrone segment ***
c *** Random age & metallicity if unknown ***
c         z=(ran1(idum)*0.9d0+0.1d0)*19000.0d0
c         PAge=log10(ran1(idum)*12000000000.0d0)
         ageindex=nint((PAge-6.6d0)/0.02d0)
         if (ageindex.lt.0) then 
            ageindex=0
         else if (ageindex.gt.181) then
            ageindex=181
         endif
c         write(6,*) 'ageindex=',ageindex,'z=',z,'d_p=',d_p
c *** extinction (CTIO) ***
         if (EXTINCT.eq.1) then
c *** Make grids l,b,d ***
            if (Schlegel.eq.1) then
               do i=0,4095
                  leS(i)=i*dl
                  beS(i)=i*db-M_PI/2.0d0
               enddo
            endif
            do i=0,1023
               if (Schlegel.ne.1) then
                  leS(i)=i*dl
                  beS(i)=i*db-M_PI/2.0d0
               endif
               le3(i)=i*dl3
               be3(i)=i*db3-M_PI/2.0d0
            enddo
            do i=0,99
               de3(i)=i*dd3-2.0d0
            enddo
c *** Model l,b,d ***
            l_p=-datan2(y_p,-x_p)+M_PI
            b_p=dasin(z_p/d_p)
            vrots=(-1.0d0*((x_p*vy_p-y_p*vx_p)/d_p))
c *** Find il,ib,id for particles ***
            il=int(l_p/dl)
            ib=int((b_p+(M_PI/2.0d0))/db)
            il3=int(l_p/dl3)
            ib3=int((b_p+(M_PI/2.0d0))/db3)
            id3=int((dlog10(d_p*100.0d0)+2.0d0)/dd3)
            if (float(id3).gt.99.0) then
               id3=99
            else if (float(id3).lt.0.0) then
               id3=0
            endif
c *** Find weighting ***
            wl2=(l_p-leS(il))/dl
            wl1=1-wl2
            wb2=(b_p-beS(ib))/db
            wb1=1-wb2
c *** Find position in ScEx array ***
            if (Schlegel.eq.1) then
               ie1=(ib)*4096+il
               ie2=(ib+1)*4096+il
               ie3=(ib)*4096+il+1
               ie4=(ib+1)*4096+il+1
            else
               ie1=(ib)*1024+il
               ie2=(ib+1)*1024+il
               ie3=(ib)*1024+il+1
               ie4=(ib+1)*1024+il+1
            endif
c *** Schlegel (or 2D) extinction value ***
            exS=datS(ie1)*wl1*wb1+datS(ie2)*wl1*wb2
     &         +datS(ie3)*wl2*wb1+datS(ie4)*wl2*wb2
c *** Weighting for 3D ***
            wl2=(l_p-le3(il3))/dl3
            wl1=1-wl2
            wb2=(b_p-be3(ib3))/db3
            wb1=1-wb2
            wd2=(dlog10(d_p*100.0d0)-de3(id3))/dd3
            wd1=1-wd2
c *** Find position in 3DEx array ***
            ie1=(1048576*id3)+(1024*ib3)+il3
            ie2=(1048576*(id3+1))+(1024*ib3)+il3
            ie3=(1048576*(id3+1))+(1024*ib3+1)+il3
            ie4=(1048576*(id3+1))+(1024*(ib3+1))+il3+1
            ie5=(1048576*(id3+1))+(1024*ib3)+il3+1
            ie6=(1048576*id3)+(1024*(ib3+1))+il3
            ie7=(1048576*id3)+(1024*(ib3+1))+il3+1
            ie8=(1048576*id3)+(1024*ib3)+il3+1
c *** 3D extinction factor ***
            ex3=dat3(ie1)*wl1*wb1*wd1+dat3(ie2)*wl1*wb1*wd2+
     &          dat3(ie3)*wl1*wb2*wd2+dat3(ie4)*wl2*wb2*wd2+
     &          dat3(ie5)*wl2*wb1*wd2+dat3(ie6)*wl1*wb2*wd1+
     &          dat3(ie7)*wl2*wb2*wd1+dat3(ie8)*wl2*wb1*wd1
c *** Final extinciton value ***
            ex=ex3*exS
c *** Schlegel Modified as per Sharma et al (2014) ***
            ex=ex*(0.6d0+0.2d0*(1.0d0-dtanh((ex-0.15d0)/0.1d0)))
            if (q.eq.0) then
               write(6,*) 'Using correction from Sharma et al. (2014)'
c            write(6,*) 'Not using correction from Sharma et al. (2014)'
            endif
         else
            ex=0.0d0
         endif
         ex31=ex*3.24d0
         ex148=ex*1.962d0
         exU=ex*4.968d0
         exB=ex*4.325d0
         exR=ex*2.634d0
         exJ=ex*0.902d0
         exH=ex*0.576d0
         exK=ex*0.367d0
c *** End Extinction ***
         MiMax=0.0d0
         do i=1,MNiso-1
            if (z.le.dble(IsoNo(i))) then
               if (dble(IsoNo(i))-z.lt.z-dble(IsoNo(i-1))) then
                  lbegin=jindex(181*(i)+ageindex)
                  lend=jindex(181*(i)+ageindex+1)-1
c$$$                  lbegin=jindex(181*(i)+ageindex)+1
c$$$                  lend=jindex(181*(i)+ageindex+1)
               else
                  lbegin=jindex(181*(i-1)+ageindex)
                  lend=jindex(181*(i-1)+ageindex+1)-1
c$$$                  lbegin=jindex(181*(i-1)+ageindex)+1
c$$$                  lend=jindex(181*(i-1)+ageindex+1)
               endif
               do l=lbegin,lend
c *** Calc MiMax ***
                  if (Mi(l).gt.MiMax) then
                     MiMax=Mi(l)
                  endif
               enddo
c *** Calc A ***
               if (xIMF.eq.1) then
                  Afac=-m_p*0.35d0/(MiMax**(-0.35d0)
     &                 -Msmin**(-0.35d0))
               else if (xIMF.eq.2) then
                  Afac1=-m_p*0.3d0/(MiMax**(-0.3d0)-0.5d0**(-0.3d0))
                  Afac2=m_p*0.7d0/(0.5d0**(0.7d0)-Msmin**(0.7d0))
c           Afac3=-m_p*0.35d0/(MiMax**(-0.35d0)-Msmin**(-0.35d0))
               endif
c *** Calc Milb ***
c$$$               Milb=10.0d0**12.0d0
               Milb=Mi(lbegin)
               mag=VMV(lbegin)+dmod+ex31
               if (mag.lt.ChosenLim) goto 667
c$$$               do l=lbegin,lend
c$$$                  mag=VMV(l)+dmod+ex31
c$$$                  mag2=VMV(l-1)+dmod+ex31
               do l=lbegin,lend-1
                  mag=VMV(l)+dmod+ex31
                  mag2=VMV(l+1)+dmod+ex31
                  if (mag.lt.ChosenLim) then
                     frac=(ChosenLim-mag2)/(mag-mag2)
                     Milb=(Mi(l)-Mi(l-1))*frac+Mi(l-1)
c$$$                     write(6,*) 'Milb: ', Milb
                     goto 667
c *** Interpolate ***
                  endif
               enddo
               Milb=Mi(lend)
c *** CHANGED ***
c               goto 601
c *** Calc number to generate ***
 667                     if (xIMF.eq.1) then 
                  Nsf=((Afac/(-1.35d0))*(MiMax**(-1.35d0)-Milb
     &                 **(-1.35d0)))
                  Ns=int(Nsf)
                  if (ran1(idum).lt.(Nsf-Ns)) then
                     Ns=Ns+1
                  endif
               else if (xIMF.eq.2) then
                  if (Milb.ge.0.5d0) then
                     Nsf=((Afac1/(-1.3d0))*(MiMax**(-1.3d0)-Milb
     &                    **(-1.3d0)))
                     Ns=int(Nsf)
                     if (ran1(idum).lt.(Nsf-Ns)) then
                        Ns=Ns+1
                     endif
                  else if (Milb.lt.0.5d0) then 
                     Nsf1=((Afac1/(-1.3d0))*(MiMax**(-1.3d0)-0.5d0
     &                    **(-1.3d0)))
                     Nsf2=((Afac2/(-0.3d0))*(0.5d0
     &                    **(-0.3d0)-Milb**(-0.3d0)))   
                     Ns1=int(Nsf1)
                     if (ran1(idum).lt.(Nsf1-Ns1)) then
                        Ns1=Ns1+1
                     endif
                     Ns2=int(Nsf2)
                     if (ran1(idum).lt.(Nsf2-Ns2)) then
                        Ns2=Ns2+1
                     endif
                     Ns=Ns1+Ns2
                  endif       
               endif 
               if (xIMF.eq.1) then
                  mtn=(Afac/(-0.35d0))
     &                 *(MiMax**(-0.35d0)-Milb**(-0.35d0))
               else if (xIMF.eq.2) then
                  if (Milb.ge.0.5d0) then
                     mtn=(Afac1/(-0.3d0))
     &                    *(MiMax**(-0.3d0)-Milb**(-0.3d0))
                  else if (Milb.lt.0.5d0) then
                     mtn=(Afac1/(-0.3d0))
     &                    *(MiMax**(-0.3d0)-0.5d0**(-0.3d0))
     &                    +(Afac2/(0.7d0))
     &                    *(0.5d0**(0.7d0)-Milb**(0.7d0))
                  endif
               endif
               if (Ns.lt.0.0d0) then
                  Ns=0
                  mtn=0.0d0
               endif
               mtotalneed=mtotalneed+mtn
               if (Ns.eq.0) then
                  goto 601
               endif
c               write(6,*) q,Ns
c *** Generate stars ***
               do m=0,Ns-1
                  R=ran1(idum)
c *** Calculate stellar mass ***
                  if (xIMF.eq.1) then
                     mgen=(R*(MiMax**(-1.35d0))+(1.0d0-R)*(Milb
     &                    **(-1.35d0)))**(1.0d0/(-1.35d0))
                  else if (xIMF.eq.2) then
                     if (Milb.ge.0.5d0) then
                        mgen=(R*(MiMax**(-1.3d0))+(1.0d0-R)
     &                       *(Milb**(-1.3d0)))**(1.0d0/(-1.3d0))
                     else if (Milb.lt.0.5d0) then
                        if (m.lt.Ns1) then
                           mgen=(R*(MiMax**(-1.3d0))+(1.0d0-R)
     &                          *(0.5d0**(-1.3d0)))**(1.0d0/(-1.3d0))
                        else
                           mgen=(R*(0.5d0**(-0.3d0))+(1.0d0-R)
     &                          *(Milb**(-0.3d0)))**(1.0d0/(-0.3d0))
                        endif                     
                     endif
                  endif
c$$$                  do l=lbegin,lend
c$$$                     if (mgen.lt.Mi(l)) then
c$$$c *** Interpolate for magnitudes ***
c$$$                        frac=(mgen-Mi(l-1))/(Mi(l)-Mi(l-1))
c$$$                        avgen=frac*(VMV(l)-VMV(l-1))+VMV(l-1)
c$$$                        aigen=frac*(VMI(l)-VMI(l-1))+VMI(l-1)
c$$$                        augen=frac*(VMU(l)-VMU(l-1))+VMU(l-1)
c$$$                        abgen=frac*(VMB(l)-VMB(l-1))+VMB(l-1)
c$$$                        argen=frac*(VMR(l)-VMR(l-1))+VMR(l-1)
c$$$                        ajgen=frac*(VMJ(l)-VMJ(l-1))+VMJ(l-1)
c$$$                        ahgen=frac*(VMH(l)-VMH(l-1))+VMH(l-1)
c$$$                        akgen=frac*(VMK(l)-VMK(l-1))+VMK(l-1)
                  do l=lbegin,lend-1
                     if (mgen.lt.Mi(l+1)) then
c *** Interpolate for magnitudes ***
                        frac=(mgen-Mi(l))/(Mi(l+1)-Mi(l))
                        avgen=frac*(VMV(l+1)-VMV(l))+VMV(l)
                        aigen=frac*(VMI(l+1)-VMI(l))+VMI(l)
                        augen=frac*(VMU(l+1)-VMU(l))+VMU(l)
                        abgen=frac*(VMB(l+1)-VMB(l))+VMB(l)
                        argen=frac*(VMR(l+1)-VMR(l))+VMR(l)
                        ajgen=frac*(VMJ(l+1)-VMJ(l))+VMJ(l)
                        ahgen=frac*(VMH(l+1)-VMH(l))+VMH(l)
                        akgen=frac*(VMK(l+1)-VMK(l))+VMK(l)
c *** Add stellar extinction (and absolute -> apparent) ***
                        vgen=avgen+dmod+ex31
c *** If satisfies criteria *** 
c *** (Not all stars will, owing to the bend in isochrones) ***
                        if (vgen.lt.ChosenLim) then
                           igen=aigen+dmod+ex148
                           ugen=augen+dmod+exU
                           bgen=abgen+dmod+exB
                           rgen=argen+dmod+exR
                           jgen=ajgen+dmod+exJ
                           hgen=ahgen+dmod+exH
                           kgen=akgen+dmod+exK
                           VIgen=vgen-igen
c *** Write stellar output ***
                           x_p=x_p*LUKPC
                           y_p=y_p*LUKPC
                           z_p=z_p*LUKPC
                           call carte_to_equatorial(x_p,y_p,z_p
     &                          ,0.0d0,0.0d0,alpha,delta,gl,gb)
                           a(1)=alpha
                           a(2)=delta 
                           a(3)=1.0d0/(d_p*100000.0d0)
                           uh=vx_p*VUKMS-usun
                           vh=vy_p*VUKMS-vlsr-vsun
                           wh=vz_p*VUKMS-wsun
                           call UVWH_to_equatorials(uh,vh,wh,a)
                           jflag=-1 
                           do j=1,4
                              ap(j)=0.0d0
                           enddo
                           do j=3,5
                              a(j)=a(j)*1000.0d0
                           enddo
c *** Add Gaia errors ***
                           if (error.ne.0) then
c *** Calibration astrometric factor:
c *** CAfactor=1.0d0 means no calibration errors added ***
c *** CAfactor=1.2d0 gives 20% calibration error (on the safe side) ***
                              CAfactor=1.2d0
                              if (error.eq.1) then
                                 call Gaia_errors(month,CAfactor,jflag
     &                                ,Vgen,VIgen,a,ao,ae,p,po,pe,ap,apo
     &                                ,ape,idum)  
                              else if (error.eq.2) then
                                 do ii=1,5
                                    Ta(ii)=a(ii)
                                 enddo
                                 call TGASErrors(Vgen,Ta,Tao,Tae,1,idum)
                              else if (error.eq.3) then
                                 do ii=1,5
                                    Ta(ii)=a(ii)
                                 enddo
                                 call TGASErrors(Vgen,Ta,Tao,Tae,2,idum)
                              endif
                           endif  
                           do j=3,5
                              a(j)=a(j)/1000.0d0
                              ao(j)=ao(j)/1000.0d0
                           enddo
                           do j=1,5
                              ae(j)=ae(j)/1000.0d0
                           enddo
                           if (error.ge.2) then
                              do j=3,5
                                 Ta(j)=Ta(j)/1000.0d0
                                 Tao(j)=Tao(j)/1000.0d0
                              enddo
                              do j=1,5
                                 Tae(j)=Tae(j)/1000.0d0
                              enddo
                           endif
                           if (error.lt.2) then
                              call coord(ao,ae,xo_p,yo_p,zo_p,Uo,Vo,Wo
     &                             ,Ue,Ve,We,lo_p,bo_p,vlo_p,vbo_p)
                              vxo_p=(Uo+usun)
                              vyo_p=(Vo+vsun+VLSR)
                              vzo_p=(Wo+wsun)
                           endif
                           if (error.ge.2) then
                              call Tcoord(Tao,Ta,xo_p,yo_p,zo_p,
     &                             lo_p,bo_p,vlo_p,vbo_p,l_p,b_p,
     &                             vl_p,vb_p)
                              od_p=1.0d0/(Tao(3)*1.0e3)
                           endif
                           rop=dsqrt((xo_p+xsun)**2
     &                          +(yo_p+ysun)**2)*LUKPC
                           xo_p=xo_p*LUKPC
                           yo_p=yo_p*LUKPC
                           zo_p=zo_p*LUKPC
                           xsun=xsun*LUKPC
                           ysun=ysun*LUKPC
                           zsun=zsun*LUKPC
                           if (error.lt.2) then
                              od_p=1.0d0/(ao(3)*1.0e3)
                              vro_p=((vxo_p*(xo_p+xsun)
     &                             +vyo_p*(yo_p+ysun))/rop)
                              vroto_p=(-1.0d0*((xo_p+xsun)*vyo_p
     &                             -(yo_p+ysun)
     &                             *vxo_p)/rop)
                              vrots=(-1.0d0*(xo_p*vyo_p-yo_p*vxo_p)/
     &                             (od_p/100.0d0))
                              rop=dsqrt((xo_p+xsun)**2+(yo_p+ysun)**2
     &                             +(zo_p+zsun)**2)
                              vlo_p=vrots/(4.74d0*(od_p*1000.0d0))
                              vbo_p=vzo_p*VUKMS/(4.74d0*(od_p*1000.0d0))
                           endif
                           xsun=xsun/LUKPC
                           ysun=ysun/LUKPC
                           zsun=zsun/LUKPC
                           x_p=x_p/LUKPC
                           y_p=y_p/LUKPC
                           z_p=z_p/LUKPC
                           d_p=d_p*LUKPC
                           GRVS=Vgen-0.0119d0-1.2092d0*VIgen+0.0188d0
     &                          *(VIgen**2)+0.0005d0*(VIgen**3)
                           makmso=ao(4)*4.74d0*od_p*1000.0d0
                           mdkmso=ao(5)*4.74d0*od_p*1000.0d0
                           makms=a(4)*4.74d0*d_p*1000.0d0
                           mdkms=a(5)*4.74d0*d_p*1000.0d0
c **** HERE YOU CAN ADD CUSTOM REQUIREMENTS FOR OUTPUTS ****
c                           if ((od_p.gt.0.0d0).and.
c     &                          (od_p.lt.40.0d0)) then
cccc                           if (GRVS.lt.16.2d0) then
                           lo_p=(lo_p*180.0d0)/M_PI
                           bo_p=(bo_p*180.0d0)/M_PI
c                           if (lo_p.gt.175.0d0) then
c                           if (lo_p.lt.185.0d0) then
c                           if (bo_p.gt.-5.0d0) then
c                           if (bo_p.lt.5.0d0) then
                           if (error.lt.2) then
                              if (Binary.eq.0) then
c *** Ascii ***
                                if ((q.eq.0).and.(m.eq.0)) then
                                   write(6,*) 'Writing Gaia ascii'
                                endif
                                write(10,102) (a(j),j=1,6),(ao(j),j=1,6)
     &                             ,(ae(j),j=1,6),(p(j),j=1,2)
     &                             ,(po(j),j=1,2),(pe(j),j=1,2)
     &                             ,vgen,d_p,x_p,y_p,z_p,vx_p,vy_p,vz_p
     &                             ,mgen,GRVS,VIgen,ex31,avgen
     &                             ,avgen-aigen,z,PAge,od_p,xo_p,yo_p
     &                             ,zo_p,vxo_p,vyo_p,vzo_p,vro_p,vroto_p
     &                             ,rp,vr_p,vrot_p,rop,makms,mdkms
     &                             ,makmso,mdkmso,l_p,b_p,lo_p,bo_p
     &                             ,vl_p,vb_p,vlo_p,vbo_p
     &                             ,ugen,bgen,rgen,jgen,hgen,kgen
     &                             ,Ue,Ve,We
c *** Binary ***
                             else
                                if ((q.eq.0).and.(m.eq.0)) then
                                   write(6,*) 'Writing Gaia Binary'
                                endif
                                write(10) (a(j),j=1,6),(ao(j),j=1,6)
     &                             ,(ae(j),j=1,6),(p(j),j=1,2)
     &                             ,(po(j),j=1,2),(pe(j),j=1,2)
     &                             ,vgen,d_p,x_p,y_p,z_p,vx_p,vy_p,vz_p
     &                             ,mgen,GRVS,VIgen,ex31,avgen
     &                             ,avgen-aigen,z,PAge,od_p,xo_p,yo_p
     &                             ,zo_p,vxo_p,vyo_p,vzo_p,vro_p,vroto_p
     &                             ,rp,vr_p,vrot_p,rop,makms,mdkms
     &                             ,makmso,mdkmso,l_p,b_p,lo_p,bo_p
     &                             ,vl_p,vb_p,vlo_p,vbo_p
     &                             ,ugen,bgen,rgen,jgen,hgen,kgen
     &                             ,Ue,Ve,We
                              endif
cccc                           endif
                           if (error.ge.2) then
c                              if (od_p.gt.0.0d0) then
c                                 if (od_p.lt.4.5d0) then
                              if ((q.eq.0).and.(m.eq.0)) then
                                 write(6,*) 'Writing TGAS ASCII'
                              endif
                              vlo_p=vlo_p*4.74d0*od_p*1000.0d0
                              vbo_p=vbo_p*4.74d0*od_p*1000.0d0
                              vl_p=vl_p*4.74d0*d_p*1000.0d0
                              vb_p=vb_p*4.74d0*d_p*1000.0d0
                              write(10,103) (Ta(j),j=1,5),
     &                         (Tao(j),j=1,5),(Tae(j),j=1,5),Vgen,
     &                         d_p,x_p*LUKPC,y_p*LUKPC,z_p*LUKPC
     &                         ,vx_p*VUKMS,vy_p*VUKMS,vz_p*VUKMS,mgen
     &                         ,GRVS,VIgen,ex31,avgen,avgen-aigen,z,
     &                         PAge,od_p,xo_p,yo_p,zo_p,rp,vr_p,vrot_p
     &                         ,l_p,b_p,vl_p,vb_p,lo_p,bo_p,vlo_p,vbo_p
     &                         ,Tao(3)/Tae(3),l2_p,b2_p
c                                 endif
c                              endif
                           endif
                           GenSel=GenSel+1
                           endif
c                           endif
c                           endif
c                           endif
c                           endif
c *** END CUSTOM SELECTIONS ***
                           d_p=d_p/LUKPC
                        endif
                        goto 602
                     endif
                  enddo
 602                           sumgen=sumgen+mgen
                  mtemp=mtemp+mgen
                  ngen=ngen+1
               enddo
               goto 601
            endif
         enddo
c *** Percentage done ***
 601         if (int((float(q)/float(10000000))*100.0d0).gt.per2) then
            per2=int((float(q)/float(10000000))*100.0d0)
c            if (inputdata.lt.2) then
               write(6,'(1I3,$)') per2
c            endif
         endif
      enddo 
c     72   
 102   format(74(1pE13.5))
 103    format(49(1pE13.5))
 104     format(17(1pE13.5)) 
      close(11)
c *** Finish ***
 603   write(6,*) '100 % complete'
      write(6,*) 'You generated',sumgen,' solar masses of stars!'
      write(6,*) 'You needed',mtotalneed,' solar masses of stars!'
      if (sumgen.gt.mtotalneed) then
         write(6,*) 'You have',sumgen-mtotalneed,' excess'
      else
         write(6,*) 'You have',mtotalneed-sumgen,' deficit'
      endif
      write(6,*) 'You generated',ngen,' stars',GenSel,' selected'
      close(30)
      close(10)

      END
