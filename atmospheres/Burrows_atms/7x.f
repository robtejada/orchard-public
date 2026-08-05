
C  For 7X
C
        subroutine setp7x
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        COMMON /PDAT/ g(31),te(31,501),T10(31,501),nte(501),ng
C
        open(9,file='Burrows_data.dat')
C
C      zero out the arrays
        do 50 jg=1,31
        g(jg) = 0.d0
        do 50 jte=1,501
        te(jg,jte)=0.d0
        t10(jg,jte)=0.d0
50      continue
C
        ng = 0
        jg = 0
100     continue
        READ(9,*) x,y
c       if(x.eq.1.d0) write(*,*) x,y
        if(x.gt.1.d0) go to 200
        if(x.le.0.d0) go to 300
        jg = jg + 1
        ng = ng + 1
        nte(jg) = 0
        g(jg) = y
        zloh = x
        if(ng.gt.1) nte(ng-1) = jte
        jte = 0
        go to 100
C
200     continue
        jte = jte + 1
C        write(*,*) jg,jte,x,y
        te(jg,jte) = x
        t10(jg,jte) = y
        go to 100
C
300     continue
        nte(ng) = jte
        close(9)

C      print out the arrays
        do 350 jg=1,ng
C       write(*,*) g(jg)
        do 350 jte=1,nte(jg)
C       write(*,*) te(jg,jte), t10(jg,jte)
350      continue
C
        END
C

C  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C
        SUBROUTINE zslv7x(gsurf,j,teff,zt10)
        IMPLICIT DOUBLE PRECISION (A-H,O-Z)
        COMMON /PDAT/ g(31),te(31,501),T10(31,501),nte(501),ng
C
C       bracket the gravity:
        ig = 0
100     continue
        ig = ig + 1
        if(ig.gt.ng) go to 200
        if(gsurf.le.g(ig)) igrav = ig-1
        if(gsurf.gt.g(ig)) go to 100
        if(gsurf.le.g(ig)) go to 300
200     continue
        igrav = ig-1
300     continue
C
        igravhi=igrav+1
        igravlo=igrav
C
        if(igrav.eq.0) igravhi=2
        if(igrav.eq.0) igravlo=1
C
        if(igrav.eq.ng) igravhi=ng
        if(igrav.eq.ng) igravlo=ng-1
C
C   Take into account that g=3e3 only extends down to T10=4036.3
        if(igravhi.eq.3.and.zt10.le.4036.3d0) igravhi=igravhi+1
        if(igravlo.eq.3.and.zt10.le.4036.3d0) igravlo=igravlo-1
C
C       bracket the T10 for igravhi:
        it = 0
400     continue
        it = it + 1
        if(it.gt.nte(igravhi)) go to 500
        if(zt10.le.t10(igravhi,it)) it10 = it-1
        if(zt10.gt.t10(igravhi,it)) go to 400
        if(zt10.le.t10(igravhi,it)) go to 600
500     continue
        it10 = it-1
600     continue
C
        it10hi=it10+1
        it10lo=it10
C
        if(it10.eq.0) it10hi=2
        if(it10.eq.0) it10lo=1
C
        if(it10.eq.nte(igravhi)) it10hi=nte(igravhi)
        if(it10.eq.nte(igravhi)) it10lo=nte(igravhi)-1
C
        tehi = te(igravhi,it10hi) + (zt10-t10(igravhi,it10hi))*
     1   (te(igravhi,it10lo)-te(igravhi,it10hi))/
     2   (t10(igravhi,it10lo)-t10(igravhi,it10hi))
C
C       bracket the T10 for igravlo:
        it = 0
700     continue
        it = it + 1
        if(it.gt.nte(igravlo)) go to 800
        if(zt10.le.t10(igravlo,it)) it10 = it-1
        if(zt10.gt.t10(igravlo,it)) go to 700
        if(zt10.le.t10(igravlo,it)) go to 900
800     continue
        it10 = it-1
900     continue
C
        it10hi=it10+1
        it10lo=it10
C
        if(it10.eq.0) it10hi=2
        if(it10.eq.0) it10lo=1
C
        if(it10.eq.nte(igravlo)) it10hi=nte(igravlo)
        if(it10.eq.nte(igravlo)) it10lo=nte(igravlo)-1
C
        telo = te(igravlo,it10hi) + (zt10-t10(igravlo,it10hi))*
     1   (te(igravlo,it10lo)-te(igravlo,it10hi))/
     2   (t10(igravlo,it10lo)-t10(igravlo,it10hi))
C
C     now interpolate in log g:
C
        teff = telo + dlog(gsurf/g(igravlo))*
     1    (tehi-telo)/dlog(g(igravhi)/g(igravlo))


        RETURN
        END
C
C  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



