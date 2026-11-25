"""CPU mirror of RGBA MSDF distance-union and coverage reconstruction."""
from __future__ import annotations

import math
import numpy as np


def bilinear(field, x, y):
    x, y = np.asarray(x)-0.5, np.asarray(y)-0.5
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x-ix, y-iy
    if field.ndim == 3:
        fx, fy = fx[...,None], fy[...,None]
    def at(x, y):
        return field[np.clip(y,0,field.shape[0]-1),np.clip(x,0,field.shape[1]-1)]
    return ((at(ix,iy)*(1-fx)+at(ix+1,iy)*fx)*(1-fy)
            +(at(ix,iy+1)*(1-fx)+at(ix+1,iy+1)*fx)*fy)


def halfplane_area(distance, dx, dy):
    """Closed form of square clipping by one half-plane (shader RGB repeats)."""
    a, b = np.abs(dx), np.abs(dy)
    extent = 0.5*(a+b)
    t = distance+extent
    positive_square = lambda value: np.maximum(value,0)**2
    denominator = 2*a*b
    numerator = positive_square(t)-positive_square(t-a)-positive_square(t-b)+positive_square(t-a-b)
    result = np.divide(numerator,denominator,out=np.zeros_like(distance),where=denominator>0)
    slope = np.maximum(a,b)
    linear = np.divide(distance,slope,out=np.zeros_like(distance),where=slope>0)+0.5
    result = np.where(denominator==0,linear,result)
    result = np.where(distance>=extent,1,result)
    result = np.where(distance<=-extent,0,result)
    # Constant zero is inside, matching clip_coverage's >= predicate.
    result = np.where((extent==0)&(distance==0),1,result)
    return np.clip(result,0,1)


def union_render(atlas, rects, width, height, opacity=1.0, *, px_range=8.0):
    """rects carry x,y,w,h,u0,v0,u1,v1,RGBA,halo RGBA,width,rotation."""
    normalized = atlas.astype(np.float64)/255.0
    ah, aw = atlas.shape[:2]
    k = 1
    for rect in rects:
        _,_,w,h,u0,v0,u1,v1,*_ = rect
        angle=rect[17] if len(rect)>17 else 0.0
        sx, sy = aw*(u1-u0)/w, ah*(v1-v0)/h
        c,s=math.cos(angle),math.sin(angle)
        k=max(k,math.ceil(max(math.hypot(sx*c,sy*s),math.hypot(sx*s,sy*c))))
    fill=np.full((height*k,width*k),-65504.0,dtype=np.float64)
    halo=fill.copy()
    for rect in rects:
        x,y,w,h,u0,v0,u1,v1=rect[:8]
        angle=rect[17] if len(rect)>17 else 0.0
        c,s=math.cos(angle),math.sin(angle)
        cx,cy=x+w*0.5,y+h*0.5
        rx,ry=(abs(c)*w+abs(s)*h)*0.5,(abs(s)*w+abs(c)*h)*0.5
        x0,x1=max(0,math.floor((cx-rx)*k)),min(width*k,math.ceil((cx+rx)*k))
        y0,y1=max(0,math.floor((cy-ry)*k)),min(height*k,math.ceil((cy+ry)*k))
        xx,yy=np.meshgrid((np.arange(x0,x1)+0.5)/k-cx,(np.arange(y0,y1)+0.5)/k-cy)
        lx,ly=c*xx+s*yy+w*0.5,-s*xx+c*yy+h*0.5
        inside=(lx>=0)&(lx<w)&(ly>=0)&(ly<h)
        sx,sy=aw*(u1-u0)/w,ah*(v1-v0)/h
        ax,ay=aw*u0+lx*sx,ah*v0+ly*sy
        sample=bilinear(normalized,ax,ay)
        median=np.median(sample[...,:3],axis=2)
        display_range=px_range/max(math.hypot(sx*c,sy*s),math.hypot(sx*s,sy*c))
        distance=(median-0.5)*display_range
        halo_distance=(sample[...,3]-0.5)*display_range+rect[16]
        for target,values,active in [(fill,distance,rect[11]>0),(halo,halo_distance,rect[15]>0 and rect[16]>0)]:
            values=np.clip(values,-65504,65504).astype(np.float16).astype(np.float64)
            values=np.where(inside&active,values,-65504)
            target[y0:y1,x0:x1]=np.maximum(target[y0:y1,x0:x1],values)
    xx,yy=np.meshgrid(np.arange(width)+0.5,np.arange(height)+0.5)
    coverages=[]
    for target in (fill,halo):
        area=np.zeros((height,width))
        for y in range(k):
            for x in range(k):
                tx=(xx+(x+0.5)/k-0.5)*k
                ty=(yy+(y+0.5)/k-0.5)*k
                distance=bilinear(target,tx,ty)
                dx=bilinear(target,tx+0.5,ty)-bilinear(target,tx-0.5,ty)
                dy=bilinear(target,tx,ty+0.5)-bilinear(target,tx,ty-0.5)
                gradient_scale=np.maximum(1.0,np.hypot(dx,dy)*k)
                dx,dy=dx/gradient_scale,dy/gradient_scale
                area+=halfplane_area(distance,dx,dy)
        coverages.append((area/(k*k)).astype(np.float32))
    return coverages


def contours(coverage):
    """Marching squares at the 0.5 iso-value in display-pixel coordinates."""
    padded=np.pad(coverage,1,constant_values=0)
    segments=[]
    for y in range(padded.shape[0]-1):
        for x in range(padded.shape[1]-1):
            points=np.array([[x-0.5,y-0.5],[x+0.5,y-0.5],[x+0.5,y+0.5],[x-0.5,y+0.5]])
            values=np.array([padded[y,x],padded[y,x+1],padded[y+1,x+1],padded[y+1,x]],dtype=float)-0.5
            crossings={}
            for edge in range(4):
                after=(edge+1)%4
                if (values[edge]>=0)!=(values[after]>=0):
                    t=values[edge]/(values[edge]-values[after])
                    crossings[edge]=points[edge]+t*(points[after]-points[edge])
            if len(crossings)==2:
                segments.append(list(crossings.values()))
            elif len(crossings)==4:
                # Bilinear asymptotic decider resolves the ambiguous cell.
                q=values[0]*values[2]-values[1]*values[3]
                pairs=((0,1),(2,3)) if q>=0 else ((0,3),(1,2))
                segments.extend([[crossings[a],crossings[b]] for a,b in pairs])
    return np.asarray(segments,dtype=float).reshape(-1,2,2)


def point_distances(points, segments):
    a,b=segments[:,0],segments[:,1]
    v=b-a
    t=np.clip(np.sum((points[:,None]-a)*v,axis=2)/np.maximum(np.sum(v*v,axis=1),np.finfo(float).tiny),0,1)
    d=np.linalg.norm(points[:,None]-(a+t[...,None]*v),axis=2)
    return d.min(axis=1)


def hausdorff(left, right):
    """Continuous polyline Hausdorff, including nearest-segment switches."""
    a,b=contours(left),contours(right)
    if not len(a) or not len(b):
        return 0.0 if not len(a) and not len(b) else float('inf')
    return max(directed_hausdorff(a,b),directed_hausdorff(b,a))


def directed_hausdorff(source,target):
    maximum=float(point_distances(source.reshape(-1,2),target).max())
    ta,tb=target[:,0],target[:,1]
    tv=tb-ta
    lengths=np.sum(tv*tv,axis=1)
    lengths=np.maximum(lengths,np.finfo(float).tiny)
    for sa,sb in source:
        sv=sb-sa
        proj0=np.sum((sa-ta)*tv,axis=1)/lengths
        proj1=np.sum(sv*tv,axis=1)/lengths
        breaks=[0.,1.]
        for boundary in (0.,1.):
            switches=np.divide(boundary-proj0,proj1,out=np.full_like(proj0,np.inf),where=proj1!=0)
            breaks.extend(switches[(switches>0)&(switches<1)])
        breaks=np.unique(breaks)
        for low,high in zip(breaks,breaks[1:]):
            mid=(low+high)*0.5
            projection=proj0+proj1*mid
            base=sa-ta
            slopes=np.broadcast_to(sv,base.shape).copy()
            before=projection<=0
            after=projection>=1
            base[after]=sa-tb[after]
            middle=~(before|after)
            base[middle]-=proj0[middle,None]*tv[middle]
            slopes[middle]-=proj1[middle,None]*tv[middle]
            coefficients=np.stack([np.sum(base*base,axis=1),2*np.sum(base*slopes,axis=1),np.sum(slopes*slopes,axis=1)],axis=1)
            i,j=np.triu_indices(len(target),1)
            differences=coefficients[i]-coefficients[j]
            c,linear,quadratic=differences.T
            discriminant=linear*linear-4*quadratic*c
            candidates=[low,high]
            linear_roots=np.divide(-c,linear,out=np.full_like(c,np.inf),where=(quadratic==0)&(linear!=0))
            candidates.extend(linear_roots[(linear_roots>low)&(linear_roots<high)])
            for sign in (-1,1):
                roots=np.divide(-linear+sign*np.sqrt(np.maximum(discriminant,0)),2*quadratic,out=np.full_like(c,np.inf),where=(quadratic!=0)&(discriminant>=0))
                candidates.extend(roots[(roots>low)&(roots<high)])
            if candidates:
                points=sa+np.asarray(candidates)[:,None]*sv
                maximum=max(maximum,float(point_distances(points,target).max()))
    return maximum


def corner_segments(segments, corner, vertices):
    """Clip to the outline vertex's Voronoi region; no chosen pixel radius."""
    output=[]
    for a,b in segments:
        low,high=0.,1.
        direction=b-a
        for other in vertices:
            normal=other-corner
            bound=(np.dot(other,other)-np.dot(corner,corner))*0.5
            value=bound-np.dot(a,normal)
            slope=np.dot(direction,normal)
            if slope>0: high=min(high,value/slope)
            elif slope<0: low=max(low,value/slope)
            elif value<0: high=-1
        if low<=high:
            output.append([a+low*direction,a+high*direction])
    return np.asarray(output).reshape(-1,2,2)


def corner_hausdorff(left,right,corner,vertices):
    a=corner_segments(contours(left),corner,vertices)
    b=corner_segments(contours(right),corner,vertices)
    if not len(a) or not len(b):
        raise AssertionError('corner must contain both measured iso-contours')
    return max(directed_hausdorff(a,b),directed_hausdorff(b,a))
