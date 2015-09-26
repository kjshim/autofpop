#%pylab inline
VERSION = "1.0"

from numpy import *


from andlib import SwipeLine, GetScreen
import traceback
import sys
import os
import subprocess
import heapq
import time
import hashlib
import logging
import logging.config
import andlib

class TableDetectionFailed(Exception):
    pass

R,G,B,H,A,Y = 1<<1, 1<<2, 1<<3, 1<<4, 1<<5, 1<<6

STRMAP = {
    Y:"Y",
    B:"B",
    R:"R",
    H:"H",
    A:"A",
    G:"G",
}

newcmap = {
    (5,1,3):H,
    (4,1,2):H,
    (5,5,2):Y,
    (4,4,2):Y,
    (5,5,3):Y,
    (2,5,2):G,
    (2,4,2):G,
    (2,4,3):G,
    (2,5,3):G,
    (3,2,3):A,
    (3,1,3):A,
    (3,1,2):A,
    (2,1,2):A,
    (5,2,2):R,
    (5,2,4):R,
    (5,3,2):R,
    (4,2,1):R,
    (2,3,5):B,
    (2,3,4):B,
    (2,4,5):B,
    (2,4,3):G,
}

def diffvector(v):
    result = []
    for i in xrange(len(v)-1):
        result.append(v[i+1] - v[i])
    return result

def CheckPositions(im):
    h, w, _ = im.shape
    mindist = w/20
    est_dist = w/6

    yarr = array([im[y, :, 0].var() for y in range(0,h)])
    xarr = array([im[:, x, 0].var() for x in range(0,w)])

    mean_y = mean(yarr)
    mean_x = mean(xarr)
    ## y
    logging.debug("Checking y positions...")
    known_points = []
    points = [ p for p in sorted(enumerate(yarr), key=lambda x:x[1]) if p > 0. and p[0] > h * 0.4 ]
    while points:
        cury = points[0][0]
        if points[0][1] > mean_y:
            break
        known_points.append(cury)
        points = [ p for p in points if abs(p[0] - cury) > mindist ]
    known_points = sorted(known_points)

    y_edges = []
    n_edge = 6

    candidates = []
    for i in xrange(len(known_points) - n_edge + 1):
        v2 = arange(0, est_dist*n_edge, est_dist) + known_points[i]
        # find nearest points
        v1 = []

        for anchor in v2:
            dists = [ (abs(p-anchor), p) for p in known_points ]
            v1.append(sorted(dists)[0][1])

        if len(set(v1)) != len(v1):
            continue

        midpoints = []
        for j in xrange(len(v1)-1):
            midpoints.append( yarr[v1[j]:v1[j+1]].mean() )

        #v1 = array(known_points[i:i+n_edge])
        score =  sum(abs(v1-v2))
        score2 = array(midpoints).var()
        if score > w / 7 or math.isnan(score2):
            continue
        logging.debug(str((score, score2, v1)))
        candidates.append((score2,v1))

        logging.debug(str((v1, v2, abs(v1-v2))))

    y_edges = sorted(candidates)[0][1]
    if not y_edges:
        raise TableDetectionFailed("Can't find grid y edges")


    ##x
    logging.debug("Checking x positions...")
    known_points = []
    points = [ p for p in sorted(enumerate(xarr), key=lambda x:x[1]) if p > 0. ]
    while points:
        curx = points[0][0]
        if points[0][1] > mean_x:
            break
        known_points.append(curx)
        points = [ p for p in points if abs(p[0] - curx) > mindist ]

    known_points = sorted(known_points)

    x_edges = []
    n_edge = 7
    candidates = []
    for i in xrange(len(known_points) - n_edge + 1):
        v2 = arange(0, est_dist*n_edge, est_dist) + known_points[i]
        # find nearest points
        v1 = []
        for anchor in v2:
            dists = [ (abs(p-anchor), p) for p in known_points ]
            v1.append(sorted(dists)[0][1])

        #v1 = array(known_points[i:i+n_edge])
        score =  sum(abs(v1-v2))
        candidates.append((score,v1))

        logging.debug(str((v1, v2, abs(v1-v2))))

    x_edges = sorted(candidates)[0][1]
    if not x_edges:
        raise TableDetectionFailed("Can't find grid x edges")


    xs = []
    ys = []
    for i in xrange(len(y_edges) - 1):
        ys.append((y_edges[i] + y_edges[i+1])/2)

    for i in xrange(len(x_edges) - 1):
        xs.append((x_edges[i] + x_edges[i+1])/2)

    logging.debug(str(("XS",xs,"YS",ys)))

    return xs, ys

def ReadTable(im, xs=None, ys=None):
    global KNOWN_POSITIONS

    if not xs or not ys:
        XS, YS = CheckPositions(im)
    else:
        XS, YS = xs, ys

    if (not XS) or (not YS):
        raise TableDetectionFailed("Can find grid")

    result = zeros((5,6),dtype=int32)

    checkpixels = int(round((XS[1] - XS[0])/12.))
    for j, x in enumerate(XS):
        for i, y in enumerate(YS):
            r = int(round(im[y-checkpixels:y+checkpixels, x-checkpixels:x+checkpixels, 0].flatten().mean() /50. ))
            g = int(round(im[y-checkpixels:y+checkpixels, x-checkpixels:x+checkpixels, 1].flatten().mean() /50.))
            b = int(round(im[y-checkpixels:y+checkpixels, x-checkpixels:x+checkpixels, 2].flatten().mean() /50.))
            logging.debug(str((i,j, x, y, r, g, b)))
            #print(str((i,j, x, y, r, g, b)))
            result[i,j] = newcmap[(r,g,b)]

            # else:
            #     sugc = sorted([ ((k[0] - r)**2 + (k[1] - g)**2 +(k[2] - b)**2,
            #         v)
            #         for k, v in newcmap.items()])[0]


            logging.debug(str((i,j, x, y, r, g, b)))
    return result


testtable = \
    array([[64, 16,  4, 32, 32, 64],
           [64,  2, 16,  2, 64, 16],
           [ 2, 64,  4,  4, 64,  4],
           [ 2, 32,  2,  2,  2, 32],
           [64, 64, 16, 16, 64,  2]])

range_rows = range(5)
range_cols = range(6)

def EvaluateTable(tt):
    score = 0
    t = tt.copy()
    removed_colors = set([])
    while True:
        lastscore = score
        result = zeros(t.shape,dtype=int32)
        found = False
        for i in range_rows:
            for j in range_cols:
                if i + 2 < t.shape[0]:
                    if t[i,j] == t[i+1,j] == t[i+2,j] and t[i,j]>0:
                        result[i,j] = t[i,j]
                        result[i+1,j] = t[i,j]
                        result[i+2,j] = t[i,j]
                        found = True
                if j + 2 < t.shape[1]:
                    if t[i,j] == t[i,j+1] == t[i,j+2] and t[i,j]>0:
                        result[i,j] = t[i,j]
                        result[i,j+1] = t[i,j]
                        result[i,j+2] = t[i,j]
                        found = True
        if found:
            cur_removed_colors = set(t[result > 0].flatten())
            for i in range_rows:
                for j in range_cols:
                    if result[i,j]>0:
                        score += 1
                        t[i,j] = 0
            for i in range_rows:
                for j in range_cols:
                    # remove
                    if t[i,j] == 0:
                        ii = i
                        while ii > 0 and t[ii-1,j] != 0:
                            t[ii-1,j], t[ii,j] = t[ii,j], t[ii-1,j]
                            ii -= 1
            removed_colors.update(cur_removed_colors)
        if lastscore == score: break

    score = score * len(removed_colors)
    return score


directions = [
    (1,0),
    (0,1),
    (-1,0),
    (0,-1),
    (1,1),
    (-1,1),
    (1,-1),
    (-1,-1),
]
"""
xs,ys = autodora.CheckPositions(im)
xv = [x for x in xs for y in ys]
yv = [y for x in xs for y in ys]
imshow(im)
scatter(xv,yv)
"""
def SearchSolution(t, steps = 10000, maxmove=20):
    # generate initial state
    h = []
    knownstates = set()
    initscore = (-EvaluateTable(t), 1)
    for i in xrange(t.shape[0]):
        for j in xrange(t.shape[1]):
            h.append((initscore, [(i,j)], t))

    bestscore = (0, 0)
    bestmv = []
    besttable = None

    for i in xrange(steps):
        sc, moves, curt = heapq.heappop(h)
        curhash = hashlib.md5(curt.data+"%d,%d"%moves[0]).hexdigest()
        if curhash in knownstates:
            continue
        else:
            knownstates.add(curhash)


        if len(moves) > maxmove: continue
        ci, cj = moves[0]
        for di, dj in directions:
            ni = ci + di
            nj = cj + dj
            if len(moves) >= 2:
                if (ni, nj) == (moves[1][0], moves[1][1]):
                    continue
            if 0 <= ni < t.shape[0] and 0 <= nj < t.shape[1]:
                newt = curt.copy()
                newt[ni,nj], newt[ci,cj] = newt[ci,cj], newt[ni,nj]

                # check if score possibly change
                score_can_change = True
                if curt[ci,cj] == curt[ni,nj]:
                    score_can_change = False
                #if newt[ci, cj]

                nmoves = [(ni,nj)] + moves
                if score_can_change:
                    nscore = (-EvaluateTable(newt), len(nmoves))
                else:
                    nscore = sc

                heapq.heappush(h, (nscore, nmoves, newt))

                if bestscore > nscore:
                    bestscore = nscore
                    bestmv = nmoves
                    besttable = newt
                    #print i, bestscore, bestmv
                    # print newt
    return bestscore, list(reversed(bestmv)), besttable


def ConvertMv(mv, xs, ys):
    curidx = 0
    curdir = (0,0)
    result = []
    i, j = mv[curidx]
    #result.append((xs[j], ys[i]))

    while curidx + 1 < len(mv):
        ni, nj = mv[curidx+1]
        nextdir =ni - i, nj - j
        if nextdir != curdir:
            result.append((xs[j], ys[i]))
            #result.append((i,j))
            curdir = nextdir
        curidx += 1
        i, j = ni, nj
    result.append((xs[j], ys[i]))
    return result


    result = []

    lasti = 0
    lastj = 0
    lastdir = (0,0)
    for i, j in mv:
        if not first:
            curdir = (i -  lasti, j - lastj)
        result.append((xs[j], ys[i]))
        first, lasti, lastj = False, i, j

    return result

def DescribeResult(tt):
    score = 0
    t = tt.copy()
    removed_colors = set([])
    while True:
        lastscore = score
        result = zeros(t.shape,dtype=int32)
        found = False
        for i in range_rows:
            for j in range_cols:
                if i + 2 < t.shape[0]:
                    if t[i,j] == t[i+1,j] == t[i+2,j] and t[i,j]>0:
                        result[i,j] = t[i,j]
                        result[i+1,j] = t[i,j]
                        result[i+2,j] = t[i,j]
                        found = True
                if j + 2 < t.shape[1]:
                    if t[i,j] == t[i,j+1] == t[i,j+2] and t[i,j]>0:
                        result[i,j] = t[i,j]
                        result[i,j+1] = t[i,j]
                        result[i,j+2] = t[i,j]
                        found = True
        if found:
            for i in range_rows:
                for j in range_cols:
                    if result[i,j]>0:
                        score += 1
                        t[i,j] = 0
                        removed_colors.add(result[i,j])
            for i in range_rows:
                for j in range_cols:
                    # remove
                    if t[i,j] == 0:
                        ii = i
                        while ii > 0 and t[ii-1,j] != 0:
                            t[ii-1,j], t[ii,j] = t[ii,j], t[ii-1,j]
                            ii -= 1
        if lastscore == score: break
    if H in removed_colors:
        removed_colors.remove(H)

    return "Removed : %s, score : %d" % (str(removed_colors), score)


def Easy(minremove=13):
    try:
        im = GetScreen()
    except Exception, e:
        logging.debug("Exception:" + traceback.format_exc())
        return

    try:
        xs, ys = CheckPositions(im)
        t = ReadTable(im)
    except Exception,e:
        raise TableDetectionFailed(str(e))
    score, mv, btbl = SearchSolution(t, 5000, 15)
    SwipeLine(ConvertMv(mv, xs, ys))
    logging.info(DescribeResult(btbl))


LOGCONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level':'INFO',
            'class':'logging.StreamHandler',
        },
        'file':{
            'level':'DEBUG',
            'class':'logging.FileHandler',
            'formatter':'standard',
            'filename':'Autodora.Log',
            'mode':'w'
        }
    },
    'loggers': {
        'root': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },

    }
    }


if __name__ == '__main__':
    logging.config.dictConfig(LOGCONFIG)
    andlib.Init()
    try:
        devid = andlib.GetDeviceId()
        if not devid:
            logging.info("Device is not connected. \r\nPlease connect your device via USB and make sure that USB debugging enabled.")
            raw_input()
            sys.exit()
    except Exception, e:
        logging.debug("Exception:" + traceback.format_exc())
        print "Error occured. Please send Autodora.Log to autodoragon@gmail.com"
        raw_input()
        sys.exit()
    logging.info("Autodora started. version : %s" % VERSION)

    nturn = 0

    while True:
        try:
            t1 = time.time()
            Easy(8)
            logging.info("Turn %d: took %.2f sec." %(nturn, time.time() - t1))
            nturn += 1
        except TableDetectionFailed, e:
            logging.debug(str(e))
        except Exception, e:
            logging.debug("Exception:" + traceback.format_exc())
            print "Error occured. Please send Autodora.Log to autodoragon@gmail.com"
            raw_input()
            break
        time.sleep(2)

#%timeit EvaluateTable(testtable)