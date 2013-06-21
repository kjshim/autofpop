import autodora
import andlib
import logging
from numpy import array


# andlib
def test_andlib_gettmpdir():
    tmpdir = andlib.GetTempDirectory()
    assert tmpdir == "/storage/emulated/legacy"

## image recognition
def test_img1():
    im = andlib.ReadRawImage('tests/img_20130612_1.raw')

    expected_tbl = array([[32, 64,  4, 32,  4, 32],
                         [64, 32,  2, 32,  2, 16],
                         [32,  2, 32, 16,  8,  2],
                         [ 2,  2,  4, 64, 64, 16],
                         [ 4, 16,  8, 16, 64,  2]])
    tbl = autodora.ReadTable(im)

    assert (tbl == expected_tbl).all()

def test_img2():
    im = andlib.ReadRawImage('tests/img_20130612_2.raw')
    expected_tbl = array([[ 8, 64, 32, 64, 16, 32],
                           [ 2, 16, 64, 16,  2,  2],
                           [64, 32, 64,  2, 64, 64],
                           [16, 64, 16, 32,  2, 16],
                           [ 8,  4,  4,  2,  8,  2]])
    tbl = autodora.ReadTable(im)
    print tbl
    assert (tbl == expected_tbl).all()


def test_img3():
    im = andlib.ReadRawImage('tests/img_20130613.raw')
    expected_tbl = array([[32, 16,  2,  4, 32, 64],
                         [64,  4,  4, 16,  4,  2],
                         [ 4,  2,  8,  2,  2,  4],
                         [32, 32, 64, 32, 64, 64],
                         [ 2,  4,  2, 16, 64,  2]])
    tbl = autodora.ReadTable(im)
    print tbl
    assert (tbl == expected_tbl).all()


def test_img_fivethree():
    im = andlib.ReadRawImage('tests/img_5.3.raw')
    expected_tbl = array([[16,  4, 16,  4, 64, 32],
                           [32, 64, 64, 16,  4, 16],
                           [64,  4,  8, 16,  4, 32],
                           [ 4, 32,  8,  2, 64, 64],
                           [32,  8,  2,  8, 32,  4]])
    tbl = autodora.ReadTable(im)
    assert (tbl == expected_tbl).all()

def test_img_normal():
    im = andlib.ReadRawImage('tests/img_normal.raw')
    tbl = autodora.ReadTable(im)
    logging.debug("\n"+str(tbl))
    expected_tbl = array([[32,  4, 16,  4, 32,  2],
                           [64, 16,  8, 64, 16,  4],
                           [32, 16,  2,  2,  4, 64],
                           [ 2,  2, 32, 32,  4, 32],
                           [ 4,  2, 16, 16,  8,  2]])

    positions = ([63, 182, 300, 418, 536, 654], [600, 719, 837, 955, 1073])

    assert autodora.CheckPositions(im) == positions
    assert (tbl == expected_tbl).all()


def test_img4():
    im = andlib.ReadRawImage('tests/img_0617.raw')
    expected_tbl = array([[64,  2,  8,  2,  4, 32],
       [ 2,  8,  2, 16,  4,  4],
       [ 2,  4, 32,  2, 16, 16],
       [ 4,  2, 16,  2,  2,  4],
       [64,  8,  4,  8,  2,  8]])
    tbl = autodora.ReadTable(im)
    print tbl
    assert (tbl == expected_tbl).all()

def test_imgbig1():
    im = andlib.ReadRawImage('tests/img_bightc.raw')
    expected_tbl = array([[ 2, 32,  2, 32,  4,  8],
       [32, 32,  8, 32,  8,  2],
       [ 8, 16, 16,  4, 16, 64],
       [32,  2,  4, 64, 64, 32],
       [ 4,  8, 16,  4, 64, 16]])
    tbl = autodora.ReadTable(im)
    print tbl
    assert (tbl == expected_tbl).all()


def test_imgbig2():
    im = andlib.ReadRawImage('tests/img_opG.raw')
    expected_tbl =array([[32, 32, 64, 32, 32, 64],
       [16,  8,  4,  2,  2,  4],
       [32, 16,  8, 64,  4, 32],
       [ 4, 32, 64, 64,  2,  8],
       [ 2, 16,  2, 16,  2,  8]])
    tbl = autodora.ReadTable(im)
    print tbl
    assert (tbl == expected_tbl).all()