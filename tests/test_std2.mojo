import numojo as nm
from numojo.core.ndarray import NDArray
from numojo.math.statistics.stats import std
from testing import assert_equal, assert_raises, assert_almost_equal, assert_true


fn test_raises() raises -> None:
    print('Testing Raises')
    with assert_raises(contains="Cannot find Standart Deviation of an empty array"):
        std(NDArray[nm.f64]())
    with assert_raises(contains="Axis cannot be greater than the rank of the array"):
        var A = NDArray[nm.f64]()
        std(A, 1)
    print('Success')

fn test_equal() raises -> None:
    print("Testing Equal")
    var A = nm.NDArray[nm.f64](data = List[SIMD[nm.f64, 1]](5.0), shape = List[Int](1, 1))
    assert_equal(std(A, 0), 5.0)
    print("Success")
    
    print("Testing Equal with default axis")
    var B = nm.NDArray[nm.f64](data = List[SIMD[nm.f64, 1]](5.0), shape = List[Int](1, 1))
    assert_equal(std(B), 5.0)
    print("Success")

    print("Testing identical elements")
    var C = nm.NDArray[nm.f64](data = List[SIMD[nm.f64, 1]](5.0), shape = List[Int](1, 1))
    assert_equal(std(C, 0), 0.0)
    print("Success")

    print("Testing with a list of positive numbers")
    var D = nm.NDArray[nm.f64](data = List[SIMD[nm.f64, 1]](1.0, 2.0, 3.0, 4.0, 5.0), shape = List[Int](5, 1))
    assert_equal(std(D, 0), 1.4142135623730951)
    print("Success")

    print("Testing with a list of negative numbers")
    var E = nm.NDArray[nm.f64](data = List[SIMD[nm.f64, 1]](-1.0, -2.0, -3.0, -4.0, -5.0), shape = List[Int](5, 1))
    assert_equal(std(E, 0), 1.4142135623730951)
    print("Success")

    print("Testing with a mixed list of positive and negative numbers")
    var F = nm.NDArray[nm.f64](data = List[SIMD[nm.f64, 1]](-1.0, 1.0, -2.0, 2.0, -3.0, 3.0), shape = List[Int](5, 1))
    assert_equal(std(F, 0), 2.160246899469287)
    print("Success")

    var large_list = NDArray[nm.f64](10000, random=False)
    for i in range(1, 10001):
        large_list[i] = i

    print("Testing with a large list of numbers")
    assert_equal(std(large_list), 2886.751331514372)
    print("Success")


