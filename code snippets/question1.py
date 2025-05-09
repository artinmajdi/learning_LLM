from typing import List
import pytest

def find_missing_ranges(nums: List[int], lower: int, upper: int) -> List[List[int]]:
    # Initialize result list
    result = []

    # Add sentinel values to handle edge cases
    nums = [lower - 1] + nums + [upper + 1]

    # Iterate through adjacent pairs in the array
    for i in range(1, len(nums)):
        # Calculate the gap between adjacent elements
        start = nums[i-1] + 1
        end = nums[i] - 1

        # If there's a gap, add the range to the result
        if start <= end:
            result.append([start, end])

    return result

def find_missing_ranges2(nums: List[int], lower: int, upper: int) -> List[List[int]]:
    n = len(nums)
    missing_ranges = []

    if n == 0:
        return [[lower, upper]]

    # Check for any missing numbers between the lower bound and nums[0].
    if lower < nums[0]:
        missing_ranges.append([lower, nums[0]-1])

    # Check for any missing numbers between successive elements of nums.
    for i in range(n - 1):
        if nums[i + 1] - nums[i] <= 1:
            continue
        missing_ranges.append([nums[i]+1, nums[i+1]-1])

    # Check for any missing numbers between the last element of nums and the upper bound.
    if upper > nums[-1]:
        missing_ranges.append([nums[-1]+1, upper])

    return missing_ranges

test_cases = [
    ([0, 1, 3, 50, 75], 0, 99, [[2, 2], [4, 49], [51, 74], [76, 99]]),
    ([-1], -1, -1, []),
    ([], 1, 10, [[1, 10]]),
    ([1, 2, 3, 4, 5], 1, 5, [])
]
@pytest.mark.parametrize("nums, lower, upper, expected", test_cases)
def test_find_missing_ranges(nums, lower, upper, expected):
    assert find_missing_ranges(nums, lower, upper) == expected
