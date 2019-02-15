### 1 - Hack the Algorithm Interview

#### 627. 最长回文串
```
class Solution:
    """
    @param s: a string which consists of lowercase or uppercase letters
    @return: the length of the longest palindromes that can be built
    """
    def longestPalindrome(self, s):
        # write your code here
        # 基本思想就是s中判断有多少个出现次数为奇数的字符
        odd_chars = set()
        for ch in s:
            if ch in odd_chars:
                odd_chars.remove(ch)
            else:
                odd_chars.add(ch)
        
        if not odd_chars:
            return len(s)
        else:
            # 减去出现次数为奇数的字符，最多可以保留一个
            return len(s) - len(odd_chars) + 1
```

#### 13. 字符串查找
```
class Solution:
    """
    @param: source: source string to be scanned.
    @param: target: target string containing the sequence of characters to match
    @return: a index to the first occurrence of target in source, or -1  if target is not part of source.
    """
    def strStr(self, source, target):
        # write your code here
        if not source:
            return 0 if not target else -1
        if not target:
            return 0
        
        for i in range(len(source) - len(target) + 1):
            j = 0
            while j < len(target):
                if source[i + j] != target[j]:
                    break
                j += 1
            else:
                return i
        
        return -1
```

#### 415. 有效回文串
```
class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """
    def isPalindrome(self, s):
        # write your code here
        if not s:
            return True
        
        l, r = 0, len(s) - 1
        while l < r:
            # ch必须是字母或者数字
            while l < r and not (s[l].isalpha() or s[l].isdigit()):
                l += 1
            while l < r and not (s[r].isalpha() or s[r].isdigit()):
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        
        return True
```

#### 200. 最长回文子串
```
class Solution:
    """
    @param s: input string
    @return: the longest palindromic substring
    """
    def longestPalindrome(self, s):
        # write your code here
        n = len(s)
        # dp[i][j]定义：s的i到j（包括j）是否是回文
        dp = [[False] * n for _ in range(n)]
        max_len = l = r = 0
        
        for i in range(n - 1, -1, -1):
            dp[i][i] = True
            for j in range(i + 1, n):
                dp[i][j] = s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1])
                if dp[i][j] and max_len < j - i + 1:
                    max_len = j - i + 1
                    l, r = i, j
        
        return s[l:r + 1]
```

#### 667. 最长的回文序列
```
class Solution:
    """
    @param s: the maximum length of s is 1000
    @return: the longest palindromic subsequence's length
    """
    def longestPalindromeSubseq(self, s):
        # write your code here
        if not s:
            return 0

        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
        return dp[0][-1]
```

#### 841. 字符串替换
```
class Solution:
    """
    @param a: The A array
    @param b: The B array
    @param s: The S string
    @return: The answer
    """
    def stringReplace(self, a, b, s):
        # Write your code here
        # 给定两个相同大小的字符串数组A和B
        # 再给一个字符串S，所有出现在S里的子串A都要替换成B。
        # 注意：从左往右，能替换的必须替换
        # 如果有多种替换方案，替换更长的优先。替换后的字符不能再做替换
        len_a, len_b, len_s = len(a), len(b), len(s)
        a_lengths = set()
        a_word_to_index = {}
        for i in range(len_a):
            a_lengths.add(len(a[i]))
            a_word_to_index[a[i]] = i
            
        # 核心：将a中所有词的长度从大到小排列
        a_lengths = sorted(a_lengths, reverse=True)
        
        res = ''
        i = 0
        while i < len_s:
            for each_length in a_lengths:
                j = i + each_length
                if j > len_s:
                    continue
                
                curr = s[i:j]
                if curr in a:
                    res += b[a_word_to_index[curr]]
                    i = j
                    break
            # 说明在此次固定i的循环中并没有找到能够替换s中substr的a中子串
            # 没有break过
            # 所以此时i累加1就好了
            else:
                res += s[i]
                i += 1
        
        return res
```

#### 594. 字符串查找 II
```
class Solution:
    """
    @param: source: A source string
    @param: target: A target string
    @return: An integer as index
    """
    def strStr2(self, source, target):
        # write your code here
        # Robin-Karp：对target串求hash code
        if source is None or target is None:
            return -1
        if not source:
            return 0 if not target else -1
        if not target:
            return 0
        
        len_s = len(source)
        len_t = len(target)
        
        M = 10 ** 6
        power = 1
        for i in range(len_t):
            power = (power * 31) % M
        
        target_code = 0
        for i in range(len_t):
            target_code = (target_code * 31 + ord(target[i])) % M
        
        curr_code = 0
        for i in range(len_s):
            curr_code = (curr_code * 31 + ord(source[i])) % M
            if i < len_t - 1:
                continue
            
            if i >= len_t:
                curr_code -= (ord(source[i - len_t]) * power) % M
                if curr_code < 0:
                    curr_code += M
            
            if curr_code == target_code:
                if source[i - len_t + 1:i + 1] == target:
                    return i - len_t + 1
        
        return -1
```

### 2 - Binary Search & LogN Algorithm

#### 458. 目标最后位置
```
class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
    def lastPosition(self, nums, target):
        # write your code here
        # 给一个升序数组，找到target最后一次出现的位置，如果没出现过返回-1
        if not nums:
            return -1

        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] <= target:
                start = mid
            else:
                end = mid - 1

        if nums[end] == target:
            return end
        if nums[start] == target:
            return start
        return -1
```

#### 585. 山脉序列中的最大值
```
class Solution:
    """
    @param nums: a mountain sequence which increase firstly and then decrease
    @return: then mountain top
    """
    def mountainSequence(self, nums):
        # write your code here
        # 给n个整数的山脉数组，即先增后减的序列，找到山顶（最大值）
        if not nums:
            return -2 ** 31
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] > nums[mid - 1]:
                start = mid
            else:
                end = mid - 1
        
        if nums[start] > nums[end]:
            return nums[start]
        return nums[end]
```

#### 460. 在排序数组中找最接近的K个数
```
class Solution:
    """
    @param A: an integer array
    @param target: An integer
    @param k: An integer
    @return: an integer array
    """
    def kClosestNumbers(self, A, target, k):
        # write your code here
        # 这道题的思路是先通过二分找到第一个大于等于target的数
        # 然后以这个数字为中心两根指针直到窗口大小为k
        if not A or not k:
            return []
        
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if A[mid] >= target:
                end = mid
            else:
                start = mid + 1
        # 注意：要用 A[start] >= target而不是 A[start] >= A[end]
        # A[start] > A[end]是不可能发生的！
        # 所以A[start] >= A[end]这个条件每次都选择了end！！！
        first_inx = start if A[start] >= target else end
        
        l, r = first_inx - 1, first_inx
        res = []
        for _ in range(k):
            if l < 0:
                res.append(A[r])
                r += 1
            elif r >= len(A):
                res.append(A[l])
                l -= 1
            else:
                if A[r] - target < target - A[l]:
                    res.append(A[r])
                    r += 1
                else:
                    res.append(A[l])
                    l -= 1
        
        return res
```

#### 447. 在大数组中查找
```
class Solution:
    """
    @param: reader: An instance of ArrayReader.
    @param: target: An integer
    @return: An integer which is the first index of target.
    """
    def searchBigSortedArray(self, reader, target):
        # write your code here
        # 先找到一个大于target的上界
        # 这里假设了target一定小于这个超大有序数组的上界
        inx_limit = 0
        # 指数退避exponential back-off 
        while reader.get(inx_limit) < target:
            # 注意inx_limit不能为2 * inx_limit
            # 因为初始inx_limit = 0了
            inx_limit = 2 * inx_limit + 1
        
        start, end = 0, inx_limit
        while start + 1 < end:
            mid = start + (end - start) // 2
            # 注意上面在找inx_limit的时候
            # 是可能出现inx_limit是越界的
            # 此时就要缩回去
            if reader.get(mid) == 2 ** 31:
                end = mid - 1
            # 题目要求的是第一次出现的位置
            # 核心必背！！！
            ##################################
            elif reader.get(mid) >= target:  #
                end = mid                    #
            else:                            #
                start = mid + 1              #
            ##################################

        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        return -1
```

#### 428. x的n次幂
```
class Solution:
    """
    @param x: the base number
    @param n: the power number
    @return: the result
    """
    def myPow(self, x, n):
        # write your code here
        if n < 0:
            return 1 / self.myPow(x, -n)
        
        if n == 0:
            return 1
        
        # 这道题是给定n一定是一个整数
        half = self.myPow(x, n // 2)
        if n % 2 == 0:
            return half * half
        return x * half * half
```

#### 159. 寻找旋转排序数组中的最小值
```
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        # rotated array系列有很多题
        # 一类是找最大/最小(比如这道)
        # 一类是找目标值
        if not nums:
            return 2 ** 31 - 1
        if len(nums) == 1:
            return nums[0]
        if nums[0] < nums[-1]:
            return nums[0]
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            if nums[mid] < nums[mid - 1]:
                return nums[mid]
            
            if nums[mid] > nums[0]:
                start = mid + 1
            else:
                end = mid - 1
        
        if nums[start] > nums[end]:
            return nums[end]
        return nums[start]
```

#### 140. 快速幂
```
class Solution:
    """
    @param a: A 32bit integer
    @param b: A 32bit integer
    @param n: A 32bit integer
    @return: An integer
    """
    def fastPower(self, a, b, n):
        # write your code here
        # 这道题根428. x的n次幂
        # 并没有本质的区别
        if n == 0:
            return 1 % b
        
        half = self.fastPower(a, b, n // 2)
        if n % 2 == 0:
            return (half * half) % b
        return (a * half * half) % b
```

#### 75. 寻找峰值
```
class Solution:
    """
    @param A: An integers array.
    @return: return any of peek positions.
    """
    def findPeak(self, A):
        # write your code here
        # 题目给定A的长度大于等于3
        # 九章官方答案有点啰嗦+confusing
        start, end = 1, len(A) - 2

        while start + 1 < end:
            mid = start + (end - start) // 2
            if A[mid] < A[mid - 1]:
                end = mid - 1
            elif A[mid] < A[mid + 1]:
                start = mid + 1
            else:
                return mid

        if A[start] > A[end]:
            return start
        return end
```

#### 74. 第一个错误的代码版本
```
class Solution:
    """
    @param n: An integer
    @return: An integer which is the first bad version.
    """
    def findFirstBadVersion(self, n):
        # write your code here
        if n == 1:
            return 1
        
        start, end = 1, n
        while start + 1 < end:
            mid = start + (end - start) // 2
            if SVNRepo.isBadVersion(mid):
                end = mid
            else:
                start = mid + 1
        
        if SVNRepo.isBadVersion(start):
            return start
        return end
```

#### 62. 搜索旋转排序数组
```
class Solution:
    """
    @param A: an integer rotated sorted array
    @param target: an integer to be searched
    @return: an integer
    """
    def search(self, A, target):
        # write your code here
        # 题目已经假设数组中不存在重复的元素
        if not A:
            return -1
        
        # 核心思想就是A[mid]和A[start] A[end]比较来确定是否落在连续的那一段
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if A[mid] == target:
                return mid
        
            if A[mid] < A[start]:
                if A[mid] < target and target <= A[end]:
                    start = mid + 1
                else:
                    end = mid - 1
            else:
                if A[mid] > target and target >= A[start]:
                    end = mid - 1
                else:
                    start = mid + 1

        if A[start] == target:
            return start
        if A[end] == target:
            return end
        return -1
```

#### 462. 目标出现总和
```
class Solution:
    """
    @param A: A an integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
    def totalOccurrence(self, A, target):
        # write your code here
        # 给一个升序的数组，以及一个target，找到它在数组中出现的次数。
        # 核心思想：二分找出target在A中第一次出现的位置和最后一次出现的位置
        if not A:
            return 0
        
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if A[mid] >= target:
                end = mid
            else:
                start = mid + 1
        left = start if A[start] == target else end
        # 注意：可能有corner case target不在A中存在
        if A[left] != target:
            return 0
        
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if A[mid] <= target:
                start = mid
            else:
                end = mid - 1
        right = end if A[end] == target else start
        
        return right - left + 1
```

#### 459. 排序数组中最接近元素
```
class Solution:
    """
    @param: A: an integer array sorted in ascending order
    @param: target: An integer
    @return: an integer
    """
    def closestNumber(self, nums, target):
        if not nums:
            return -1
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            # 这道题核心：因为要找最接近的数字
            # 所以不能轻易的+1或者-1
            # 这里要怂一点
            if nums[mid] > target:
                end = mid
            else:
                start = mid
        
        if abs(nums[start] - target) > abs(nums[end] - target):
            return end
        return start
```

#### 235. 分解质因数
```
class Solution:
    """
    @param num: An integer
    @return: an integer array
    """
    def primeFactorization(self, num):
        # write your code here
        # 基本思路就将num // 2然后遍历
        # 这道题保证了n >= 2
        res = []
        i = 2
        while i * i <= num:
            # 很有意思的思路：
            # 因为i是从小到大变化的
            # 所以比如i=2，就已经将num中所有2的n次方的全部除掉了
            # 换句话说当遍历i = 6的时候
            # 由于2和3在之前已经出现过了
            # 所以这里的num肯定不会再出现num能整除2或者整除3的情况了
            # 隐含着此时的i就一定是质数！！！！！
            # 这道题可以变形用来求小于某个上界的质数！！！
            while num % i == 0:
                num //= i
                res.append(i)
            i += 1
        if num != 1:
            res.append(num)
        return res
```

#### 254. 丢鸡蛋
```
class Solution:
    """
    @param: n: An integer
    @return: The sum of a and b
    """
    def dropEggs(self, n):
        # write your code here
        # 等间距扔鸡蛋的最坏情况：
        # 出现在最后一个区间的最后一个数
        # 可以每过一个区间把区间长度-1
        # 这样就能够达到最佳的方案
        # x + (x - 1) + (x - 2) + ... + 1 = n
        # 等差数列求和的问题
        x = 1
        while x * (x + 1) // 2 < n:
            x += 1
        return x
```

#### 28. Search a 2D Matrix
```
class Solution:
    """
    @param matrix: matrix, a list of lists of integers
    @param target: An integer
    @return: a boolean, indicate whether matrix contains target
    """
    def searchMatrix(self, matrix, target):
        # write your code here
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        row, col = m - 1, 0
        while row >= 0 and col < n:
            if matrix[row][col] == target:
                return True
            if matrix[row][col] > target:
                row -= 1
            else:
                col += 1
        
        return False
```

#### 14. 二分查找
```
class Solution:
    """
    @param nums: The integer array.
    @param target: Target to find.
    @return: The first position of target. Position starts from 0.
    """
    def binarySearch(self, nums, target):
        # write your code here
        if not nums:
            return -1
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] >= target:
                end = mid
            else:
                start = mid + 1
        
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1
```

#### 414. 两个整数相除
```
class Solution:
    """
    @param dividend: the dividend
    @param divisor: the divisor
    @return: the result
    """
    def divide(self, dividend, divisor):
        # write your code here
        if divisor == 0:
            raise ValueError('Divisor can not be 0!')
        
        if dividend == 0:
            return 0
        
        sign = 1
        if (dividend > 0 and divisor < 0 ) or (dividend < 0) and (divisor > 0):
            sign = -1
        
        dividend = abs(dividend)
        divisor = abs(divisor)
        res = 0
        while dividend >= divisor:
            shift = 0
            # 相当于将divisor除以2了
            while dividend >= (divisor << shift):
                shift += 1
            dividend -= divisor << (shift - 1)
            # 相当于2进制的加法
            res += 1 << (shift - 1)
        
        if sign * res > 2 ** 31 - 1:
            return 2 ** 31 - 1
        return sign * res
```

#### 61. 搜索区间
```
class Solution:
    """
    @param: nums: an integer sorted array
    @param: target: an integer to be inserted
    @return: a list of length 2, [index1, index2]
    """
    def searchRange(self, nums, target):
        # write your code here
        if not nums:
            return [-1, -1]
        n = len(nums)
        
        start, end = 0, n - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] >= target:
                end = mid
            else:
                start = mid + 1
        
        start_pos = -1
        if nums[end] == target:
            start_pos = end
        if nums[start] == target:
            start_pos = start
        
        start, end = 0, n - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] <= target:
                start = mid
            else:
                end = mid - 1
        
        end_pos = -1
        if nums[start] == target:
            end_pos = start
        if nums[end] == target:
            end_pos = end
        
        if start_pos != -1 and end_pos != -1:
            return [start_pos, end_pos]
        return [-1, -1]
```

#### 38. 搜索二维矩阵 II
```
class Solution:
    """
    @param matrix: A list of lists of integers
    @param target: An integer you want to search in matrix
    @return: An integer indicate the total occurrence of target in the given matrix
    """
    def searchMatrix(self, matrix, target):
        # write your code here
        # 跟28 I几乎一模一样
        # 只不过I是问存不存在
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        
        i, j = m - 1, 0
        res = 0
        while i >= 0 and j < n:
            if matrix[i][j] == target:
                res += 1
                i -= 1
                j += 1
            elif matrix[i][j] > target:
                i -= 1
            else:
                j += 1
        
        return res
```

#### 600. 包裹黑色像素点的最小矩形
```
class Solution:
    """
    @param image: a binary matrix with '0' and '1'
    @param x: the location of one of the black pixels
    @param y: the location of one of the black pixels
    @return: an integer
    """
    def minArea(self, image, x, y):
        # write your code here
        if not image or not image[0]:
            return 0
        
        m, n = len(image), len(image[0])
        
        start, end = 0, x
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_row(mid, image):
                end = mid
            else:
                start = mid
        top_level = start if self._check_row(start, image) else end
        
        start, end = x, m - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_row(mid, image):
                start = mid
            else:
                end = mid
        bottom_level = end if self._check_row(end, image) else start
        
        start, end = 0, y
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_col(mid, image):
                end = mid
            else:
                start = mid
        left_level = start if self._check_col(start, image) else end
        
        start, end = y, n - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_col(mid, image):
                start = mid
            else:
                end = mid
        right_level = end if self._check_col(end, image) else start
        
        return (right_level - left_level + 1) * (bottom_level - top_level + 1)
    
    def _check_row(self, row, image):
        if any(i == '1' for i in image[row]):
            return True
        return False
    
    def _check_col(self, col, image):
        if any(i[col] == '1' for i in image):
            return True
        return False
```

#### 457. 经典二分查找问题
```
class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """
    def findPosition(self, nums, target):
        # write your code here
        if not nums:
            return -1
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid - 1
            else:
                start = mid + 1
        
        if nums[start] == target:
            return start
        if nums[end] == target:
            return end
        return -1
```

#### 141. x的平方根
```
class Solution:
    """
    @param x: An integer
    @return: The sqrt of x
    """
    def sqrt(self, x):
        # write your code here
        if x <= 1:
            return x
        
        start, end = 1, x
        while start + 1 < end:
            mid = start + (end - start) // 2
            if mid * mid > x:
                end = mid - 1
            else:
                start = mid
        
        # 先找最大的
        if end * end < x:
            return end
        return start
```

#### 617. 最大平均值子数组 II
```
class Solution:
    """
    @param nums: an array with positive and negative numbers
    @param k: an integer
    @return: the maximum average
    """
    def maxAverage(self, nums, k):
        # write your code here
        # 基本思路：平均值一定小于等于数组中的最大值，并且大于等于数组中的最小值
        low, high = min(nums), max(nums)
        while high - low > 1e-7:
            mid = (low + high) / 2
            if self._check(nums, k, mid):
                low = mid
            else:
                high = mid
        
        # 此时return low或者high都是一样的
        return low
    
    # check函数作用是检查nums中是否存在长度大于k
    # 并且和大于mid的子数组
    # 为什么呢？因为如果我们已经算出来了最大的平均值是max_mean,
    # 则这个数组中任何的子数组中每个元素减去这个max_mean之后的sum和一定小于0
    # 只有满足max_mean条件的那个子数组的和这么操作后的sum和是等于0的
    # 这样我们就能找到折半方向了
    def _check(self, nums, k, mean):
        min_sum = 2 ** 31 - 1
        # 核心之一：
        # 注意这里的pre_sum和一般定义的前缀和数组不同
        # 是去mean之后的前缀和!!!!!
        pre_sum = [0] * (len(nums) + 1)
        
        for i in range(1, len(nums) + 1):
            pre_sum[i] = pre_sum[i - 1] + nums[i - 1] - mean
            if i >= k:
                min_sum = min(min_sum, pre_sum[i - k])
                if pre_sum[i] - min_sum > 0:
                    # 这里的min_sum一定前面遍历过的
                    # 长度大于k的子数组的pre_sum中的最小值
                    # 此时如果发现大于0
                    # 说明nums中是存在长度大于k并且均值大于mean的子数组的
                    # 可以说明此时的mean有点小了
                    # 在二分中可以通过增大low来增大mean
                    return True
        
        return False
```

#### 586. 对x开根II
```
class Solution:
    """
    @param: x: a double
    @return: the square root of x
    """
    def sqrt(self, x):
        # write your code here
        # 注意边界情况：当x是小于1的数字的时候，平方根就是1
        # 其他的和开平方I那道题一模一样
        low, high = 0, x if x > 1 else 1
        while high - low > 1e-12:
            mid = (low + high) / 2
            if mid * mid > x:
                high = mid
            else:
                low = mid
        return low
```

#### 160. 寻找旋转排序数组中的最小值 II
```
class Solution:
    """
    @param nums: a rotated sorted array
    @return: the minimum number in the array
    """
    def findMin(self, nums):
        # write your code here
        if not nums:
            return -1
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            # 这种旋转数组类型的题
            # 优先考虑和num[end]比较
            if nums[mid] > nums[end]:
                start = mid + 1
            elif nums[mid] < nums[end]:
                end = mid
            else:
                end -= 1
        
        if nums[start] < nums[end]:
            return nums[start]
        return nums[end]
```

#### 63. 搜索旋转排序数组 II
```
class Solution:
    """
    @param A: an integer ratated sorted array and duplicates are allowed
    @param target: An integer
    @return: a boolean 
    """
    def search(self, A, target):
        # write your code here
        if not A:
            return False
        
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if A[mid] == target:
                return True
            
            if A[mid] > A[start]:
                if A[mid] > target >= A[start]:
                    end = mid - 1
                else:
                    start = mid + 1
            elif A[mid] < A[start]:
                if A[mid] < target <= A[end]:
                    start = mid + 1
                else:
                    end = mid - 1
            else:
                start += 1
        
        if A[start] == target or A[end] == target:
            return True
        return False
```

#### 437. 书籍复印
```
class Solution:
    """
    @param pages: an array of integers
    @param k: An integer
    @return: an integer
    """
    def copyBooks(self, pages, k):
        # write your code here
        # 这道题是很精彩的二分题目，需要多多的看和理解！！！
        if not pages:
            return 0
        
        # 最短能有多短的时间能最快的复印完？
        # 肯定是有多少本书就有多少个人来复印
        # 这样每个人并行的复印一本书
        # 最短的时间肯定是由最多的页数决定的
        min_time = max(pages)
        
        # 耗时最多的方式就是一个人复印完所有的书
        # 所以最多的时间就是所有的书的页数和
        max_time = sum(pages)
        
        # 核心之一：这样我们就找到了二分的上下界，可以进行二分了
        start, end = min_time, max_time
        while start + 1 < end:
            given_time = start + (end - start) // 2
            # 说明在given这么多时间的情况下
            # 完成复印的工作
            # 需要的人数大于k
            # 所以需要给更多的时间
            # 此时start就应该增加了
            if self._num_copiers_required(pages, given_time) > k:
                start = given_time + 1
            else:
                end = given_time
        
        if self._num_copiers_required(pages, start) <= k:
            return start
        return end
    
    def _num_copiers_required(self, pages, given_time):
        curr_works = pages[0]
        cnt = 1
        for i in range(1, len(pages)):
            # 首先我们确定传进来的given_time一定大于等于pages里的任何一本书
            # 因为我们求得min_time就是pages里的最大值
            # 在这步里
            # 如果新加的工作超过了给定的时间
            # 就说明要加人了
            # 而且新加的这个人最坏情况下独立完成这本书的复印
            # 一定不会出现新加人也完不成的情况
            # 所以可以将curr_works先清零了
            # 最后在if外面再加回来
            if curr_works + pages[i] > given_time:
                cnt += 1
                curr_works = 0
            curr_works += pages[i]
        return cnt
```

#### 183. 木材加工
```
class Solution:
    """
    @param L: Given n pieces of wood with length L[i]
    @param k: An integer
    @return: The maximum length of the small pieces
    """
    def woodCut(self, L, k):
        # write your code here
        # 说明此时我们就算1厘米1厘米的切，也凑不成k块
        if sum(L) < k:
            return 0
        # 木头最小能切到1
        # 最大能就是这堆木材中的最长木头长度 
        start, end = 1, max(L)
        while start + 1 < end:
            mid = start + (end - start) // 2
            # 看看当前切成mid长度的小块
            # 能不能切出k块来
            # 如果能切出k块来
            # 说明此时的mid长度可以再大一点（毕竟题目要求最长的长度）
            # 应该通过增大start来增大mid
            
            # 重点：这道题实际上类似有序数组中寻找最后一个位置的目标元素！！！
            if self._check_chunks_count(L, mid) >= k:
                start = mid
            else:
                end = mid - 1
        
        if self._check_chunks_count(L, end) >= k:
            return end
        return start
    
    def _check_chunks_count(self, L, length):
        count = 0
        for each in L:
            count += each // length
        return count
```

### 3 - Two Pointers Algorithm

#### 228. 链表的中点
```
class Solution:
    """
    @param head: the head of linked list.
    @return: a middle node of the linked list
    """
    def middleNode(self, head):
        # write your code here
        if not head or not head.next:
            return head
        
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
```

#### 607. 两数之和 III-数据结构设计
```
class TwoSum:
    """
    @param: number: An integer
    @return: nothing
    """
    
    def __init__(self):
        self._hash = dict()
    
    def add(self, number):
        # write your code here
        if number in self._hash:
            self._hash[number] += 1
        else:
            self._hash[number] = 1

    def find(self, value):
        # write your code here
        for each in self._hash:
            diff = value - each
            if diff in self._hash:
                if diff != each:
                    return True
                else:
                    if self._hash[each] > 1:
                        return True
        return False
```

#### 539. 移动零
```
class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """
    def moveZeroes(self, nums):
        # write your code here
        zero_inx = -1
        for i in range(len(nums)):
            if nums[i] != 0:
                zero_inx += 1
                nums[i], nums[zero_inx] = nums[zero_inx], nums[i]
```

#### 521. 去除重复元素
```
class Solution:
    """
    @param nums: an array of integers
    @return: the number of unique integers
    """
    def deduplication(self, nums):
        # write your code here
        if not nums:
            return 0
        
        nums.sort()
        # end含义是数组中不重复的子数组的最后一个位置
        # 初始设置为开头元素的坐标即可
        end = 0
        
        for i in range(1, len(nums)):
            if nums[i] != nums[end]:
                end += 1
                nums[end] = nums[i]
        
        return end + 1
```

#### 464. 整数排序 II
```
class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers2(self, A):
        # write your code here
        self._quick_sort(A, 0, len(A) - 1)
        return A
    
    def _quick_sort(self, A, l, r):
        if l >= r:
            return
        p = self._partition(A, l, r)
        self._quick_sort(A, l, p - 1)
        self._quick_sort(A, p + 1, r)
    
    def _partition(self, A, l, r):
        pivot_value = A[l]
        j = l
        for i in range(l + 1, r + 1):
            if A[i] < pivot_value:
                j += 1
                A[i], A[j] = A[j], A[i]
        A[l], A[j] = A[j], A[l]
        return j
```

#### 608. 两数和 II-输入已排序的数组
```
class Solution:
    """
    @param: nums: an array of Integer
    @param: target: target = nums[index1] + nums[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, nums, target):
        # write your code here
        
        if not nums:
            return [-1, -1]
        
        left, right = 0, len(nums) - 1
        
        while left < right:
            total = nums[left] + nums[right]
            if total == target:
                return [left + 1, right + 1]
            if total < target:
                left += 1
            else:
                right -= 1
        
        return [-1, -1]
```

#### 143. 排颜色 II
```
class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sortColors2(self, colors, k):
        # write your code here
        # l和r的定义为colors中需要排序的起点和终点
        # 换句说话l和r的位置上的元素是没有排过序的
        l, r = 0, len(colors) - 1
        # num0和num2的定义为当前需要排序的颜色
        num0, num2 = 1, k

        while num0 < num2:
            # 每次大循环就是将num0和num2排好序
            i = l
            while i <= r:
                if colors[i] == num0:
                    colors[i], colors[l] = colors[l], colors[i]
                    l += 1
                    i += 1
                elif colors[i] == num2:
                    colors[i], colors[r] = colors[r], colors[i]
                    r -= 1
                    # 由于此时并不知道当前换过来以后的colors[i]到底会不会
                    # 此时的r是从未来换过来的所以并不确定是什么
                    # 还是num0, 所以i还不能+1
                else:
                    i += 1
            num0 += 1
            num2 -= 1
```

#### 57. 三数之和
```
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, numbers):
        # write your code here
        if not numbers:
            return []
        
        n = len(numbers)
        numbers.sort()
        res = []
        for i in range(n - 2):
            # 这一步是为了去重
            if i > 0 and numbers[i] == numbers[i - 1]:
                continue
            curr = numbers[i]
            l, r = i + 1, n - 1
            while l < r:
                total = curr + numbers[l] + numbers[r]
                if total == 0:
                    res.append([numbers[i], numbers[l], numbers[r]])
                    # l和r要先加减1
                    # 否则在while的条件里会越界！！！
                    l += 1
                    r -= 1
                    while l < r and numbers[l] == numbers[l - 1]:
                        l += 1
                    while l < r and numbers[r] == numbers[r + 1]:
                        r -= 1
                elif total > 0:
                    r -= 1
                else:
                    l += 1
        
        return res
```

#### 31. 数组划分
```
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partitionArray(self, nums, k):
        # write your code here
        if not nums:
            return 0
        
        l, r = 0, len(nums) - 1
        while l < r:
            # 将满足条件的l和r先去掉
            # 最终停下来的l和r一定是不满足条件而需要交换的
            # 这里为什么要取等号？？？要再想想
            while l <= r and nums[l] < k:
                l += 1
            while l <= r and nums[r] >= k:
                r -= 1
            if l < r:
                nums[l], nums[r] = nums[r], nums[l]
                l += 1
                r -= 1
        
        # 记住：l和r的位置一定是不满足条件的位置
        # 所以此时应该返回l
        return l
```

#### 5. 第k大元素
```
class Solution:
    """
    @param k: An integer
    @param nums: An array
    @return: the Kth largest element
    """
    def kthLargestElement(self, k, nums):
        # write your code here
        n = len(nums)
        l, r = 0, n - 1
        while True:
            p = self._partition(nums, l, r)
            if p == k - 1:
                return nums[p]
            elif p > k - 1:
                nums[l]
                r = p - 1
            else:
                l = p + 1
    
    def _partition(self, nums, l, r):
        pivot_value = nums[l]
        # j的定义是左边最后一个排过的数字的下标！！！！
        # 即j + 1就是下一个可以用来交换新数字的位置
        j = l
        for i in range(l + 1, r + 1):
            if nums[i] > pivot_value:
                j += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[l], nums[j] = nums[j], nums[l]
        return j
```

#### 604. 滑动窗口内数的和
```
class Solution:
    """
    @param nums: a list of integers.
    @param k: length of window.
    @return: the sum of the element inside the window at each moving.
    """
    def winSum(self, nums, k):
        # write your code here
        if not nums:
            return []
        if k == 0:
            return [0]
        
        # pre_sum数组：
        # 如果要求原数组中[l, r]左闭右闭的区间的和
        # 在pre_sum数组中就是pre_sum[r + 1] - pre_sum[l]
        pre_sum = [0]
        for i in range(len(nums)):
            pre_sum.append(nums[i] + pre_sum[-1])
        
        res = []
        for i in range(len(nums) - k + 1):
            res.append(pre_sum[i + k] - pre_sum[i])

        return res
```

#### 56. 两数之和
```
class Solution:
    def twoSum(self, numbers, target):
        # write your code here
        mapping = {}
        for i in range(len(numbers)):
            if target - numbers[i] in mapping:
                return [mapping[target - numbers[i]], i]
            mapping[numbers[i]] = i

        return []
```

#### 609. 两数和-小于或等于目标值
```
class Solution:
    """
    @param nums: an array of integer
    @param target: an integer
    @return: an integer
    """
    def twoSum5(self, nums, target):
        # write your code here
        # 注意审题！！！
        # 这道题问的是有多少对（2个元素）的和为target
        # 不是子数组的和
        if not nums:
            return 0
        
        nums.sort()
        l, r = 0, len(nums) - 1
        
        res = 0
        while l < r:
            curr = nums[l] + nums[r]
            if curr > target:
                r -= 1
            else:
                # 怎么理解？
                # 由于nums是sort过的
                # 所以这里实际上是说以r结尾的对儿有多少个
                # 所以是r - l个
                res += r - l
                l += 1
        
        return res
```

#### 587. 两数之和 - 不同组成
```
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """
    def twoSum6(self, nums, target):
        # write your code here
        # 给一整数数组, 找到数组中有多少组
        # 不同的元素对儿，有相同的和, 且和为给出的target值, 返回对数.
        if not nums:
            return 0
        
        nums.sort()
        l, r = 0, len(nums) - 1
        
        res = 0
        while l < r:
            curr = nums[l] + nums[r]
            if curr == target:
                res += 1
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
                while l < r and nums[r] == nums[r + 1]:
                    r -= 1
            elif curr > target:
                r -= 1
            else:
                l += 1
        
        return res
```

#### 533. 两数和的最接近值
```
class Solution:
    """
    @param nums: an integer array
    @param target: An integer
    @return: the difference between the sum and the target
    """
    def twoSumClosest(self, nums, target):
        # write your code here
        # 找到两个数字使得他们和最接近target
        if not nums:
            return 2 ** 31 - 1
        
        nums.sort()
        l, r = 0, len(nums) - 1
        res = 2 ** 31 - 1
        
        while l < r:
            curr = nums[l] + nums[r]
            if curr - target == 0:
                return 0
            elif curr > target:
                res = min(res, curr - target)
                r -= 1
            else:
                # 注意这里由于此时curr < target
                # 所以应该是target - curr
                res = min(res, target - curr)
                l += 1
        
        return res
```

#### 443. 两数之和 II
```
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: an integer
    """
    def twoSum2(self, nums, target):
        # write your code here
        # 给一组整数，问能找出多少对整数
        # 他们的和大于一个给定的目标值。
        if not nums:
            return 0
        
        nums.sort()
        l, r = 0, len(nums) - 1

        res = 0
        while l < r:
            curr = nums[l] + nums[r]
            if curr > target:
                res += r - l
                r -= 1
            else:
                l += 1
        
        return res
```

#### 461. 无序数组K小元素
```
class Solution:
    """
    @param k: An integer
    @param nums: An integer array
    @return: kth smallest element
    """
    def kthSmallest(self, k, nums):
        # write your code here
        if not nums:
            return 2 ** 31 - 1
        if k > len(nums):
            return -2**31 - 1
        
        l, r = 0, len(nums) - 1
        while True:
            p = self._partition(nums, l, r)
            if p == k - 1:
                return nums[p]
            # 说明此时找到的是更大的数字
            # 需要缩小r
            elif p > k - 1:
                r = p - 1
            # 说明此时找到的是更小的狮子（p < k - 1）
            # 需要增大l
            else:
                l = p + 1
        
        return 2 ** 31 - 1
    
    def _partition(self, nums, l, r):
        pivot_value = nums[l]
        j = l
        for i in range(l + 1, r + 1):
            if nums[i] < pivot_value:
                j += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[j], nums[l] = nums[l], nums[j]
        return j
```

#### 382. 三角形计数
```
class Solution:
    """
    @param S: A list of integers
    @return: An integer
    """
    def triangleCount(self, S):
        # write your code here
        # 给定一个整数数组，在该数组中，寻找三个数
        # 分别代表三角形三条边的长度
        # 问可以寻找到多少组这样的三个数来组成三角形？
        if len(S) < 3:
            return 0
        
        S.sort()
        res = 0
        
        for i in range(len(S) - 1, -1, -1):
            l, r = 0, i - 1
            while l < r:
                if S[l] + S[r] > S[i]:
                    res += r - l
                    r -= 1
                else:
                    l += 1
            
        return res
```

#### 148. 颜色分类
```
class Solution:
    """
    @param nums: A list of integer which is 0, 1 or 2 
    @return: nothing
    """
    def sortColors(self, nums):
        # write your code here
        # l和r的定义是nums中需要排序的起点和终点坐标
        l, r = 0, len(nums) - 1
        i = 0
        while i <= r:
            if nums[i] == 0:
                nums[i], nums[l] = nums[l], nums[i]
                l += 1
                i += 1
            elif nums[i] == 1:
                i += 1
            else:
                nums[i], nums[r] = nums[r], nums[i]
                r -= 1
```

#### 59. 最接近的三数之和
```
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @param target: An integer
    @return: return the sum of the three integers, the sum closest target.
    """
    def threeSumClosest(self, numbers, target):
        # write your code here
        if not numbers:
            return 2 ** 31 - 1
        
        n = len(numbers)
        numbers.sort()
        res = 2 ** 31 - 1
        
        for i in range(n - 2):
            l, r = i + 1, n - 1
            while l < r:
                total = numbers[i] + numbers[l] + numbers[r]
                if abs(total - target) < abs(res - target):
                    res = total
                
                if total >= target:
                    r -= 1
                else:
                    l += 1
        
        return res
```

#### 894. 烙饼排序
```
class Solution:
    """
    @param array: an integer array
    @return: nothing
    """
    def pancakeSort(self, array):
        # Write your code here
        for i in range(len(array) - 1, 0, -1):
            for j in range(i, 0, -1):
                if array[j] > array[0]:
                    FlipTool.flip(array, j)
            FlipTool.flip(array, i)
```

#### 625. 数组划分II
```
class Solution:
    """
    @param nums: an integer array
    @param low: An integer
    @param high: An integer
    @return: nothing
    """
    def partition2(self, nums, low, high):
        # write your code here
        # 基本思路跟sort color I一模一样
        if not nums:
            return
        
        l, r = 0, len(nums) - 1
        i = 0
        while i <= r:
            if nums[i] < low:
                nums[i], nums[l] = nums[l], nums[i]
                l += 1
                i += 1
            elif nums[i] > high:
                nums[i], nums[r] = nums[r], nums[i]
                r -= 1
            else:
                i += 1
```

#### 610. 两数和 - 差等于目标值
```
class Solution:
    """
    @param nums: an array of Integer
    @param target: an integer
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum7(self, nums, target):
        # write your code here
        # 给定一个整数数组，找到两个数的差等于目标值
        # index1必须小于index2
        # 注意返回的index1和index2不是 0-based。
        if not nums:
            return [-1, -1]
        
        n = len(nums)
        num_with_index = []
        # 这道题好坑！！！
        # 因为我们解法的思路就是两根指针求一个差值的正值
        # 所以这里的target要取正值！！！
        target = abs(target)
        for inx, val in enumerate(nums):
            num_with_index.append((val, inx))
        num_with_index.sort()

        r = 0
        for l in range(n):
            if r == l:
                r += 1
            while r < n and num_with_index[r][0] - num_with_index[l][0] < target:
                r += 1
            if r < n and num_with_index[r][0] - num_with_index[l][0] == target:
                if num_with_index[l][1] < num_with_index[r][1]:
                    return [num_with_index[l][1] + 1, num_with_index[r][1] + 1]
                else:
                    return [num_with_index[r][1] + 1, num_with_index[l][1] + 1]
        
        return [-1, -1]
        
        
        # O(n)的解法（需要O(n)空间）
        # 跟two sum一模一样
        # if not nums:
        #     return [-1, -1]
        
        # mapping = {}
        # for i in range(len(nums)):
        #     if nums[i] + target in mapping:
        #         return [mapping[nums[i] + target] + 1, i + 1]
        #     if nums[i] - target in mapping:
        #         return [mapping[nums[i] - target] + 1, i + 1]
        #     mapping[nums[i]] = i
        
        # return [-1, -1]
```

#### 380. 两个链表的交叉
```
"""
Definition of ListNode
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    """
    @param head1: the first list
    @param head2: the second list
    @return: a ListNode
    """
    def getIntersectionNode(self, head1, head2):
        # write your code here
        if not head1 or not head2:
            return None
        
        len1 = 0
        curr = head1
        while curr:
            len1 += 1
            curr = curr.next
        
        len2 = 0
        curr = head2
        while curr:
            len2 += 1
            curr = curr.next
        
        curr1, curr2 = head1, head2
        if len1 > len2:
            diff = len1 - len2
            while diff:
                head1 = head1.next
                diff -= 1
        elif len1 < len2:
            diff = len2 - len1
            while diff:
                head2 = head2.next
                diff -= 1
        
        while head1:
            if head1 is head2:
                return head1
            head1 = head1.next
            head2 = head2.next
        
        return None
        
        
        # LC高票
        # if not head1 or not head2:
        #     return None
        
        # curr1, curr2 = head1, head2
        # while curr1 is not curr2:
        #     if curr1 is not None:
        #         curr1 = curr1.next
        #     else:
        #         curr1 = curr2
        #     if curr2 is not None:
        #         curr2 = curr.next
        #     else:
        #         curr2 = curr1
```

#### 102. 带环链表
```
class Solution:
    """
    @param head: The first node of linked list.
    @return: True if it has a cycle, or false
    """
    # 返回是否有环
    def hasCycle(self, head):
        # write your code here
        if not head:
            return False
        
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        
        return False
```

#### 58. 四数之和
```
class Solution:
    """
    @param numbers: Give an array
    @param target: An integer
    @return: Find all unique quadruplets in the array which gives the sum of zero
    """
    def fourSum(self, numbers, target):
        # write your code here
        if not numbers:
            return []
        
        numbers.sort()
        n = len(numbers)
        res = []
        
        for i in range(n - 3):
            if i != 0 and numbers[i] == numbers[i - 1]:
                continue
            for j in range(i + 1, n - 2):
                if j != i + 1 and numbers[j] == numbers[j - 1]:
                    continue
                l, r = j + 1, n - 1
                curr_target = target - numbers[i] - numbers[j]
                while l < r:
                    temp = numbers[l] + numbers[r]
                    if temp == curr_target:
                        res.append([numbers[i], numbers[j], numbers[l], numbers[r]])
                        l += 1
                        r -= 1
                        while l < r and numbers[l] == numbers[l - 1]:
                            l += 1
                        while l < r and numbers[r] == numbers[r + 1]:
                           r -= 1
                    elif temp > curr_target:
                        r -= 1
                    else:
                        l += 1
        
        return res
```

#### 103. 带环链表 II
```
class Solution:
    """
    @param head: The first node of linked list.
    @return: The node where the cycle begins. if there is no cycle, return null
    """
    def detectCycle(self, head):
        # write your code here
        if not head or not head.next:
            return None
        
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                break
        
        if slow is fast:
            # 非常非常巧妙的思路！
            # 在找环的时候
            # 设置another_slow一步一步走
            # 则因为有环
            # 这两个node早晚会相遇！！！
            another_slow = head
            while another_slow is not slow:
                slow = slow.next
                another_slow = another_slow.next
            return another_slow
        
        return None
```

### 4 - BFS & Topological Sort

#### 433. 岛屿的个数
```
from collections import deque

class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """
    def numIslands(self, grid):
        # write your code here
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        DIR = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and not visited[i][j]:
                    res += 1
                    queue = deque()
                    queue.append((i, j))
                    visited[i][j] = True
                    while queue:
                        ci, cj = queue.popleft()
                        for di, dj in DIR:
                            newi, newj = ci + di, cj + dj
                            if not 0 <= newi < m or \
                                not 0 <= newj < n or \
                                grid[newi][newj] == 0 or \
                                visited[newi][newj]:
                                continue
                            queue.append((newi, newj))
                            visited[newi][newj] = True
        
        return res
```

#### 69. 二叉树的层次遍历
```
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
from collections import deque

class Solution:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        # write your code here
        if not root:
            return []

        res = []
        queue = deque()
        queue.append(root)
        
        while queue:
            q_len = len(queue)
            curr = []
            for _ in range(q_len):
                temp_node = queue.popleft()
                curr.append(temp_node.val)
                if temp_node.left:
                    queue.append(temp_node.left)
                if temp_node.right:
                    queue.append(temp_node.right)
            res.append(curr)
        
        return res
```

#### 615. 课程表
```
from collections import defaultdict
from collections import deque

class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: true if can finish all courses or false
    """
    def canFinish(self, numCourses, prerequisites):
        # write your code here
        indegrees = [0] * numCourses
        edges = defaultdict(list)
        for course, pre_course in prerequisites:
            indegrees[course] += 1
            edges[pre_course].append(course)
        
        queue = deque()
        for inx, degree in enumerate(indegrees):
            if degree == 0:
                queue.append(inx)
        
        res = []
        while queue:
            curr = queue.popleft()
            res.append(curr)
            for each_course in edges[curr]:
                indegrees[each_course] -= 1
                if indegrees[each_course] == 0:
                    queue.append(each_course)
        
        return len(res) == numCourses
```

#### 616. 安排课程
```
from collections import defaultdict
from collections import deque

class Solution:
    """
    @param: numCourses: a total of n courses
    @param: prerequisites: a list of prerequisite pairs
    @return: the course order
    """
    def findOrder(self, numCourses, prerequisites):
        # write your code here
        indegrees = [0] * numCourses
        edges = defaultdict(list)
        for to_, from_ in prerequisites:
            indegrees[to_] += 1
            edges[from_].append(to_)
        
        queue = deque()
        for inx, degree in enumerate(indegrees):
            if degree == 0:
                queue.append(inx)
    
        res = []
        while queue:
            curr = queue.popleft()
            res.append(curr)
            for each_course in edges[curr]:
                indegrees[each_course] -= 1
                if indegrees[each_course] == 0:
                    queue.append(each_course)
        
        return res if len(res) == numCourses else []
```

#### 611. 骑士的最短路线
```
from collections import deque

class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path 
    """
    # 这道题问的是从起点到终点的最短路线
    # 注意这里的棋盘是有障碍物的
    _DIRS = [
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
    ]
    
    def shortestPath(self, grid, source, destination):
        # write your code here
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        
        queue = deque()
        queue.append((source.x, source.y))
        visited[source.x][source.y] = True
        
        res = 0
        while queue:
            q_len = len(queue)
            for _ in range(q_len):
                ci, cj = queue.popleft()
                if ci == destination.x and cj == destination.y:
                    return res
                for di, dj in self._DIRS:
                    newi, newj = ci + di, cj + dj
                    if not 0 <= newi < m or \
                        not 0 <= newj < n or \
                        grid[newi][newj] == 1 or \
                        visited[newi][newj]:
                        continue
                    queue.append((newi, newj))
                    visited[newi][newj] = True
            res += 1
        
        return -1
```

#### 605. 序列重构
```
from collections import defaultdict
from collections import deque

class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        # write your code here
        indegrees = defaultdict(int)
        edges = defaultdict(list)
        
        for each_seq in seqs:
            # 这里是为了避开seqs=[[]]的情况
            if not each_seq:
                continue
            # 这里是为了避开seqs=[[1]]的情况
            indegrees[each_seq[0]] += 0
            for i in range(len(each_seq) - 1):
                from_, to_ = each_seq[i], each_seq[i + 1]
                indegrees[to_] += 1
                edges[from_].append((from_, to_))
        
        queue = deque()
        for each, degree in indegrees.items():
            if degree == 0:
                queue.append(each)
        if len(queue) > 1:
            return False

        res = []
        while queue:
            curr = queue.popleft()
            res.append(curr)
            temp = []
            for _, to_ in edges[curr]:
                indegrees[to_] -= 1
                if indegrees[to_] == 0:
                    temp.append(to_)
            # 核心之一：这里每次temp只能size为0或者1
            # 因为如果大于1了说明有多条路径可以走
            # 题目要求是问是否有唯一的路径
            # 所以直接返回False即可
            if len(temp) > 1:
                return False
            queue += temp
        
        # 最终判断拓扑排序之后的结果是否是要求的org结果
        return res == org
```

#### 137. 克隆图
```
class Solution:
    """
    @param: node: A undirected graph node
    @return: A undirected graph node
    """
    def cloneGraph(self, node):
        # write your code here
        return self._bfs(node, dict())
    
    def _bfs(self, node, node_map):
        if not node:
            return
        if node.label in node_map:
            return node_map[node.label]
        
        new_node = UndirectedGraphNode(node.label)
        node_map[node.label] = new_node
        for each_neighbor in node.neighbors:
            new_node.neighbors.append(self._bfs(each_neighbor, node_map))
        
        return new_node
```

#### 127. 拓扑排序
```
"""
Definition for a Directed graph node
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""
from collections import defaultdict
from collections import deque

class Solution:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        # write your code here
        indegrees = defaultdict(int)
        # 这道题tricky之一：需要一个mapping关系，这道题每个节点的label是不同的
        # 所以可以用label做mapping的key
        node_mapping = {}
        edges = defaultdict(list)
        
        for each_node in graph:
            node_mapping[each_node.label] = each_node
            indegrees[each_node.label] += 0
            for each_neighbor in each_node.neighbors:
                indegrees[each_neighbor.label] += 1
                edges[each_node.label].append(each_neighbor)
                node_mapping[each_neighbor.label] = each_neighbor
        
        queue = deque()
        for label, degree in indegrees.items():
            if degree == 0:
                queue.append(node_mapping[label])
        
        res = []
        while queue:
            curr = queue.popleft()
            res.append(curr)
            for edge in edges[curr.label]:
                indegrees[edge.label] -= 1
                if indegrees[edge.label] == 0:
                    queue.append(node_mapping[edge.label])
        
        return res
```

#### 7. 二叉树的序列化和反序列化
```
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
from collections import deque

class Solution:
    """
    @param root: An object of TreeNode, denote the root of the binary tree.
    This method will be invoked first, you should design your own algorithm 
    to serialize a binary tree which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    """
    def serialize(self, root):
        # write your code here
        if not root:
            return ''
        
        queue = deque()
        queue.append(root)
        
        res = []
        while queue:
            curr = queue.popleft()
            if curr:
                res.append(str(curr.val))
                queue.append(curr.left)
                queue.append(curr.right)
            else:
                res.append('#')
        
        return ','.join(res)

    """
    @param data: A string serialized by your serialize method.
    This method will be invoked second, the argument data is what exactly
    you serialized at method "serialize", that means the data is not given by
    system, it's given by your own serialize method. So the format of data is
    designed by yourself, and deserialize it here as you serialize it in 
    "serialize" method.
    """
    def deserialize(self, data):
        # write your code here
        if not data:
            return
        
        nodes = data.split(',')
        head = TreeNode(int(nodes[0]))
        queue = deque()
        queue.append(head)
        
        is_left = True
        for val in nodes[1:]:
            if val != '#':
                new_node = TreeNode(int(val))
                if is_left:
                    queue[0].left = new_node
                else:
                    queue[0].right = new_node
                queue.append(new_node)
            # 说明这一层我们现在处理完了
            # 可以将队头pop出来处理下一层了
            if not is_left:
                queue.popleft()
            is_left = not is_left
        
        return head
```

#### 120. 单词接龙
```
from collections import deque

class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: word_set: a set of string
    @return: An integer
    """
    def ladderLength(self, start, end, word_set):
        # write your code here
        word_set.add(end)
        n = len(start)
        
        queue = deque()
        queue.append((start, 1))
        
        while queue:
            curr, steps = queue.popleft()
            if curr == end:
                return steps
            for i in range(n):
                for ch in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = curr[:i] + ch  + curr[i + 1:]
                    if new_word in word_set:
                        # 这里的word_set实际上也可以表示还可以用的词
                        # 所以每次用掉一个词就可以remove掉了
                        queue.append((new_word, steps + 1))
                        word_set.remove(new_word)
        
        return 0
```

#### 242. 将二叉树按照层级转化为链表
```
from collections import deque

class Solution:
    # @param {TreeNode} root the root of binary tree
    # @return {ListNode[]} a lists of linked list
    def binaryTreeToLists(self, root):
        # Write your code here
        if not root:
            return []

        queue = deque()
        queue.append(root)
        
        res = []
        while queue:
            q_len = len(queue)
            dummy_node = ListNode(-1)
            curr = dummy_node
            for _ in range(q_len):
                node = queue.popleft()
                curr.next = ListNode(node.val)
                curr = curr.next
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(dummy_node.next)

        return res
```

#### 624. 移除子串
```
from collections import deque

class Solution:
    """
    @param: s: a string
    @param: word_sets: a set of n substrings
    @return: the minimum length
    """
    def minLength(self, s, word_sets):
        # write your code here
        # 这道题问的是如果不停的用word_sets中的单词去替换s中的子串
        # 最终到不能再替换位置
        # 最短的长度是多少
        
        min_length = len(s)
        queue = deque()
        queue.append(s)
        replaced_word_sets = set()
        
        while queue:
            curr_word = queue.popleft()
            for word in word_sets:
                index = curr_word.find(word)
                while index != -1:
                    replaced_word = curr_word[:index] + curr_word[index + len(word):]
                    # 说明本次替换我们之前没有找到过
                    # 可以放入队列中了
                    # 并且因为此时是一个新的替换
                    # 我们可以更新下全局的min_length
                    if replaced_word not in replaced_word_sets:
                        queue.append(replaced_word)
                        replaced_word_sets.add(replaced_word)
                        min_length = min(min_length, len(replaced_word))
                    index = curr_word.find(word, index + 1)
        
        return min_length
```

#### 618. 搜索图中节点
```
from collections import deque

class Solution:
    """
    @param: graph: a list of Undirected graph node
    @param: values: a hash mapping, <UndirectedGraphNode, (int)value>
    @param: node: an Undirected graph node
    @param: target: An integer
    @return: a node
    """
    def searchNode(self, graph, values, node, target):
        # write your code here
        # 给定一张无向图
        # 一个节点以及一个目标值
        # 返回距离这个节点最近且值为目标值的节点
        # 如果不能找到则返回 NULL
        if not graph or not values:
            return None
        
        queue = deque()
        queue.append(node)
        
        while queue:
            curr = queue.popleft()
            if values[curr] == target:
                return curr
            for each_neighbor in curr.neighbors:
                queue.append(each_neighbor)
        
        return None
```

#### 598. 僵尸矩阵
```
from collections import deque

class Solution:
    """
    @param grid: a 2D integer grid
    @return: an integer
    """
    def zombie(self, grid):
        # write your code here
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        queue = deque()
        num_of_ppl = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    queue.append((i, j))
                if grid[i][j] == 0:
                    num_of_ppl += 1
        
        days = -1
        while queue:
            q_len = len(queue)
            for _ in range(q_len):
                ci, cj = queue.popleft()
                for di, dj in dirs:
                    newi, newj = ci + di, cj + dj
                    if not 0 <= newi < m or \
                        not 0 <= newj < n or \
                        grid[newi][newj] != 0:
                        continue
                    grid[newi][newj] = 1
                    # 最终要检查一下是否全部的人都已经变成了僵尸
                    num_of_ppl -= 1
                    queue.append((newi, newj))
            days += 1
        
        return days if num_of_ppl == 0 else -1
```

#### 531. 六度问题
```
"""
Definition for Undirected graph node
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""
from collections import deque

class Solution:
    """
    @param: graph: a list of Undirected graph node
    @param: s: Undirected graph node
    @param: t: Undirected graph nodes
    @return: an integer
    """
    def sixDegrees(self, graph, s, t):
        # write your code here
        queue = deque()
        queue.append((s, 0))
        visited = set()
        while queue:
            curr, step = queue.popleft()
            if curr.label == t.label:
                return step
            for each_neighbor in curr.neighbors:
                if each_neighbor.label not in visited:
                    queue.append((each_neighbor, step + 1))
                    visited.add(each_neighbor.label)
        
        return -1
```

#### 178. 图是否是树
```
from collections import deque

class Solution:
    """
    @param n: An integer
    @param edges: a list of undirected edges
    @return: true if it's a valid tree, or false
    """
    def validTree(self, n, edges):
        # write your code here
        if not edges:
            return n <= 1
        
        graph = [set() for _ in range(n)]
        
        for point1, point2 in edges:
            graph[point1].add(point2)
            graph[point2].add(point1)
        
        queue = deque()
        visited = set()
        # 初始随便一个就行
        queue.append(edges[0][0])
        visited.add(edges[0][0])
        
        while queue:
            curr_point = queue.popleft()
            for each in graph[curr_point]:
                if each in visited:
                    return False
                queue.append(each)
                visited.add(each)
                # 核心之一：一定要从graph的each节点中remove掉通往curr_point的路径
                # 因为当前访问的就是从curr_point到each这个节点的路径
                # 不需要再次访问了
                graph[each].remove(curr_point)
        
        # 如果是一棵valid tree
        # 肯定能全部访问过，并且只访问一次(以bfs的方式)
        return len(visited) == n
```

#### 431. 找无向图的连通块
```
"""
Definition for a undirected graph node
class UndirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []
"""
from collections import deque

class Solution:
    """
    @param: nodes: a array of Undirected graph node
    @return: a connected set of a Undirected graph
    """
    def connectedSet(self, nodes):
        # write your code here
        # BFS解法
        if not nodes:
            return []
        
        res = []
        visited = set()
        for node in nodes:
            if node.label in visited:
                continue
            queue = deque()
            queue.append(node)
            visited.add(node.label)
            temp = []
            while queue:
                curr = queue.popleft()
                temp.append(curr.label)
                for each_neighbor in curr.neighbors:
                    if each_neighbor.label not in visited:
                        queue.append(each_neighbor)
                        visited.add(each_neighbor.label)
            
            res.append(sorted(temp))
        
        return res
```

#### 71. 二叉树的锯齿形层次遍历
```
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
from collections import deque

class Solution:
    """
    @param root: A Tree
    @return: A list of lists of integer include 
    the zigzag level order traversal of its nodes' values.
    """
    def zigzagLevelOrder(self, root):
        # write your code here
        if not root:
            return []
        
        queue = deque()
        queue.append(root)
        res = []
        reverse = False
        while queue:
            q_len = len(queue)
            temp = []
            for _ in range(q_len):
                curr = queue.popleft()
                temp.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            if reverse:
                temp = temp[::-1]
            res.append(temp)
            reverse = not reverse
        
        return res
```

#### 70. 二叉树的层次遍历 II
```
from collections import deque

class Solution:
    """
    @param root: A tree
    @return: buttom-up level order a list of lists of integer
    """
    def levelOrderBottom(self, root):
        # write your code here
        if not root:
            return []
        
        queue = deque()
        queue.append(root)
        res = []
        
        while queue:
            q_len = len(queue)
            temp = []
            for _ in range(q_len):
                curr = queue.popleft()
                temp.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            res.append(temp)
        
        return res[::-1]
```

#### 794. 滑动拼图 II
```
# 在一个3x3的网格中，放着编号1到8的8块板，以及一块编号为0的空格。
# 一次移动可以把空格0与上下左右四邻接之一的板子交换。
# 给定初始和目标的板子排布，返回到目标排布最少的移动次数。
# 如果不能从初始排布移动到目标排布，返回-1.

from collections import deque
class Solution:
    """
    @param init_state: the initial state of chessboard
    @param final_state: the final state of chessboard
    @return: return an integer, denote the number of minimum moving
    """
    def minMoveStep(self, init_state, final_state):
        # # write your code here
        init_state_str = ''.join(str(i) for i in self._board_to_list(init_state))
        final_state_str = ''.join(str(i) for i in self._board_to_list(final_state))
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        queue = deque()
        queue.append(init_state)
        visited = set(init_state_str)
        
        res = 0
        while queue:
            q_len = len(queue)
            for _ in range(q_len):
                curr_board = queue.popleft()
                curr_state_str = ''.join(str(i) for i in self._board_to_list(curr_board))
                if curr_state_str == final_state_str:
                    return res
                ci, cj = self._get_zero_index(curr_board)
                for di, dj in dirs:
                    newi, newj = ci + di, cj + dj
                    if not 0 <= newi < 3 or \
                        not 0 <= newj < 3:
                        continue
                    new_board = [i[:] for i in curr_board]
                    new_board[ci][cj], new_board[newi][newj] = \
                        new_board[newi][newj], new_board[ci][cj]
                    new_state_str = ''.join(str(i) for i in self._board_to_list(new_board))
                    if new_state_str in visited:
                        continue
                    queue.append(new_board)
                    visited.add(new_state_str)
            res += 1
        
        return -1
    
    def _board_to_list(self, grid):
        res = []
        for i in range(3):
            for j in range(3):
                res.append(grid[i][j])
        return res
    
    def _list_to_board(self, curr_list):
        res = []
        for i in range(0, 9, 3):
            res.append(curr_list[i:i + 3])
        return res
    
    def _get_zero_index(self, board):
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return i, j
        raise ValueError('Can not find 0 index!')
```

#### 573. 邮局的建立 II
```
from collections import deque

class Solution:
    """
    @param grid: a 2D grid
    @return: An integer
    """
    def shortestDistance(self, grid):
        # write your code here
        if not grid or not grid[0]:
            return 2 ** 31 - 1
        
        m, n = len(grid), len(grid[0])
        building_cnt = 0
        distance = [[0] * n for _ in range(m)]
        reachable_cnt = [[0] * n for _ in range(m)]
        DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    building_cnt += 1
                    queue = deque()
                    visited = [[False] * n for _ in range(m)]
                    queue.append((i, j))
                    curr_dist = 0
                    # 这道题很有意思
                    # 起始入队的是一个grid[i][j]为1的点
                    # 后面在while loop里入队的是grid[i][j]为0的点
                    while queue:
                        q_len = len(queue)
                        curr_dist += 1
                        for _ in range(q_len):
                            ci, cj = queue.popleft()
                            for di, dj in DIRS:
                                newi, newj = ci + di, cj + dj
                                if not 0 <= newi < m or \
                                    not 0 <= newj < n or \
                                    grid[newi][newj] != 0 or \
                                    visited[newi][newj]:
                                    continue
                                distance[newi][newj] += curr_dist
                                reachable_cnt[newi][newj] += 1
                                visited[newi][newj] = True
                                queue.append((newi, newj))
        
        res = 2 ** 31 - 1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and reachable_cnt[i][j] == building_cnt:
                    res = min(res, distance[i][j])
        
        return res if res < 2 ** 31 - 1 else -1
```

#### 434. 岛屿的个数II
```
class Union:
    def __init__(self, n):
        self._parent = [i for i in range(n)]
        self.count = 0
    
    def _find(self, a):
        if self._parent[a] == a:
            return self._parent[a]
        self._parent[a] = self._find(self._parent[a])
        return self._parent[a]
    
    def connect(self, a, b):
        root_a = self._find(a)
        root_b = self._find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b
            self.count -= 1

class Solution:
    """
    @param n: An integer
    @param m: An integer
    @param operators: an array of point
    @return: an integer array
    """
    def numIslands2(self, n, m, operators):
        # write your code here
        board = [[0] * n for _ in range(m)]
        union = Union(n * m)
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        res = []
        for point in operators:
            ci, cj = point.y, point.x
            if board[ci][cj] == 0:
                board[ci][cj] = 1
                union.count += 1
                for di, dj in dirs:
                    newi, newj = ci + di, cj + dj
                    if 0 <= newi < m and 0 <= newj < n and board[newi][newj] == 1:
                        union.connect(ci * n + cj, newi * n + newj)
            res.append(union.count)
        
        return res
```

#### 600. 包裹黑色像素点的最小矩形
```
class Solution:
    """
    @param image: a binary matrix with '0' and '1'
    @param x: the location of one of the black pixels
    @param y: the location of one of the black pixels
    @return: an integer
    """
    def minArea(self, image, x, y):
        # write your code here
        if not image or not image[0]:
            return 0
        
        m, n = len(image), len(image[0])
        
        start, end = 0, x
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_row(mid, image):
                end = mid
            else:
                start = mid
        top_level = start if self._check_row(start, image) else end
        
        start, end = x, m - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_row(mid, image):
                start = mid
            else:
                end = mid
        bottom_level = end if self._check_row(end, image) else start
        
        start, end = 0, y
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_col(mid, image):
                end = mid
            else:
                start = mid
        left_level = start if self._check_col(start, image) else end
        
        start, end = y, n - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_col(mid, image):
                start = mid
            else:
                end = mid
        right_level = end if self._check_col(end, image) else start
        
        return (right_level - left_level + 1) * (bottom_level - top_level + 1)
    
    def _check_row(self, row, image):
        if any(i == '1' for i in image[row]):
            return True
        return False
    
    def _check_col(self, col, image):
        if any(i[col] == '1' for i in image):
            return True
        return False
```

#### 574. 邮局的建立
```
class Solution:
    """
    @param grid: a 2D grid
    @return: An integer
    """
    def shortestDistance(self, grid):
        # 思路是对的，但是本解法只能通过93%的test cases
        # write your code here
        # 注意这道题是可以穿过1的
        # 比如 [...1, 1, ...]是可以水平方向上两个1都访问到的
        if not grid or not grid[0]:
            return 2 ** 31 - 1
        
        m, n = len(grid), len(grid[0])
        house_x, house_y = [], []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    house_x.append(i)
                    house_y.append(j)
        
        min_dist = 2 ** 31 - 1
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    min_dist = min(
                        min_dist,
                        self._get_all_distance(i, j, house_x, house_y),
                    )
        
        return min_dist
    
    # 某个点到另外一个点的曼哈顿距离，就是在水平方向上的和加上垂直方向上的和
    def _get_all_distance(self, i, j, house_x, house_y):
        res = 0
        for x in house_x:
            res += abs(i - x)
        for y in house_y:
            res += abs(j - y)
        return ress
```

### 5 - Binary Tree & Tree-based DFS

#### 900. 二叉搜索树中最接近的值
```
class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @return: the value in the BST that is closest to the target
    """
    def closestValue(self, root, target):
        # write your code here
        if not root:
            return 2 ** 31 - 1
        
        res = root.val
        while root:
            if abs(res - target) > abs(root.val - target):
                res = root.val
            
            # 下面大于或者大于等于都是可以的
            if target >= root.val:
                root = root.right
            else:
                root = root.left
        
        return res
```

#### 596. 最小子树
```
class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the minimum subtree
    """
    def findSubtree(self, root):
        # write your code here
        self._min_node = None
        self._min_sum = 2 ** 31 - 1
        self._dfs(root)
        return self._min_node
    
    def _dfs(self, node):
        if not node:
            return 0
        
        left_sum = self._dfs(node.left)
        right_sum = self._dfs(node.right)
        curr_node_sum = node.val + left_sum + right_sum
        
        if curr_node_sum < self._min_sum:
            self._min_sum = curr_node_sum
            self._min_node = node
        
        return curr_node_sum
```

#### 480. 二叉树的所有路径
```
class Solution:
    """
    @param: root: the root of the binary tree
    @return: all root-to-leaf paths
    """
    def binaryTreePaths(self, root):
        # write your code here
        if not root:
            return []
        
        res = []
        self._dfs(root, [], res)
        return res
    
    
    def _dfs(self, node, curr, res):
        curr.append(str(node.val))
        if not node.left and not node.right:
            res.append('->'.join(curr))
            curr.pop()
            return
        
        if node.left:
            self._dfs(node.left, curr, res)
        if node.right:
            self._dfs(node.right, curr, res)
        
        curr.pop()
```

#### 453. 将二叉树拆成链表
```
class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        # write your code here
        while root:
            if root.left and root.right:
                curr = root.left
                while curr.right:
                    curr = curr.right
                curr.right = root.right
            if root.left:
                root.right = root.left
                root.left = None
            root = root.right
```

#### 93. 平衡二叉树
```
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        # write your code here
        if not root:
            return True
        
        left_height = self._get_max_height(root.left)
        right_height = self._get_max_height(root.right)
        
        return abs(left_height - right_height) <= 1 and \
            self.isBalanced(root.left) and self.isBalanced(root.right)
    
    def _get_max_height(self, node):
        if not node:
            return 0
        
        return max(
            self._get_max_height(node.left),
            self._get_max_height(node.right),
        ) + 1
```

#### 902. BST中第K小的元素
```
class Solution:
    """
    @param root: the given BST
    @param k: the given k
    @return: the kth smallest element in BST
    """
    def kthSmallest(self, root, k):
        if not root:
            return 2 ** 31 - 1
        # write your code here
        # 核心之一：计算当前root的左子树一共有多少个点
        # 然后判断当前的k是属于左半部分还是属于有半部分
        # 然后二分
        cnt = self._count_nodes(root.left)
        if k == cnt + 1:
            return root.val
        elif k > cnt + 1:
            return self.kthSmallest(root.right, k - cnt - 1)
        else:
            return self.kthSmallest(root.left, k)
    
    def _count_nodes(self, node):
        if not node:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right) 
```

#### 578. 最近公共祖先 III
```
class Solution:
    """
    @param: root: The root of the binary tree.
    @param: A: A TreeNode
    @param: B: A TreeNode
    @return: Return the LCA of the two nodes.
    """
    def lowestCommonAncestor3(self, root, A, B):
        # write your code here
        # 这道题跟Leetcode 236的区别在于
        # 这道题是允许A或者B不在树上的
        # 所以要加判断
        a, b, lca = self._dfs(root, A, B)
        if a and b:
            return lca
        else:
            return None
     
    # dfs定义：
    # 返回三个值：含义分别是A在不在以root为根的树上；B在不在以root为根的树上，
    # 以及A和B在以root为根的树上的LCA
    def _dfs(self, node, A, B):
        if not node:
            return False, False, None
        
        left_a, left_b, left_lca = self._dfs(node.left, A, B)
        right_a, right_b, right_lca = self._dfs(node.right, A, B)
        
        a = left_a or right_a or A.val == node.val
        b = left_b or right_b or B.val == node.val
        
        # 第三个条件是说在左子树上A和B的LCA是存在的，右子树上A和B的LCA也是存在的
        # 所以它俩的公共LCA就是当前的node
        if A.val == node.val or B.val == node.val or (left_lca and right_lca):
            return a, b, node

        return a, b, (left_lca if left_lca else right_lca)
```

#### 95. 验证二叉查找树
```
class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    _INT_MAX = 10 ** 10
    _INT_MIN = -10 ** 10

    def isValidBST(self, root):
        # write your code here
        return self._dfs(root, self._INT_MIN, self._INT_MAX)
    
    def _dfs(self, node, min_value, max_value):
        if not node:
            return True
        
        if not min_value < node.val < max_value:
            return False
        
        return self._dfs(node.left, min_value, node.val) and \
            self._dfs(node.right, node.val, max_value)
```

#### 901. 二叉搜索树中最接近的值 II
```
class Solution:
    """
    @param root: the given BST
    @param target: the given target
    @param k: the given k
    @return: k values in the BST that are closest to the target
    """
    def closestKValues(self, root, target, k):
        # write your code here
        if not root:
            return []
        
        arr = []
        self._in_order(root, arr)
        
        arr_list = []
        for inx, value in enumerate(arr):
            arr_list.append((abs(value - target), inx))
        
        # 必须sort一遍
        arr_list.sort()
        return [arr[i[1]] for i in arr_list[:k]]
    
    def _in_order(self, node, arr):
        if not node:
            return
        
        self._in_order(node.left, arr)
        arr.append(node.val)
        self._in_order(node.right, arr)
```

#### 86. 二叉查找树迭代器
```
class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        # do intialization if necessary
        self._stack = []
        self._push_left(root)
    """
    @return: True if there has next node, or false
    """
    def hasNext(self, ):
        # write your code here
        return len(self._stack) > 0
    """
    @return: return next node
    """
    def next(self):
        # write your code here
        top = self._stack.pop()
        self._push_left(top.right)
        return top
    
    def _push_left(self, node):
        while node:
            self._stack.append(node)
            node = node.left
```

#### 597. 具有最大平均数的子树
```
class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the maximum average of subtree
    """
    def findSubtree2(self, root):
        # write your code here
        self._node = None
        self._mean = -2 ** 31
        self._dfs(root)
        return self._node
    
    # dfs返回两个值：以node为根的树的sum是多少，以及以node为根的树有多少个节点
    def _dfs(self, node):
        if not node:
            return 0, 0
        
        left_sum, left_counts = self._dfs(node.left)
        right_sum, right_counts = self._dfs(node.right)
        
        node_sum = node.val + left_sum + right_sum
        node_counts = left_counts + right_counts + 1
        
        node_mean = node_sum / node_counts
        if node_mean > self._mean:
            self._mean = node_mean
            self._node = node
        
        return node_sum, node_counts
```

#### 474. 最近公共祖先 II
```
class Solution:
    """
    @param: root: The root of the tree
    @param: A: node in the tree
    @param: B: node in the tree
    @return: The lowest common ancestor of A and B
    """
    def lowestCommonAncestorII(self, root, A, B):
        # write your code here
        # 这道题限定了A和B一定在以root为根的树中
        # 基本思路就是沿着A的parent指针往上走，不停的将A的parent加入到
        # 一个set中
        # 则早晚会找到A和B的共同祖先
        if not root:
            return
        
        parents = set()
        while A:
            parents.add(A.val)
            A = A.parent
        while B:
            if B.val in parents:
                return B
            B = B.parent
        
        # 由于限定了A和B一定在树中
        # return一定会发生在上面B的循环里
        raise ValueError('should not see me!')
```

246. 二叉树的路径和 II
```
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum2(self, root, target):
        # write your code here
        if not root:
            return []
        
        res = []
        self._dfs(root, target, 0, [], res)
        return res
    
    def _dfs(self, node, target, level, curr, res):
        if not node:
            return

        curr.append(node.val)
        
        required = target
        for l in range(level, -1, -1):
            required -= curr[l]
            if required == 0:
                # 注意这里不能直接return
                # 因为还可能有其他的解
                res.append(curr[l:])
        
        self._dfs(node.left, target, level + 1, curr, res)
        self._dfs(node.right, target, level + 1, curr, res)
        
        curr.pop()
```

#### 155. 二叉树的最小深度
```
class Solution:
    """
    @param root: The root of binary tree
    @return: An integer
    """
    def minDepth(self, root):
        # write your code here
        if not root:
            return 0
        
        if not root.left and not root.right:
            return 1
        
        if not root.left:
            return 1 + self.minDepth(root.right)
        
        if not root.right:
            return 1 + self.minDepth(root.left)
        
        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))
```

#### 97. 二叉树的最大深度
```
class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxDepth(self, root):
        # write your code here
        if not root:
            return 0
        
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

#### 68. 二叉树的后序遍历
```
from collections import deque

class Solution:
    """
    @param root: A Tree
    @return: Postorder in ArrayList which contains node values.
    """
    def postorderTraversal(self, root):
        # write your code here
        if not root:
            return []
        # 很好的思路: 正好跟先序遍历反着来，用队列来存结果(不停的放到队头)
        res = deque()
        stack = [root]
        while stack:
            top = stack.pop()
            res.appendleft(top.val)
            if top.left:
                stack.append(top.left)
            if top.right:
                stack.append(top.right)
        return list(res)
```

#### 67. 二叉树的中序遍历
```
class Solution:
    """
    @param root: A Tree
    @return: Inorder in ArrayList which contains node values.
    """
    def inorderTraversal(self, root):
        # write your code here
        # 基本思路就是先把所有的左孩子入栈
        res = []
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            top = stack.pop()
            res.append(top.val)
            root = top.right
        return res
```

#### 66. 二叉树的前序遍历
```
class Solution:
    """
    @param root: A Tree
    @return: Preorder in ArrayList which contains node values.
    """
    def preorderTraversal(self, root):
        # write your code here
        if not root:
            return []
        
        res = []
        stack = [root]
        while stack:
            top = stack.pop()
            res.append(top.val)
            if top.right:
                stack.append(top.right)
            if top.left:
                stack.append(top.left)
        
        return res
```

#### 915. BST的中序前驱节点
```
class Solution:
    """
    @param root: the given BST
    @param p: the given node
    @return: the in-order predecessor of the given node in the BST
    """
    def inorderPredecessor(self, root, p):
        # write your code here
        # 这里的prev就是遍历时候每次都记录一下当前在递归中
        # 遍历的root的前一个节点是什么
        # 当发现root == p的时候
        # 将prev节点赋给最终的result_node就好了
        self._prev = None
        self._result_node = None
        self._dfs(root, p)
        return self._result_node
        
        """更好的写法（利用了BST性质）:
            if root.val > p.val:
                return self.inorderPredecessor(root.left, p)
            elif root.val == p.val:
                # 因为是中序遍历，当前root的前去就是root的left节点
                return root.left
            else:
                node = self.inorderPredecessor(root.right, p)
                if node is None:
                    return root
                else:
                    return node
        """
    
    def _dfs(self, root, p):
        if not root:
            return
        self._dfs(root.left, p)
        if root == p:
            # 实际上九章的答案里不是直接return的
            # 这里直接return即可，还能起到剪枝作用
            self._result_node = self._prev
            return
        self._prev = root
        self._dfs(root.right, p)
```

#### 448. 二叉查找树的中序后继
```
class Solution:
    """
    @param: root: The root of the BST.
    @param: p: You need find the successor node of p.
    @return: Successor of p.
    """
    def inorderSuccessor(self, root, p):
        # write your code here
        # 这道题就是说在BST中找p的下一个比它大的节点
        res = None
        while root:
            if root.val > p.val:
                res = root
                root = root.left
            else:
                root = root.right
        return res
```

#### 88. 最近公共祖先
```
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        # write your code here
        # 这道题是限定了给出的两个节点都在树中存在
        if not root or A.val == root.val or B.val == root.val:
            return root
        
        left_node = self.lowestCommonAncestor(root.left, A, B)
        right_node = self.lowestCommonAncestor(root.right, A, B)
        
        if left_node and right_node:
            return root
        
        return left_node if left_node else right_node
```

#### 472. 二叉树的路径和 III
```
class Solution:
    """
    @param: root: the root of binary tree
    @param: target: An integer
    @return: all valid paths
    """
    def binaryTreePathSum3(self, root, target):
        # write your code here
        # 这道题跟LeetCode 437 Path Sum III不同的地方在于
        # 这道题要求求所有的方案, 那道题需要求方案的种类数
        # 而且每个节点有一个parent指针
        # 这道题是两层递归
        res = []
        self._dfs(root, target, res)
        return res
    
    def _dfs(self, node, target, res):
        if not node:
            return
        
        # 先去递归巡展包含node本身的path
        self._find_sum(node, None, target, [], res)
        
        # 再去递归不包含node本身的path
        self._dfs(node.left, target, res)
        self._dfs(node.right, target, res)
    
    def _find_sum(self, node, from_, target, curr, res):
        curr.append(node.val)
        
        target -= node.val
        if target == 0:
            res.append(curr[:])
        
        if node.parent and node.parent is not from_:
            self._find_sum(node.parent, node, target, curr, res)
        
        if node.left and node.left is not from_:
            self._find_sum(node.left, node, target, curr, res)
        
        if node.right and node.right is not from_:
            self._find_sum(node.right, node, target, curr, res)
        
        curr.pop()
```

#### 595. 二叉树最长连续序列
```
class Solution:
    """
    @param root: the root of binary tree
    @return: the length of the longest consecutive sequence path
    """
    def longestConsecutive(self, root):
        # write your code here
        self._res = 0
        self._dfs(root)
        return self._res
    
    def _dfs(self, node):
        if not node:
            return 0
        
        left_longest = self._dfs(node.left)
        right_longest = self._dfs(node.right)
        
        node_length = 1
        if node.left and node.left.val - 1 == node.val:
            node_length = left_longest + 1
        if node.right and node.right.val - 1 == node.val:
            node_length = max(node_length, right_longest + 1)
        
        # 返回的是局部最优
        # 但是用局部最优去更新全局最优
        self._res = max(self._res, node_length)
        return node_length
```

#### 619. 二叉树的最长连续子序列III
```
class Solution:
    # @param {MultiTreeNode} root the root of k-ary tree
    # @return {int} the length of the longest consecutive sequence path
    def longestConsecutive3(self, root):
        # Write your code here
        max_len, _, _ = self._dfs(root)
        return max_len
    
    def _dfs(self, node):
        if not node:
            return 0, 0, 0
        
        max_len = up = down = 0
        for child in node.children:
            child_max_len, child_up, child_down = self._dfs(child)
            max_len = max(max_len, child_max_len)
            if node.val - 1 == child.val:
                down = max(down, child_down + 1)
            if node.val + 1 == child.val:
                up = max(up, child_up + 1)
        
        max_len = max(max_len, up + down + 1)
        return max_len, up, down
```

#### 614. 二叉树的最长连续子序列 II
```
class Solution:
    """
    @param root: the root of binary tree
    @return: the length of the longest consecutive sequence path
    """
    def longestConsecutive2(self, root):
        # write your code here
        self._res = 0
        self._dfs(root)
        return self._res
    
    # DFS做的是以node为根，最长能递减多少，最长能递增多少
    def _dfs(self, node):
        if not node:
            return 0, 0
        
        # 这里的inc和dec在两次循环中是不相关独立的
        # 所以最终可以直接相加
        inc = dec = 0
        for each in (node.left, node.right):
            if not each:
                continue
            each_inc, each_dec = self._dfs(each)
            if each.val - 1 == node.val:
                inc = max(inc, each_inc)
            if each.val + 1 == node.val:
                dec = max(dec, each_dec)
        
        self._res = max(self._res, dec + inc + 1)

        return inc + 1, dec + 1
```

#### 520. Consistent Hashing II
```
from random import randint
from bisect import bisect_left

class Solution:
    """
    @param {int} n a positive integer
    @param {int} k a positive integer
    @return {Solution} a Solution object
    """
    @classmethod
    def create(cls, n, k):
        # Write your code here
        # n是将圆环分成多少个小区间，通常n = 2 ** 64
        # k是有多少个虚拟点，通常是1000
        obj = cls()
        obj.ids = set()
        obj.machines = dict()
        obj.n = n
        obj.k = k
        return obj
    """
    @param: machine_id: An integer
    @return: a list of shard ids
    """
    def addMachine(self, machine_id):
        # write your code here
        # 如果这个machine_id已经在当前的圆环里了
        # 直接返回
        if machine_id in self.machines:
            return

        res = []
        for _ in range(self.k):
            inx = randint(0, self.n - 1)
            # 因为这个inx不能是重复的
            # 所以如果生成了一个重复的
            # 就要重新生成
            while inx in self.ids:
                inx = randint(0, self.n - 1)
            self.ids.add(inx)
            res.append(inx)

        res.sort()
        self.machines[machine_id] = res
        return res
    """
    @param: hashcode: An integer
    @return: A machine id
    """
    def getMachineIdByHashCode(self, hashcode):
        # write your code here
        # 这个函数是说，通过某个hash函数已经算出来了一个数据的hashCode
        # 要寻找这个数据对应在哪台machine上
        # 返回这个machine id
        # 所以基本思路就是在圆环顺时针里找跟当前距离最近的machine_id
        # 注意这个machine_id已经存成了id: 点集合的形式
        res = -1
        # 因为最大的点的id就是n - 1
        distance = self.n
        
        for machine_id, locations in self.machines.items():
            # 注意这里的locations已经在addMachine函数中sort过了
            # bisect_left函数就是二分查找插入的位置
            # 和bisect_right的区别就是当查找的元素已经在array里的时候
            # bisect_left返回的是最左边可以插入的位置
            # bisect_right返回的是最右边可以插入的位置
            inx = bisect_left(locations, hashcode) % len(locations)
            dist = locations[inx] - hashcode
            if dist < 0:
                dist += self.n
            if dist < distance:
                distance = dist
                res = machine_id
        
        return res
```

#### 475. 二叉树的最大路径和 II
```
class Solution:
    """
    @param root: the root of binary tree.
    @return: An integer
    """
    _INT_MIN = -2 ** 31
    def maxPathSum2(self, root):
        # write your code here
        if not root:
            return self._INT_MIN
        
        if not root.left and not root.right:
            return root.val
        
        left_max = self.maxPathSum2(root.left)
        right_max = self.maxPathSum2(root.right)
        return root.val + max(0, left_max, right_max)
```

#### 94. 二叉树中的最大路径和
```
class Solution:
    """
    @param root: The root of binary tree.
    @return: An integer
    """
    def maxPathSum(self, root):
        # write your code here
        self._res = -2 ** 31
        self._dfs(root)
        return self._res
    
    def _dfs(self, node):
        if not node:
            return 0
        
        left = max(self._dfs(node.left), 0)
        right = max(self._dfs(node.right), 0)
        
        self._res = max(self._res, left + right + node.val)
        return max(left, right) + node.val
```

### 6 - Combination-based DFS

#### 680. 分割字符串
```
class Solution:
    """
    @param: : a string to be split
    @return: all possible split string array
    """

    def splitString(self, s):
        # write your code here
        res = []
        self._dfs(s, 0, [], res)
        return res
    
    def _dfs(self, s, start, curr, res):
        if start >= len(s):
            if start == len(s):
                res.append(curr[:])
            return
        
        # DFS有多少种选择，在每种选择里递归下去  
        for i in range(1, 3):
            curr.append(s[start:start + i])
            self._dfs(s, start + i, curr, res)
            curr.pop()
```

#### 136. 分割回文串
```
class Solution:
    """
    @param: s: A string
    @return: A list of lists of string
    """
    def partition(self, s):
        # write your code here
        res = []
        self._dfs(s, 0, [], res)
        return res
    
    def _dfs(self, s, start, curr, res):
        if start == len(s):
            res.append(curr[:])
            return
        # i的含义理解是从start开始加多少个字符
        for i in range(start, len(s)):
            # 注意i从是start开始的
            if self._check(s[start:i + 1]):
                curr.append(s[start:i + 1])
                # 因为已经从s中加了s[start:i + 1]到curr中了
                # 所以下一次递归要从i + 1开始
                self._dfs(s, i + 1, curr, res)
                curr.pop()
    
    def _check(self, word):
        l, r = 0, len(word) - 1
        while l < r:
            if word[l] != word[r]:
                return False
            l += 1
            r -= 1
        return True
```

#### 153. 数字组合 II
```
class Solution:
    """
    @param nums: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, nums, target):
        # write your code here
        nums.sort()
        res = []
        self._dfs(nums, target, 0, [], res)
        return res
    
    def _dfs(self, nums, target, start, curr, res):
        if target == 0:
            res.append(curr[:])
            return 
        
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            if target - nums[i] >= 0:
                curr.append(nums[i])
                self._dfs(nums, target - nums[i], i + 1, curr, res)
                curr.pop()
            else:
                break
```

#### 152. 组合
```
class Solution:
    """
    @param n: Given the range of numbers
    @param k: Given the numbers of combinations
    @return: All the combinations of k numbers out of 1..n
    """
    def combine(self, n, k):
        # write your code here
        res = []
        self._dfs(n, k, 1, [], res)
        return res
    
    def _dfs(self, n, k, start, curr, res):
        if k == 0:
            res.append(curr[:])
            return
        
        for i in range(start, n + 1):
            curr.append(i)
            self._dfs(n, k - 1, i + 1, curr, res)
            curr.pop()
```

#### 135. 数字组合
```
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """
    def combinationSum(self, candidates, target):
        # write your code here
        candidates.sort()
        res = []
        self._dfs(candidates, target, 0, [], res)
        return res
    
    def _dfs(self, candidates, target, start, curr, res):
        if target == 0:
            res.append(curr[:])
            return

        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            if target - candidates[i] >= 0:
                curr.append(candidates[i])
                self._dfs(candidates, target - candidates[i], i, curr, res)
                curr.pop()
            else:
                break
```

#### 18. 子集 II
```
class Solution:
    """
    @param nums: A set of numbers.
    @return: A list of lists. All valid subsets.
    """
    def subsetsWithDup(self, nums):
        # write your code here
        nums.sort()
        res = []
        self._dfs(nums, 0, [], res)
        return res
    
    def _dfs(self, nums, start, curr, res):
        res.append(curr[:])
        if start == len(nums):
            return
        
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            curr.append(nums[i])
            self._dfs(nums, i + 1, curr, res)
            curr.pop()
```

#### 17. 子集
```
class Solution:
    """
    @param nums: A set of numbers
    @return: A list of lists
    """
    def subsets(self, nums):
        # write your code here
        nums.sort()
        res = []
        self._dfs(nums, 0, [], res)
        return res
    
    def _dfs(self, nums, start, curr, res):
        res.append(curr[:])
        if start == len(nums):
            return
        
        for i in range(start, len(nums)):
            curr.append(nums[i])
            self._dfs(nums, i + 1, curr, res)
            curr.pop()
```

#### 582. 单词拆分II
```
class Solution:
    """
    @param: s: A string
    @param: wordDict: A set of words.
    @return: All possible sentences.
    """
    def wordBreak(self, s, wordDict):
        # write your code here
        res = []
        valid_range = [True] * len(s)
        self._dfs(s, 0, set(wordDict), valid_range, [], res)
        return res
    
    def _dfs(self, s, start, words, valid_range, curr, res):
        if start == len(s):
            res.append(' '.join(curr))
            return
        
        for i in range(start, len(s)):
            if s[start:i + 1] not in words:
                continue
            if not valid_range[i]:
                continue
            # 这道题的trick有两个：
            # 1. 用了一个valid_range数组来剪枝，valid_range[i]表示
            #    从i到最后的s string能否增加结果
            #    如果不能 就设置为False, 起到剪枝的作用
            # 2. 先存一下旧结果的长度，如果新结果的长度并不能增加
            #    旧结果的长度，将valid_range[i]设置为False
            old_res_length = len(res)
            curr.append(s[start:i + 1])
            self._dfs(s, i + 1, words, valid_range, curr, res)
            curr.pop()
            if len(res) == old_res_length:
                valid_range[i] = False
```

#### 192. 通配符匹配
```
class Solution:
    """
    @param s: A string 
    @param p: A string includes "?" and "*"
    @return: is Match?
    """
    def isMatch(self, s, p):
        # write your code here
        n, m = len(s), len(p)
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        
        if n == 0 and p.count('*') == m:
            return True
        
        dp[0][0] = True
        for i in range(n + 1):
            for j in range(m + 1):
                if i > 0 and j > 0:
                    # 条件1
                    dp[i][j] = dp[i - 1][j - 1] and \
                        (s[i - 1] == p[j - 1] or p[j - 1] in ('?', '*'))
                    # 条件2
                    dp[i][j] |= dp[i - 1][j] and p[j - 1] in ('?', '*')
                if j > 0:
                    # 条件3
                    dp[i][j] |= dp[i][j - 1] and p[j - 1] == '*'
        
        return dp[n][m]
```

#### 154. 正则表达式匹配
```
class Solution:
    """
    @param s: A string 
    @param p: A string includes "." and "*"
    @return: A boolean
    """
    def __init__(self):
        self._hash = {}

    def isMatch(self, s, p):
        # write your code here
        # 注意*号表示前面一个或多个字母
        # 但是必须是同一个字母！！！
        # 这道题要用记忆化搜索 否则OJ不通过
        key = s + ',' + p
        if key in self._hash:
            return self._hash[key]
        
        if not p:
            self._hash[key] = not s
            return self._hash[key]
        
        # 注意这里是正则匹配，需要看前面的字符是什么（跟通配符匹配不同）
        # 所以此时p[0]不可以为"*"
        first_match = bool(s) and p[0] in (s[0], '.')
        
        if len(p) > 1 and p[1] == '*':
            # 下面两个条件表示或者认为s头一个字母已经匹配上了
            # 或者不取p的前两个字母
            self._hash[key] = (first_match and self.isMatch(s[1:], p)) or \
                self.isMatch(s, p[2:])
        # 此时len(p) == 1或者p[1]不为"*"
        else:
            self._hash[key] = first_match and self.isMatch(s[1:], p[1:])
        
        return self._hash[key]
```

#### 652. 因式分解
```
from math import sqrt

class Solution:
    """
    @param n: An integer
    @return: a list of combination
    """
    def getFactors(self, n):
        # write your code here
        if n < 2:
            return []

        res = []
        self._dfs(n, 2, [], res)
        return res
    
    def _dfs(self, n, start, curr, res):
        if n == 1:
            # 为啥？
            # 因为可能出现curr里只有一个数字
            # 这个数字正好是n本身
            if len(curr) > 1:
                res.append(curr[:])
            return
    
        for i in range(start, int(sqrt(n)) + 1):
            # 发现满足条件的一个解了，可以开始递归
            if n % i == 0:
                curr.append(i)
                self._dfs(n // i, i, curr, res)
                curr.pop()
        
        # 上面使用sqrt虽然可以剪枝
        # 但是会漏掉一个非常重要的case:
        # 就是当curr已经有东西（length > 0 而且说明剩下的能被整除）
        # 到底漏掉的是什么case？？？
        if n >= start:
            curr.append(n)
            self._dfs(1, n, curr, res)
            curr.pop()
        # 下面的也是一样的
        # if curr:
        #     res.append(curr + [n])
```

#### 570. 寻找丢失的数 II
```
class Solution:
    """
    @param n: An integer
    @param string: a string with number from 1-n in random order and miss one number
    @return: An integer
    """
    def findMissing2(self, n, string):
        # write your code here
        # 典型DFS题目
        visited = [False] * (n + 1)
        return self._dfs(n, string, 0, visited)
    
    def _dfs(self, n, string, start, visited):
        if start == len(string):
            missing = []
            for i in range(1, n + 1):
                if not visited[i]:
                    missing.append(i)
            return missing[0] if len(missing) == 1 else -1
        
        if string[start] == '0':
            return -1
        
        for i in range(1, 3):
            num = int(str(string[start:start + i]))
            if 1 <= num <= n and not visited[num]:
                visited[num] = True
                # 坑：注意这里的下一次递归的start应该是start + i
                # 因为这里的start定义是从string的哪里开始
                inx = self._dfs(n, string, start + i, visited)
                if inx != -1:
                    return inx
                visited[num] = False
        
        return -1
```

#### 426. 恢复IP地址
```
class Solution:
    """
    @param s: the IP string
    @return: All possible valid IP addresses
    """
    def restoreIpAddresses(self, s):
        # write your code here
        # 给一个由数字组成的字符串
        # 求出其可能恢复为的所有IP地址
        if not s:
            return []
        
        res = []
        self._dfs(s, 0, [], res)
        return res
    
    def _dfs(self, s, segs, curr, res):
        if segs == 4:
            if not s:
                res.append('.'.join(curr))
            return

        if s and s[0] == '0':
            curr.append(s[0])
            self._dfs(s[1:], segs + 1, curr, res)
            curr.pop()
            return
        
        for i in range(1, 4):
            if i > len(s):
                break
            if 0 <= int(s[:i]) <= 255:
                curr.append(s[:i])
                self._dfs(s[i:], segs + 1, curr, res)
                curr.pop()
```

#### 427. 生成括号
```
class Solution:
    """
    @param n: n pairs
    @return: All combinations of well-formed parentheses
    """
    def generateParenthesis(self, n):
        # write your code here
        res = []
        self._dfs(n, n, '', res)
        return res
    
    def _dfs(self, left, right, curr, res):
        if left > right:
            return
        
        if left == 0 and right == 0:
            res.append(curr)
            return
        
        if left > 0:
            self._dfs(left - 1, right, curr + '(', res)
        if right > 0:
            self._dfs(left, right - 1, curr + ')', res)
```

#### 780. 删除无效的括号
```
from collections import deque

class Solution:
    """
    @param s: The input string
    @return: Return all possible results
    """
    def removeInvalidParentheses(self, s):
        # Write your code here
        visited = set()
        done = False
        queue = deque()
        queue.append(s)
        visited.add(s)
        res = []
        
        while queue:
            curr = queue.popleft()
            if self._is_valid(curr):
                done = True
                res.append(curr)

            # 在已经出现done的情况下
            # 只需要把平级（指的是长度一样的情况）
            # 的字符串处理完即可
            # 不再需要往队列里加东西了
            # 即下面的for循环是没有必要的
            if done:
                continue
            
            for i in range(len(curr)):
                if curr[i] in ('(', ')'):
                    new_str = curr[:i] + curr[i + 1:]
                    if new_str not in visited:
                        queue.append(new_str)
                        visited.add(new_str)
        
        return res
        
    def _is_valid(self, s):
        count = 0
        for ch in s:
            if ch == '(':
                count += 1
            elif ch == ')':
                count -= 1
            
            if count < 0:
                return False
        
        return count == 0
```

#### 683. 单词拆分 III
```
class Solution:
    """
    @param: : A string
    @param: : A set of word
    @return: the number of possible sentences.
    """

    def wordBreak3(self, s, word_set):
        # Write your code here
        if not s or not word_set:
            return 0
        
        new_word_set = {i.lower() for i in word_set}
        res = set()
        self._dfs(s.lower(), new_word_set, 0, [], res)
        return len(res)
    
    def _dfs(self, s, word_set, start, curr, res):
        if start == len(s):
            new_string = ' '.join(curr)
            if new_string not in res:
                res.add(new_string)
            return
        
        for i in range(start, len(s)):
            sub_string = s[start:i + 1]
            if sub_string in word_set:
                curr.append(sub_string)
                self._dfs(s, word_set, i + 1, curr, res)
                curr.pop()
```

#### 196. 寻找缺失的数
```
class Solution:
    """
    @param nums: An array of integers
    @return: An integer
    """
    def findMissing(self, nums):
        # write your code here
        
        # if not nums:
        #     return -1
            
        # sum_val = sum(nums)
        # n = len(nums)
        # return n * (n + 1) // 2 - sum_val
        
        nums.sort()
        n = len(nums)
        for i in range(n):
            if nums[i] != i:
                return i
        return n
```

#### 107. 单词拆分 I
```
class Solution:
    """
    @param: s: A string
    @param: word_dict: A dictionary of words dict
    @return: A boolean
    """
    def wordBreak(self, s, word_dict):
        # write your code here
        if not s:
            return True
        if not word_dict:
            return False
        
        n = len(s)
        # dp定义：s中的前i个字符能否用一个或多个空格分开
        # 使得每个分开的子串都出现在word_dict里
        dp = [False] * (n + 1)
        dp[0] = True
        
        # max_word_len主要起到剪枝作用
        max_word_len = max(len(i) for i in word_dict)
        for i in range(1, n + 1):
            for j in range(1, min(i, max_word_len) + 1):
                # 如果s的前i - j个字符串满足条件
                # 并且当前s的子串s[i - j:i]出现在word_dict里
                # 说明当前的dp[i]满足条件
                if dp[i - j] and s[i - j:i] in word_dict:
                    dp[i] = True
                    break
        
        return dp[-1]

# 这道题的DFS会TLE
# class Solution:
#     """
#     @param: s: A string
#     @param: word_dict: A dictionary of words dict
#     @return: A boolean
#     """
#     def wordBreak(self, s, word_dict):
#         # write your code here
#         return self._dfs(s, 0, [-1] * len(s), word_dict)
    
#     def _dfs(self, s, start, is_valid, word_dict):
#         if start == len(s):
#             return True
        
#         if is_valid[start] != -1:
#             return is_valid[start]
        
#         for i in range(start, len(s)):
#             if s[start:i + 1] in word_dict:
#                 if self._dfs(s, i + 1, is_valid, word_dict):
#                     is_valid[start] = 1
#                     return True
        
#         is_valid[start] = 0
#         return False
```

#### 108. 分割回文串 II
```
class Solution:
    """
    @param s: A string
    @return: An integer
    """
    def minCut(self, s):
        # write your code here
        # 将s分割成一些子串
        # 使每个子串都是回文
        # 返回s符合要求的的最少分割次数
        # 核心思路：
        # dp[i]是dp[j - 1] + 1 (j <= i) if s的[j:i]是回文
        n = len(s)
        # is_palin[i][j]表示s中的子串[i-j]左闭右闭是不是palindrome
        is_palin = [[False] * n for _ in range(n)]
        # dp[i]表示s中的子串[0-i]左闭右闭的min cut是多少
        dp = [n] * n
        
        for i in range(n):
            # 比如第i=1个子串ab只有两个字符
            # 上限的cuts就是1，砍一刀就行
            min_cut = i
            for j in range(i + 1):
                # 后一个条件表示s的子串j + 1到i - 1是不是palindrome
                if s[i] == s[j] and (j + 1 > i - 1 or is_palin[j + 1][i - 1]):
                    is_palin[j][i] = True
                    if j > 0:
                        min_cut = min(min_cut, dp[j - 1] + 1)
                    else:
                        # 此时j = 0, 则如果s当前s[0:i]是回文
                        # 不需要砍了，min_cut就是0
                        min_cut = 0
            dp[i] = min_cut
        
        return dp[-1]
```

### 7 - Permutation-based & Graph-based DFS

#### 425. 电话号码的字母组合
```
class Solution:
    """
    @param digits: A digital string
    @return: all posible letter combinations
    """

    LETTER_MAP = {
        '0': ' ',
        '1': '',
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz',
    }

    def letterCombinations(self, digits):
        # write your code here
        if not digits:
            return []
        
        res = []
        self._dfs(digits, 0, '', res)
        return res
    
    def _dfs(self, digits, start, curr, res):
        if start == len(digits):
            res.append(curr)
            return
        
        letters = self.LETTER_MAP[digits[start]]
        for ch in letters:
            self._dfs(digits, start + 1, curr + ch, res)
```

#### 10. 字符串的不同排列
```
class Solution:
    """
    @param string: A string
    @return: all permutations
    """
    def stringPermutation2(self, string):
        # write your code here
        res = []
        new_string = ''.join(sorted(string))
        self._dfs(new_string, '', res)
        return res
    
    def _dfs(self, string, curr, res):
        if not string:
            res.append(curr)
            return
        
        for i in range(len(string)):
            if i > 0 and string[i] == string[i - 1]:
                continue
            new_str = string[:i] + string[i + 1:]
            self._dfs(new_str, curr + string[i], res)
```

#### 34. N皇后问题 II
```
class Solution:
    """
    @param n: The number of queens.
    @return: The total number of distinct solutions.
    """
    def totalNQueens(self, n):
        # write your code here
        self._res = 0
        self._dfs(0, [None] * n)
        return self._res
    
    def _dfs(self, row, pos):
        n = len(pos)
        if row == n:
            self._res += 1
            return
        
        for col in range(n):
            if self._valid(row, col, pos):
                pos[row] = col
                self._dfs(row + 1, pos)
        
    def _valid(self, row, col, pos):
        for i in range(row):
            # 第二个条件表示
            # 在第i行上的pos[i]列上有一个皇后
            # 然后看这个皇后和现在在(row, col)上放置的皇后行差和列差是不是相等
            # 如果相等，说明是出于对角线上的，返回False
            if pos[i] == col or abs(row - i) == abs(col - pos[i]):
                return False
        return True
```

#### 33. N皇后问题
```
class Solution:
    """
    @param: n: The number of queens
    @return: All distinct solutions
    """
    def solveNQueens(self, n):
        # write your code here
        res = []
        self._dfs([-1] * n, 0, [], res)
        return res
    
    def _dfs(self, positions, curr_row_inx, path, res):
        if curr_row_inx == len(positions):
            res.append(path[:])
            return
        
        for col in range(len(positions)):
            # 将curr_row_inx的行的第col列上尝试放置皇后
            positions[curr_row_inx] = col
            if self._valid(positions, curr_row_inx):
                temp = '.' * len(positions)
                path.append(temp[:col] + 'Q' + temp[col + 1:])
                self._dfs(
                    positions,
                    curr_row_inx + 1,
                    path,
                    res,
                )
                path.pop()
    
    def _valid(self, positions, curr_row_inx):
        for i in range(curr_row_inx):
            if curr_row_inx - i == abs(positions[curr_row_inx] - positions[i]) or \
                positions[curr_row_inx] == positions[i]:
                return False
        return True
```

#### 16. 带重复元素的排列
```
class Solution:
    """
    @param: :  A list of integers
    @return: A list of unique permutations
    """

    def permuteUnique(self, nums):
        # write your code here
        if not nums:
            return [[]]
        
        nums.sort()
        res = []
        self._dfs(nums, [], res)
        return res
    
    def _dfs(self, nums, curr, res):
        if not nums:
            res.append(curr[:])
        
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            curr.append(nums[i])
            self._dfs(nums[:i] + nums[i + 1:], curr, res)
            curr.pop()
```

#### 15. 全排列
```
class Solution:
    """
    @param: nums: A list of integers.
    @return: A list of permutations.
    """
    def permute(self, nums):
        # write your code here
        res = []
        self._dfs(nums, [], res)
        return res
    
    def _dfs(self, nums, curr, res):
        if not nums:
            res.append(curr[:])
            return
        
        for i in range(len(nums)):
            curr.append(nums[i])
            self._dfs(nums[:i] + nums[i + 1:], curr, res)
            curr.pop()
```

#### 829. 字模式 II
```
class Solution:
    """
    @param pattern: a string,denote pattern string
    @param string: a string, denote matching string
    @return: a boolean
    """
    def wordPatternMatch(self, pattern, string):
        # write your code here
        return self._dfs(pattern, string, dict(), set())
    
    def _dfs(self, pattern, string, mapping, used):
        if not pattern:
            return not string
        
        ch = pattern[0]
        if ch in mapping:
            word = mapping[ch]
            if not string.startswith(word):
                return False
            return self._dfs(pattern[1:], string[len(word):], mapping, used)
        
        for i in range(len(string)):
            word = string[:i + 1]
            if word in used:
                continue
            
            mapping[ch] = word
            used.add(word)
            
            if self._dfs(pattern[1:], string[len(word):], mapping, used):
                return True
            
            del mapping[ch]
            used.remove(word)
        
        return False
```

#### 132. 单词搜索 II
```
class TrieNode:
    def __init__(self):
        self.children = [None] * 26
        self.is_word = False
        self.string = ''

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def add(self, word):
        curr = self.root
        for ch in word:
            inx = ord(ch) - ord('a')
            if not curr.children[inx]:
                curr.children[inx] = TrieNode()
            curr = curr.children[inx]
        curr.is_word = True
        curr.string = word
    
    def find(self, word):
        curr = self.root
        for ch in word:
            inx = ord(ch) - ord('a')
            if not curr.children[inx]:
                return False
            curr = curr.children[inx]
        return curr.is_word

class Solution:
    """
    @param board: A list of lists of character
    @param words: A list of string
    @return: A list of string
    """
    DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    def wordSearchII(self, board, words):
        # write your code here
        if not board or not board[0] or not words:
            return []

        trie = Trie()
        for word in words:
            trie.add(word)
        
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        res = set()
        for i in range(m):
            for j in range(n):
                self._dfs(board, trie.root, i, j, visited, res)
        
        return list(res)
    
    # trie的作用是：
    # 快速判断以root为根的树上存不存在以某个字符开头的词
    def _dfs(self, board, root, i, j, visited, res):
        visited[i][j] = True
        
        m, n = len(board), len(board[0])
        inx = ord(board[i][j]) - ord('a')
        if root.children[inx]:
            root = root.children[inx]
            if root.is_word and root.string not in res:
                res.add(root.string)
            for di, dj in self.DIRS:
                newi, newj = i + di, j + dj
                if not 0 <= newi < m or not 0 <= newj < n or visited[newi][newj]:
                    continue
                self._dfs(board, root, newi, newj, visited, res)
        
        visited[i][j] = False
```

#### 121. 单词接龙 II
```
from collections import defaultdict
from collections import deque

class Solution:
    """
    @param: start: a string
    @param: end: a string
    @param: dict: a set of string
    @return: a list of lists of string
    """
    def findLadders(self, start, end, word_list):
        # write your code here
        word_set = set(word_list)
        word_set.add(start)
        word_set.add(end)
        
        remove_one_char = self._remove_one_char_mapping(word_set)
        # distance里面的定义是每个词（包括start）
        # 到end这个词的距离
        distance = dict()
        self._bfs(end, distance, remove_one_char)
        
        if start not in distance:
            return []
        
        res = []
        self._dfs(start, end, distance, remove_one_char, [start], res)
        return res
    
    def _bfs(self, begin_word, distance, remove_one_char):
        # 这里的begin_word实际上是终点词
        # 我们将终点词当成begin word
        # 相当于反向遍历bfs
        distance[begin_word] = 0
        queue = deque()
        queue.append(begin_word)
        while queue:
            word = queue.popleft()
            for next_word in self._get_next_word(word, remove_one_char):
                if next_word not in distance:
                    distance[next_word] = distance[word] + 1
                    queue.append(next_word)
    
    def _dfs(self, start, end, distance, remove_one_char, curr_path, res):
        if start == end:
            res.append(curr_path[:])
            return
        
        for word in self._get_next_word(start, remove_one_char):
            if distance[word] != distance[start] - 1:
                continue
            curr_path.append(word)
            self._dfs(word, end, distance, remove_one_char, curr_path, res)
            curr_path.pop()
        
    def _remove_one_char_mapping(self, word_set):
        remove_one_char = defaultdict(set)
        for word in word_set:
            for i in range(len(word)):
                new_pattern = word[:i] + '%' + word[i + 1:]
                remove_one_char[new_pattern].add(word)
        return remove_one_char
    
    def _get_next_word(self, word, remove_one_char):
        next_words = set()
        for i in range(len(word)):
            pattern = word[:i] + '%' + word[i + 1:]
            for next_word in remove_one_char[pattern]:
                next_words.add(next_word)
        return next_words
```

#### 190. 下一个排列
```
class Solution:
    """
    @param nums: An array of integers
    @return: nothing
    """
    def nextPermutation(self, nums):
        # write your code here
        if not nums or len(nums) <= 1:
            return
        
        if len(nums) == 2:
            nums[0], nums[1] = nums[0], nums[1]
            return
        
        n = len(nums)
        for pos1 in range(n - 2, -1, -1):
            if nums[pos1] < nums[pos1 + 1]:
                break
        else:
            start = 0
            end = n - 1
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
            return
        
        for pos2 in range(n - 1, pos1, -1):
            if nums[pos2] > nums[pos1]:
                break
        
        nums[pos1], nums[pos2] = nums[pos2], nums[pos1]
        start = pos1 + 1
        end = n - 1
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
```

#### 198. 排列序号II
```
from collections import defaultdict

class Solution:
    """
    @param A: An array of integers
    @return: A long integer
    """
    def permutationIndexII(self, A):
        # write your code here
        # 跟I那道题不同的地方就在于A里面可能包含了重复数字
        factor = 1
        multi_factor = 1
        mapping = defaultdict(int)
        res = 1
        
        for i in range(len(A) - 1, -1, -1):
            count = 0
            mapping[A[i]] += 1
            multi_factor *= mapping[A[i]]
            for j in range(i + 1, len(A)):
                if A[j] < A[i]:
                    count += 1
            res += factor * count // multi_factor
            factor *= len(A) - i
        
        return res
```

#### 197. 排列序号
```
class Solution:
    """
    @param A: An array of integers
    @return: A long integer
    """
    def permutationIndex(self, A):
        # write your code here
        # 核心就是计算每一位后面有多少个比它小的数字
        factor = 1
        res = 1
        
        for i in range(len(A) - 1, -1, -1):
            count = 0
            for j in range(i + 1, len(A)):
                if A[j] < A[i]:
                    count += 1
            res += factor * count
            factor *= len(A) - i
        
        return res
```

#### 52. 下一个排列
```
class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers
    """
    def nextPermutation(self, nums):
        # write your code here
        # 这道题和190几乎一模一样
        # 只不过190要求原地排序
        # 这道题要求返回新数组
        # 直接在原数组上修改再讲原数组返回即可
        if not nums or len(nums) <= 1:
            return nums
        
        if len(nums) == 2:
            nums[0], nums[1] = nums[1], nums[0]
            return nums
        
        n = len(nums)
        for pos1 in range(n - 2, -1, -1):
            if nums[pos1] < nums[pos1 + 1]:
                break
        else:
            start = 0
            end = n - 1
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
            return nums
        
        for pos2 in range(n - 1, pos1, -1):
            if nums[pos2] > nums[pos1]:
                break
        
        nums[pos1], nums[pos2] = nums[pos2], nums[pos1]
        start = pos1 + 1
        end = n - 1
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
        
        return nums
```

#### 51. 上一个排列
```
class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers that's previous permuation
    """
    def previousPermuation(self, nums):
        # write your code here
        pos1 = -1
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] > nums[i + 1]:
                pos1 = i
                break
        else:
            return nums[::-1]
        
        pos2 = -1
        for j in range(len(nums) - 1, pos1, -1):
            if nums[j] < nums[pos1]:
                pos2 = j
                break
        
        nums[pos1], nums[pos2] = nums[pos2], nums[pos1]
        return nums[:pos1 + 1] + nums[pos1 + 1:][::-1]
```

#### 634. 单词矩阵
```
from collections import defaultdict

class Solution:
    """
    @param: words: a set of words without duplicates
    @return: all word squares
    """
    def wordSquares(self, words):
        # write your code here
        if not words or not words[0]:
            return []
        
        n = len(words[0])
        mapping = defaultdict(set)
        
        for word in words:
            for i in range(n):
                mapping[word[:i]].add(word)
        
        res = []
        for word in words:
            self._dfs(word, 0, [], mapping, res)
        return res
    
    def _dfs(self, word, row, curr, mapping, res):
        curr.append(word)
        # 这里是假定words里所有的词的长度都应该是一样的
        if row == len(word) - 1:
            res.append(curr[:])
        else:
            # 核心之一
            # 下一次递归中的行是row + 1
            # 则要保证当前已经完成的矩形里
            # row + 1那一列上凑成的前缀应该是row + 1行开头的前缀
            prefix = ''.join(curr[i][row + 1] for i in range(row + 1))
            for next_word in mapping[prefix]:
                self._dfs(next_word, row + 1, curr, mapping, res)
        curr.pop()
```

#### 828. 字模式
```
class Solution:
    """
    @param pattern: a string, denote pattern string
    @param teststr: a string, denote matching string
    @return: an boolean, denote whether the pattern string and the matching string match or not
    """
    def wordPattern(self, pattern, teststr):
        # write your code here
        if not teststr:
            return not pattern

        words = teststr.split(' ')
        if len(pattern) != len(words):
            return False
        
        n = len(pattern)
        mapping_word = {}
        mapping_ch = {}
        for inx, word in enumerate(words):
            if word not in mapping_word:
                mapping_word[word] = pattern[inx]
            else:
                if mapping_word[word] != pattern[inx]:
                    return False
            
            if pattern[inx] not in mapping_ch:
                mapping_ch[pattern[inx]] = word
            else:
                if mapping_ch[pattern[inx]] != word:
                    return False
        
        return True
```

#### 211. 字符串置换
```
class Solution:
    """
    @param A: a string
    @param B: a string
    @return: a boolean
    """
    def Permutation(self, A, B):
        # write your code here
        mapping = [0] * 256
        
        for ch in A:
            mapping[ord(ch)] += 1
        
        for ch in B:
            mapping[ord(ch)] -= 1
            if mapping[ord(ch)] < 0:
                return False
        
        return sum(mapping) == 0
```

#### 123. 单词搜索
```
class Solution:
    """
    @param board: A list of lists of character
    @param word: A string
    @return: A boolean
    """

    _DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def exist(self, board, word):
        # write your code here
        if not board or not board[0]:
            return False
        
        if not word:
            return True
        
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        
        if m == 1 and n == 1 and len(word) == 1:
            return board[0][0] == word[0]
        
        for i in range(m):
            for j in range(n):
                if self._dfs(board, word, i, j, visited):
                    return True
        
        return False
    
    # dfs定义是在board上以当前的ci cj为起点，能不能找到word这个词
    def _dfs(self, board, word, ci, cj, visited):
        if not word:
            return True
        
        m, n = len(board), len(board[0])
        for di, dj in self._DIRS:
            newi, newj = di + ci, dj + cj
            if 0 <= newi < m and 0 <= newj < n and not visited[newi][newj] and board[newi][newj] == word[0]:
                visited[newi][newj] = True
                if self._dfs(board, word[1:], newi, newj, visited):
                    return True
                visited[newi][newj] = False
        
        return False
```

### 8 - Data Structure - Stack, Queue, Hash, Heap

#### 642. 数据流滑动窗口平均值
```
class MovingAverage:

    def __init__(self, size):
        self._presum = [0]
        self._size = size
        
    def next(self, val):
        self._presum.append(self._presum[-1] + val)
        if len(self._presum) - 1 < self._size:
            return self._presum[-1] / (len(self._presum) - 1)
        else:
            return (self._presum[-1] - self._presum[-self._size - 1]) / self._size
```

#### 494. 双队列实现栈
```
from collections import deque

class Stack:
    
    def __init__(self):
        # 实际上此时两个队列是平等的（跟两个队列实现栈那道题不一样）
        # 我们只需要保证始终有一个队列为空即可
        # 这里的queue1只是表示优先操作的队列
        # 实际上这道题用一个queue也能完成
        # 参见495实现栈那道题
        self._queue1 = deque()
        self._queue2 = deque()

    def push(self, x):
        self._queue1.append(x)
        
    def pop(self):
        # queue1里只保留一个元素
        while len(self._queue1) > 1:
            self._queue2.append(self._queue1.popleft())
        # 此时queue1位空
        # queue2不为空
        # 然后交换二者的引用
        res = self._queue1.popleft()
        self._queue1, self._queue2 = self._queue2, self._queue1
        return res
       
    def top(self):
        while len(self._queue1) > 1:
            self._queue2.append(self._queue1.popleft())
        res = self._queue1.popleft()
        self._queue1, self._queue2 = self._queue2, self._queue1
        self._queue1.append(res)
        return res

    def isEmpty(self):
        return len(self._queue1) == 0
```

#### 209. 第一个只出现一次的字符
```
from collections import defaultdict

class Solution:
    """
    @param string: str: the given string
    @return: char: the first unique character in a given string
    """
    def firstUniqChar(self, string):
        # Write your code here
        mapping = defaultdict(int)
        for ch in string:
            mapping[ch] += 1
        
        for inx, ch in enumerate(string):
            if mapping[ch] == 1:
                return ch
```

#### 657. Insert Delete GetRandom O(1)
```
from random import choice

class RandomizedSet:
    
    def __init__(self):
        # do intialization if necessary
        # 这道题说的是实现一个支持随机取东西的set
        # 全部的操作都是O(1)
        self._data = []
        self._mapping = {}
    """
    @param: val: a value to the set
    @return: true if the set did not already contain the specified element or false
    """
    def insert(self, val):
        # write your code here
        if val in self._mapping:
            return False
    
        self._data.append(val)
        self._mapping[val] = len(self._data) - 1
        return True

    """
    @param: val: a value from the set
    @return: true if the set contained the specified element or false
    """
    def remove(self, val):
        # write your code here
        if val not in self._mapping:
            return False
        
        if val == self._data[-1]:
            self._data.pop()
            del self._mapping[val]
            return True
        
        # 想要从array里O(1)删除一个目标数字怎么做？
        # 首先保存下最后一个数字
        # 再找到目标数字的index
        # 将对应的index上覆盖为temp
        # 再去修改temp对应的坐标为目标数字的index
        # 最后别忘了删除掉这个key
        temp = self._data.pop()
        val_inx = self._mapping[val]
        self._data[val_inx] = temp
        self._mapping[temp] = val_inx
        del self._mapping[val]
    """
    @return: Get a random element from the set
    """
    def getRandom(self):
        # write your code here
        return choice(self._data)

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param = obj.insert(val)
# param = obj.remove(val)
# param = obj.getRandom()
```

#### 612. K个最近的点
```
from heapq import heappop
from heapq import heappush

class Solution:
    """
    @param points: a list of points
    @param origin: a point
    @param k: An integer
    @return: the k closest points
    """
    def kClosest(self, points, origin, k):
        # write your code here
        # 好题！多多体会
        # 实际上generalize一下这道题:
        # 从一个数组中找到前k小元素（使用堆做）就是这种思路
        # 因为python是最小堆 这里应该用一个最大堆(堆的大小始终维护为k)
        # 在脑海中过这道题的时候，用这个例子：
        # [1, 2, 3, 4, 5]找到前2小的数字使用堆做
        # 很明显默认的最小堆是不行的
        # 因为我们是先入堆 再出堆
        # 这样当3入堆以后 1会被pop出来
        # 所以应该最大堆
        # 这样当3入堆以后 在紧跟着的pop操作中会被马上pop出来!!!
        hp = []
        ox, oy = origin.x, origin.y
        
        for point in points:
            px, py = point.x, point.y
            dist = (px - ox) ** 2 + (py - oy) ** 2
            # python是最小堆
            # 但是这道题要求的是前k个最接近（最小值）的堆
            # 应该用最大堆
            # 这样会不停的pop堆顶(which is 当前距离最大的，保留当前距离最小的)
            heappush(hp, (-dist, -px, -py))
            if len(hp) > k:
                heappop(hp)

        # 原来的hp是没有序的
        hp.sort(reverse=True)
        return [Point(-i[1], -i[2]) for i in hp]
```

#### 544. 前K大数
```
class Solution:
    """
    @param: nums: an integer array
    @param: k: An integer
    @return: the top k largest numbers in array
    """
    def topk(self, nums, k):
        # write your code here
        if k == 0 or k > len(nums) or not nums:
            return -2 ** 31
        
        l, r = 0, len(nums) - 1
        while True:
            p = self._partition(nums, l, r)
            if p == k - 1:
                # 这道题下面的写法过不了OJ，不应该
                # return nums[:p + 1]
                return sorted(nums[:p + 1], reverse=True)
            elif p > k - 1:
                r = p - 1
            else:
                l = p + 1
    
    def _partition(self, nums, l, r):
        pivot_value = nums[l]
        j = l
        for i in range(l + 1, r + 1):
            if nums[i] > pivot_value:
                j += 1
                nums[i], nums[j] = nums[j], nums[i]
        nums[j], nums[l] = nums[l], nums[j]
        return j
```

#### 104. 合并k个排序链表
```
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        # write your code here
        if not lists:
            return
        
        if len(lists) == 1:
            return lists[0]
        
        if len(lists) == 2:
            return self._merge(lists[0], lists[1])
        
        n = len(lists)
        half1 = self.mergeKLists(lists[:n // 2])
        half2 = self.mergeKLists(lists[n // 2:])
        return self._merge(half1, half2)
    
    def _merge(self, l1, l2):
        dummy_node = ListNode(-1)
        curr = dummy_node
        
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        if l1:
            curr.next = l1
        if l2:
            curr.next = l2
        
        return dummy_node.next
```

#### 40. 用栈实现队列
```
class MyQueue:
    # 思路很简单：
    # 用两个栈，一个只放数据，一个负责出数据
    # 要注意的是两点：
    # 1是如果stack_push要倒数据到stack_pop中，必须一次性全部压入
    # 2是如果stack_pop不为空，stack_push是不能往里压数据的
    def __init__(self):
        # do intialization if necessary
        # 这个栈只负责添加数据
        self._stack_push = []
        # 这个栈只负责pop数据
        self._stack_pop = []

    def push(self, element):
        # write your code here
        self._stack_push.append(element)

    def pop(self):
        # write your code here
        self._adjust()
        return self._stack_pop.pop()

    def top(self):
        # write your code here
        self._adjust()
        return self._stack_pop[-1]
    
    def _adjust(self):
        # 只有当stack_pop为空的时候才能调整！！！
        if not self._stack_pop:
            while self._stack_push:
                self._stack_pop.append(self._stack_push.pop())
```

#### 4. 丑数 II
```
from heapq import heappop
from heapq import heappush

class Solution:
    """
    @param n: An integer
    @return: return a  integer as description.
    """
    def nthUglyNumber(self, n):
        # write your code here
        # 这道题用heap感觉好理解，并且面试时候也容易解释思路
        # 基本思路就是每个数字都只跟2 3 5乘，分别生成新数字
        # 然后每个新数字再分别和2 3 5乘
        # 只要没出现过 就加入到堆中
        # 这里是一个最小堆
        # 这样就n次循环（每次可能push多词）最后一次pop出来的
        # 就是第n小的丑数了
        hp = [1]
        visited = set([1])
        
        res = -2 ** 31 - 1
        for _ in range(n):
            res = heappop(hp)
            for factor in (2, 3, 5):
                if res * factor not in visited:
                    heappush(hp, res * factor)
                    visited.add(res * factor)
        
        return res
```

#### 134. LRU缓存策略
```
from collections import OrderedDict

class LRUCache:
    """
    @param: capacity: An integer
    """
    def __init__(self, capacity):
        self._capacity = capacity
        self._cache = OrderedDict()
        
    def get(self, key):
        if key in self._cache:
            val = self._cache.pop(key)
            self._cache[key] = val
            return val
        else:
            return -1

    def set(self, key, value):
        if key in self._cache:
            self._cache.pop(key)
        else:
            # 此时要往cache里加数据了
            # 要检查下是不是满了
            if len(self._cache) == self._capacity:
                self._cache.popitem(last=False)
        self._cache[key] = value
```

#### 495. 实现栈
```
from collections import deque

class Stack:
    def __init__(self):
        self._queue = deque()

    def push(self, x):
        self._queue.append(x)

    def pop(self):
        for _ in range(len(self._queue) - 1):
            self._queue.append(self._queue.popleft())
        return self._queue.popleft()
        
    def top(self):
        top = None
        for _ in range(len(self._queue)):
            top = self._queue.popleft()
            self._queue.append(top)
        return top
        
    def isEmpty(self):
        return len(self._queue) == 0
```

#### 128. 哈希函数
```
class Solution:
    """
    @param key: A string you should hash
    @param HASH_SIZE: An integer
    @return: An integer
    """
    def hashCode(self, key, HASH_SIZE):
        # write your code here
        res = 0
        for ch in key:
            res = (res * 33 + ord(ch)) % HASH_SIZE
        return res
```

#### 685. 数据流中第一个唯一的数字
```
from collections import OrderedDict

class Solution:
    """
    @param nums: a continuous stream of numbers
    @param number: a number
    @return: returns the first unique number
    """
    def firstUniqueNumber(self, nums, number):
        # Write your code here
        unique_dict = OrderedDict()
        dup_set = set()
        
        for num in nums:
            # 说明要么这个num第一次出现；要么之前只出现过一次
            if num not in dup_set:
                if num not in unique_dict:
                    # 如果这个数字既不在dup里，又不在unique里
                    # 就说明这个数字第一次出现
                    # 在unique里给这个数字赋个占位符
                    unique_dict[num] = None
                else:
                    # 此时说明这个数字之前已经出现过一次了
                    # 此时一定是第二次出现
                    dup_set.add(num)
                    del unique_dict[num]
            
            # 说明出现终止数字了，此时需要返回
            if num == number:
                # 如果此时unique_dict为空
                # 说明从来没有出现过unique数字 直接返回-1
                if not unique_dict:
                    return -1
                return list(unique_dict.items())[0][0]
        
        # 说明number从来没有在num中出现过
        # 直接返回-1
        return -1
```

#### 613. 优秀成绩
```
from collections import defaultdict
from heapq import heappop
from heapq import heappush

class Solution:
    # @param {Record[]} results a list of <student_id, score>
    # @return {dict(id, average)} find the average of 5 highest scores for each person
    # <key, value> (student_id, average_score)
    def highFive(self, results):
        # 核心：就是利用最小堆实现top k large
        # 最小堆（python默认）实现top k large
        # 最大堆实现top k small
        student_scores = defaultdict(list)
        for each in results:
            _id, score = each.id, each.score
            heappush(student_scores[_id], score)
            if len(student_scores[_id]) > 5:
                heappop(student_scores[_id])
        
        res = dict()
        for _id, scores in student_scores.items():
            res[_id] = sum(scores) / len(scores)
        return res
```

#### 606. 第K大的元素 II
```
from heapq import heappop
from heapq import heappush

class Solution:
    """
    @param nums: an integer unsorted array
    @param k: an integer from 1 to n
    @return: the kth largest element
    """
    def kthLargestElement2(self, nums, k):
        # write your code here
        # 找到数组中第K大的元素，N远大于K
        # 这种海量数据用堆 不要用quick select
        # 因为内存小，只需要保留k个数据即可
        hp = []
        for num in nums:
            heappush(hp, num)
            if len(hp) > k:
                heappop(hp)
        
        return hp[0]
```

#### 601. 摊平二维向量
```
class Vector2D(object):

    # @param vec2d {List[List[int]]}
    def __init__(self, vec2d):
        # Initialize your data structure here
        self._data = [i[:] for i in vec2d]
        self._row = 0
        self._col = 0
        self._max_row = len(vec2d)

    # @return {int} a next element
    def next(self):
        # Write your code here
        res = self._data[self._row][self._col]
        self._col += 1
        return res

    # @return {boolean} true if it has next element
    # or false
    def hasNext(self):
        # Write your code here
        while self._row < self._max_row and \
            self._col == len(self._data[self._row]):
            self._row += 1
            self._col = 0
        return self._row < self._max_row
```

#### 545. 前K大数 II
```
from heapq import heappop
from heapq import heappush

class Solution:
    """
    @param: k: An integer
    """
    def __init__(self, k):
        # do intialization if necessary
        self._data = []
        self._k = k
    """
    @param: num: Number to be added
    @return: nothing
    """
    def add(self, num):
        # write your code here
        heappush(self._data, num)
        if len(self._data) > self._k:
            heappop(self._data)
    """
    @return: Top k element
    """
    def topk(self):
        # write your code here
        return sorted(self._data, reverse=True)
```

#### 526. 负载均衡器
```
from random import randint
from random import choice

class LoadBalancer:
    def __init__(self):
        self._servers = []
        
    def add(self, server_id):
        num_servers = len(self._servers)
        # 注意python randint函数randint(a, b)
        # 是产生在[a, b]左闭右闭区间内的一个随机数
        random_pos = randint(0, num_servers)
        self._servers.insert(random_pos, server_id)
        
    def remove(self, server_id):
        if server_id in self._servers:
            self._servers.remove(server_id)
        
    def pick(self):
        return choice(self._servers)
```

#### 486. 合并k个排序数组
```
from heapq import heappop
from heapq import heappush

class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        # write your code here
        # 好题，多多体会
        hp = []
        for i in range(len(arrays)):
            # 这里是因为有可能arrays里有空list
            if arrays[i]:
                heappush(hp, (arrays[i][0], i, 0))
        
        res = []
        while hp:
            val, i, j = heappop(hp)
            res.append(val)
            if j + 1 < len(arrays[i]):
                heappush(hp, (arrays[i][j + 1], i, j + 1))
        
        return res
```

#### 130. 堆化
```
class Solution:
    """
    @param: A: Given an integer array
    @return: nothing
    """
    def heapify(self, A):
        # 堆的基础操作！好题目！
        # 基本思路就是从后往前
        # 找出A的最后一个非叶子节点然后往前sift_down操作
        # 那么如何找A的最后一个非叶子节点？
        # 其实就是最后一个节点的父亲节点
        # 最后一个节点下标是len(A) - 1
        # 则它的父亲节点就是(len(A) - 1) // 2
        # 反过来，一个节点k的两个孩子节点下标就是：
        # k * 2 + 1和k * 2 + 2
        for i in range((len(A) - 1) // 2, -1, -1):
            self._sift_down(A, i)
    
    def _sift_down(self, A, k):
        while k * 2 + 1 < len(A):
            son1_inx = k * 2 + 1
            son2_inx = k * 2 + 2

            # 最小堆
            # 就是找当前k的俩儿子
            # 如果k对应的值比最小的儿子还小
            # 说明不需要把k再下沉下去了 直接break
            smaller_son_inx = son1_inx
            if son2_inx < len(A) and A[son2_inx] < A[son1_inx]:
                smaller_son_inx = son2_inx
            
            # 这里要做最小堆
            # 如果小儿子的值都还比父亲k小
            # 说明不需要再sift_down了，直接break即可
            if A[k] < A[smaller_son_inx]:
                break
            
            A[k], A[smaller_son_inx] = A[smaller_son_inx], A[k]
            k = smaller_son_inx
```

#### 129. 重哈希
```
class Solution:
    """
    @param hash_table: A list of The first node of linked list
    @return: A list of The first node of linked list which have twice size
    """
    def rehashing(self, hash_table):
        # write your code here
        new_hash_table = [None] * (2 * len(hash_table))
        for curr in hash_table:
            while curr:
                self._add_node(new_hash_table, curr.val)
                curr = curr.next
        return new_hash_table
    
    def _add_node(self, hash_table, val):
        inx = val % len(hash_table)
        if hash_table[inx] is None:
            hash_table[inx] = ListNode(val)
        else:
            curr = hash_table[inx]
            while curr.next:
                curr = curr.next
            curr.next = ListNode(val)
```

#### 124. 最长连续序列
```
class Solution:
    """
    @param nums: A list of integers
    @return: An integer
    """
    def longestConsecutive(self, nums):
        # write your code here
        if not nums:
            return 0
        
        nums_set = set(nums)
        res = 0
        for num in nums:
            if num not in nums_set:
                continue
            nums_set.remove(num)

            asc = num + 1
            while asc in nums_set:
                # 优化的核心之一
                nums_set.remove(asc)
                asc += 1
            
            desc = num - 1
            while desc in nums_set:
                # 优化的核心之一
                nums_set.remove(desc)
                desc -= 1
            
            res = max(res, asc - desc - 1)
            
            
        return res
```

#### 551. 嵌套列表的加权和
```
class Solution(object):
    # @param {NestedInteger[]} nestedList a list of NestedInteger Object
    # @return {int} an integer
    def depthSum(self, nestedList):
        # Write your code here
        # 这道题的weight是外层的最小
        # 越嵌套weight越大
        if not nestedList:
            return 0
        
        stack = []
        for each in nestedList:
            stack.append((each, 1))
        
        res = 0
        while stack:
            curr, level = stack.pop()
            if curr.isInteger():
                res += curr.getInteger() * level
            else:
                for each in curr.getList():
                    stack.append((each, level + 1))
        
        return res
```

#### 494. 双队列实现栈
```
from collections import deque

class Stack:
    
    def __init__(self):
        # 实际上此时两个队列是平等的（跟两个队列实现栈那道题不一样）
        # 我们只需要保证始终有一个队列为空即可
        # 这里的queue1只是表示优先操作的队列
        # 实际上这道题用一个queue也能完成
        # 参见495实现栈那道题
        self._queue1 = deque()
        self._queue2 = deque()

    def push(self, x):
        self._queue1.append(x)
        
    def pop(self):
        # queue1里只保留一个元素
        while len(self._queue1) > 1:
            self._queue2.append(self._queue1.popleft())
        # 此时queue1位空
        # queue2不为空
        # 然后交换二者的引用
        res = self._queue1.popleft()
        self._queue1, self._queue2 = self._queue2, self._queue1
        return res
       
    def top(self):
        while len(self._queue1) > 1:
            self._queue2.append(self._queue1.popleft())
        res = self._queue1.popleft()
        self._queue1, self._queue2 = self._queue2, self._queue1
        self._queue1.append(res)
        return res

    def isEmpty(self):
        return len(self._queue1) == 0
```

#### 575. 字符串解码
```
class Solution:
    """
    @param s: an expression includes numbers, letters and brackets
    @return: a string
    """
    def expressionExpand(self, s):
        # write your code here
        stack = []
        num = 0
        for ch in s:
            if ch.isdigit():
                num = 10 * num + int(ch)
            elif ch == '[':
                stack.append(num)
                num = 0
            elif ch == ']':
                temp_string = ''
                # 遇到]就开始出栈
                # 最先遇到的肯定是内层的]
                while stack:
                    curr = stack.pop()
                    if type(curr) is int:
                        stack.append(temp_string * curr)
                        break
                    temp_string = curr + temp_string
            else:
                stack.append(ch)
        
        return ''.join(stack)
```

#### 541. 左旋右旋迭代器 II
```
from collections import deque

class ZigzagIterator2:
    """
    @param: vecs: a list of 1d vectors
    """
    def __init__(self, vecs):
        self._queue = deque(deque(v) for v in vecs if v)

    def next(self):
        curr_v = self._queue.popleft()
        res = curr_v.popleft()
        if curr_v:
            self._queue.append(curr_v)
        return res

    def hasNext(self):
        return bool(self._queue)
```

#### 540. 左旋右旋迭代器
```
from collections import deque

class ZigzagIterator:
    """
    @param: v1: A 1d vector
    @param: v2: A 1d vector
    """
    def __init__(self, v1, v2):
        self._queue = deque(deque(v) for v in (v1, v2) if v)
        
    def next(self):
        curr_v = self._queue.popleft()
        res = curr_v.popleft()
        if curr_v:
            self._queue.append(curr_v)
        return res
        
    def hasNext(self):
        return bool(self._queue)
```

#### 528. 摊平嵌套的列表
```
class NestedIterator:

    def __init__(self, nestedList):
        self._stack = nestedList
        
    def next(self):
        return self._stack.pop().getInteger()
        
    def hasNext(self):
        while self._stack:
            if self._stack[-1].isInteger():
                return True
            curr = self._stack.pop()
            # 由于栈后入先出
            # 如果要保持pop出来的顺序
            # 就需要反向append
            # 实际上这道题用deque更适合
            for each in curr.getList()[::-1]:
                self._stack.append(each)
        return False
```

#### 471. 最高频的K个单词
```
from collections import defaultdict
from functools import cmp_to_key

class Solution:
    """
    @param words: an array of string
    @param k: An integer
    @return: an array of string
    """
    def topKFrequentWords(self, words, k):
        # write your code here
        # 注意大坑！！！
        # 这道题标准的最小堆解法python不好做！！！
        # 除非自己定义数据结构
        # 很坑很坑（尤其是要求单词字典序）
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1
        
        temp = []
        for word, counts in word_count.items():
            temp.append((counts, word))
        temp.sort(key=cmp_to_key(self._cmp))
        
        res = []
        for _, word in temp[:k]:
            res.append(word)
        return res
    
    def _cmp(self, a, b):
        if a[0] > b[0] or (a[0] == b[0] and a[1] < b[1]):
            return -1
        elif a[0] == b[0] and a[1] == b[1]:
            return 0
        else:
            return 1
```

#### 224. 用一个数组实现三个栈
```
class ThreeStacks:

    def __init__(self, size):
        self._stack_pointer = [
            [0, -1],
            [size, -1],
            [size * 2, -1],
        ]
        self._stack = [0] * size * 3

    def push(self, stackNum, value):
        self._stack_pointer[stackNum][1] += 1
        start_inx, off_set = self._stack_pointer[stackNum]
        self._stack[start_inx + off_set] = value
        
    def pop(self, stackNum):
        start_inx, off_set = self._stack_pointer[stackNum]
        res = self._stack[start_inx + off_set]
        self._stack_pointer[stackNum][1] -= 1
        return res
        
    def peek(self, stackNum):
        start_inx, off_set = self._stack_pointer[stackNum]
        return self._stack[start_inx + off_set]
        
    def isEmpty(self, stackNum):
        return self._stack_pointer[stackNum][1] == -1
```

#### 24. LFU缓存
```
# 没看懂-_-，有空的话再回头看吧
class LFUCache:
    """
    @param: capacity: An integer
    """
    def __init__(self, capacity):
        self.cache = {}
        self.capacity = capacity
        self.D = FreqNode(-1)
        self.d = FreqNode(-1)
        self.D.nxt, self.d.pre = self.d, self.D

    """
    @param: key: An integer
    @param: val: An integer
    @return: nothing
    """
    def set(self, key, val):
        if self.capacity <= 0:
            return

        if key in self.cache:
            self._update_item(key, val)
            return

        if len(self.cache) >= self.capacity:
            self._evict_item()

        self._add_item(key, val)

    """
    @param: key: An integer
    @return: An integer
    """
    def get(self, key):
        if key not in self.cache:
            return -1

        self._update_item(key)
        return self.cache[key].val

    def _add_item(self, key, val):
        cache_node = CacheNode(key, val)
        self.cache[key] = cache_node

        freq_head = self.D.nxt
        if freq_head and freq_head.freq == 0:
            freq_head.append_tail(cache_node)
            return

        freq_head = FreqNode(0)
        freq_head.append_tail(cache_node)
        self.D.after(freq_head)

    def _evict_item(self):
        freq_head = self.D.nxt
        cache_node = freq_head.pop_head()
        self.cache.pop(cache_node.key)

        if freq_head.is_empty():
            freq_head.unlink()

    def _update_item(self, key, val=None):
        cache_node = self.cache[key]

        if val:
            cache_node.val = val

        from_freq = cache_node.freq_node
        to_freq = None

        if from_freq.nxt and from_freq.nxt.freq == from_freq.freq + 1:
            to_freq = from_freq.nxt
        else:
            to_freq = FreqNode(from_freq.freq + 1)
            from_freq.after(to_freq)

        cache_node.unlink()
        to_freq.append_tail(cache_node)

        if from_freq.is_empty():
            from_freq.unlink()

class CacheNode:
    def __init__(self, key, val=None, freq_node=None, pre=None, nxt=None):
        self.key = key
        self.val = val
        self.freq_node = freq_node
        self.pre = pre
        self.nxt = nxt

    # to change self in cache nodes
    def unlink(self):
        self.pre.nxt = self.nxt
        self.nxt.pre = self.pre

        self.pre = self.nxt = self.freq_node = None


class FreqNode:
    def __init__(self, freq, pre=None, nxt=None):
        self.freq = freq
        self.pre = pre
        self.nxt = nxt
        self.D = CacheNode(-1)
        self.d = CacheNode(-1)
        self.D.nxt, self.d.pre = self.d, self.D

    # to change self in freq nodes
    def unlink(self):
        self.pre.nxt = self.nxt
        self.nxt.pre = self.pre

        self.pre = self.nxt = self.D = self.d = None

    # to change self in freq nodes
    def after(self, node):
        node.pre = self
        node.nxt = self.nxt
        self.nxt.pre = node
        self.nxt = node

    # to manage cache nodes
    def is_empty(self):
        return self.D.nxt is self.d

    # to manage cache nodes
    def pop_head(self):
        if self.is_empty():
            return

        head = self.D.nxt
        head.unlink()
        return head

    # to manage cache nodes
    def append_tail(self, node):
        node.freq_node = self
        node.pre = self.d.pre
        node.nxt = self.d
        self.d.pre.nxt = node
        self.d.pre = node
```

### 9 - Data Structure - Interval, Array, Matrix & Binary Indexed Tree

#### 839. 合并两个排序的间隔列表
```
class Solution:
    """
    @param list1: one of the given list
    @param list2: another list
    @return: the new sorted list of interval
    """
    def mergeTwoInterval(self, list1, list2):
        # write your code here
        p1 = p2 = 0
        n1, n2 = len(list1), len(list2)
        last_unused = curr = None
        res = []
        
        while p1 < n1 and p2 < n2:
            if list1[p1].start < list2[p2].start:
                curr = list1[p1]
                p1 += 1
            else:
                curr = list2[p2]
                p2 += 1
            # 本题核心之一：出现gap再去merge
            # 也是处理此类问题的一个通用思路
            last_unused = self._merge(res, last_unused, curr)
        
        while p1 < n1:
            last_unused = self._merge(res, last_unused, list1[p1])
            p1 += 1
        
        while p2 < n2:
            last_unused = self._merge(res, last_unused, list2[p2])
            p2 += 1
        
        if last_unused:
            res.append(last_unused)
        
        return res
    
    def _merge(self, res, last_unused, curr):
        if not last_unused:
            return curr
        
        # 此时说明curr和last_unused之间不重叠
        # 出现了gap
        # 可以直接将last_unused加入到res中，完成其使命
        # 并将curr当成新的last_unused返回
        if curr.start > last_unused.end:
            res.append(last_unused)
            return curr
        else:
            # 此时表明出现了重叠
            # 不急着merge
            # 等到下一个间隔出来再去merge
            # 因为很可能下一个curr还是会和(本次合并之后的)last_unused重叠
            last_unused.end = max(last_unused.end, curr.end)
            return last_unused
```

#### 547. 两数组的交集
```
class Solution:
    def intersection(self, nums1, nums2):
        # write your code here
        return list(set(nums1) & set(nums2))
```

#### 138. 子数组之和
```
class Solution:
    """
    @param nums: A list of integers
    @return: A list of integers includes the index of the first number and the index of the last number
    """
    def subarraySum(self, nums):
        # write your code here
        # 这道题直接用presum数组会TLE
        # 很好的思路！使用hash来存
        # 注意返回值[presum_hash[presum] + 1, inx]
        presum_hash = {0: -1}
        presum = 0
        for inx, num in enumerate(nums):
            presum += num
            if presum in presum_hash:
                return [presum_hash[presum] + 1, inx]
            presum_hash[presum] = inx
        
        return [-1, -1]
```

#### 64. 合并排序数组
```
class Solution:
    """
    @param: A: sorted integer array A which has m elements, but size of A is m+n
    @param: m: An integer
    @param: B: sorted integer array B which has n elements
    @param: n: An integer
    @return: nothing
    """
    def mergeSortedArray(self, A, m, B, n):
        # write your code here
        # 这道题是说将A和B合并到A中
        # A和B已经是有序的
        # 注意这里的m和n是A和B的有效索引
        pa, pb = m - 1, n - 1
        while pa >= 0 and pb >= 0:
            if A[pa] > B[pb]:
                # 核心之一：注意这里的索引是pa + pb + 1
                # 要加1的
                # 比方说A[0] = 1, B[0] = 2
                # 最后应该在A[1]的位置放置2（0 + 0 + 1 = 1）
                A[pa + pb + 1] = A[pa]
                pa -= 1
            else:
                A[pa + pb + 1] = B[pb]
                pb -= 1
        
        while pa >= 0:
            A[pa + pb + 1] = A[pa]
            pa -= 1
        
        while pb >= 0:
            A[pa + pb + 1] = B[pb]
            pb -= 1
```

#### 41. 最大子数组
```
class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        # 这道题里的nums是有正有负的
        if not nums:
            return -2 ** 31
        
        local_max = global_max = -2 ** 31
        for num in nums:
            # nums中遍历的顺序
            # 要么先正后负
            # 要么先负后正
            local_max = max(local_max + num, num)
            global_max = max(global_max, local_max)
        
        return global_max
```

#### 944. 最大子矩阵
```
class Solution:
    """
    @param matrix: the given matrix
    @return: the largest possible sum
    """
    def maxSubmatrix(self, matrix):
        # write your code here
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        presum = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 这里的presum就是从原点到右下角的矩阵和
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                presum[i][j] = presum[i - 1][j] + matrix[i - 1][j - 1]
        
        res = -2 ** 31 - 1
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                curr = 0
                for k in range(1, n + 1):
                    curr += presum[j][k] - presum[i][k]
                    res = max(res, curr)
                    curr = max(curr, 0)
        
        return res
```

#### 931. K 个有序数组的中位数
```
from bisect import bisect_right

class Solution:
    """
    @param nums: the given k sorted arrays
    @return: the median of the given k sorted arrays
    """
    def findMedian(self, nums):
        # write your code here
        # 这道题lintcode上merge sort会超时
        # 还是二分法
        start, end = 2 ** 31 - 1, -2 ** 31
        
        # 先求一下nums中的最大值和最小值
        counts = 0
        for inx, each in enumerate(nums):
            if not each:
                continue
            start = min(start, each[0])
            end = max(end, each[-1])
            counts += len(each)
        
        if counts == 0:
            return 0.0
        
        mid = counts // 2
        if counts % 2 == 0:
            return (self._find_kth(nums, start, end, mid) + \
                self._find_kth(nums, start, end, mid + 1)) / 2.0
        else:
            return self._find_kth(nums, start, end, mid + 1) / 1.0
    
    def _find_kth(self, nums, start, end, k):
        while start + 1 < end:
            mid = start + (end - start) // 2
            # 如果当前nums中比mid小的数字多于或者等于k个
            # 说明我们这时的mid大了，但是上限还可能是mid的
            if self._count_less_or_eq_to(nums, mid) >= k:
                end = mid
            # 反之，如果此时nums中比mid小的数字小于k个了
            # 说明此时的start到mid之间肯定不包含解
            else:
                start = mid + 1
        
        # 最终start和end中一定有一个数字是第k大的
        # 分别测试即可
        if self._count_less_or_eq_to(nums, start) == k:
            return start
        return end
    
    def _count_less_or_eq_to(self, nums, target):
        res = 0
        for i in range(len(nums)):
            # 很好的思路！使用bisect_right找到小于等于target的数字的个数
            res += bisect_right(nums[i], target)
        return res
```

#### 840. 可变范围求和
```
class BinaryIndexTree:
    def __init__(self, nums):
        n = len(nums)
        self._nums = [0] * (n + 1)
        self._presum = [0] * (n + 1)
        for inx, val in enumerate(nums):
            self.setter(inx + 1, val)
    
    def _low_bit(self, x):
        return x & -x
    
    def setter(self, inx, new_val):
        diff = new_val - self._nums[inx]
        self._nums[inx] = new_val
        while inx < len(self._presum):
            self._presum[inx] += diff
            # 核心之一：注意这里是+=
            inx += self._low_bit(inx)
    
    def getter(self, inx):
        res = 0
        while inx > 0:
            res += self._presum[inx]
            # 核心之一：注意这里是-=
            inx -= self._low_bit(inx)
        return res

class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self._bit = BinaryIndexTree(nums)

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        self._bit.setter(i + 1, val)

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self._bit.getter(j + 1) - self._bit.getter(i)
```

#### 654. 稀疏矩阵乘法
```
class Solution:
    """
    @param A: a sparse matrix
    @param B: a sparse matrix
    @return: the result of A * B
    """
    def multiply(self, A, B):
        # write your code here
        row = len(A)
        col = len(B[0])
        common = len(A[0])
        
        mat = [[0] * col for _ in range(row)]
        for i in range(row):
            for j in range(common):
                if A[i][j] == 0:
                    continue
                for k in range(col):
                    if B[j][k] == 0:
                        continue
                    mat[i][k] += A[i][j] * B[j][k]
        
        return mat
```

#### 577. 合并K个排序间隔列表
```
from heapq import heappush
from heapq import heappop

class Solution:
    """
    @param intervals: the given k sorted interval lists
    @return:  the new sorted interval list
    """
    # lintcode py3不给通过:(
    def mergeKSortedIntervalLists(self, intervals):
        # write your code here
        if not intervals:
            return []
        
        hp = []
        for each in intervals:
            if not each:
                continue
            for i in range(len(each)):
                heappush(hp, (each[i].start, each[i]))
        
        temp = []
        while hp:
            start, interval = heappop(hp)
            temp.append(interval)
        
        res = [temp[0]]
        for i in range(1,len(temp)):
            # 说明此时出现间隔了
            # 不需要merge 可以直接往res里放结果
            if res[-1].end < temp[i].start:
                res.append(temp[i])
            else:
                # 此时为什么就是merge？
                # 相当于跳过处理了temp[i]
                # 而通过直接修改res[-1]的方式来做到了merge！！！
                # 很重要的思路！！！
                res[-1].end = max(res[-1].end, temp[i].end)
        
        return res
```

#### 486. 合并k个排序数组
```
from heapq import heappop
from heapq import heappush

class Solution:
    """
    @param arrays: k sorted integer arrays
    @return: a sorted array
    """
    def mergekSortedArrays(self, arrays):
        # write your code here
        # 好题，多多体会
        hp = []
        for i in range(len(arrays)):
            # 这里是因为有可能arrays里有空list
            if arrays[i]:
                heappush(hp, (arrays[i][0], i, 0))
        
        res = []
        while hp:
            val, i, j = heappop(hp)
            res.append(val)
            if j + 1 < len(arrays[i]):
                heappush(hp, (arrays[i][j + 1], i, j + 1))
        
        return res
```

#### 65. 两个排序数组的中位数
```
class Solution:
    """
    @param: A: An integer array
    @param: B: An integer array
    @return: a double whose format is *.5 or *.0
    """
    def findMedianSortedArrays(self, A, B):
        # write your code here
        n = len(A) + len(B)
        
        if n % 2 == 0:
            # 注意kth 是bast-1索引
            smaller = self._find_kth_largest(A, B, n // 2)
            larger = self._find_kth_largest(A, B, n // 2 + 1)
            return (smaller + larger) / 2
        else:
            return self._find_kth_largest(A, B, n // 2 + 1)
        
    def _find_kth_largest(self, A, B, k):
        if not A:
            return B[k - 1]
        if not B:
            return A[k - 1]
        
        if k == 1:
            return min(A[0], B[0])
        
        a = A[k // 2 - 1] if len(A) >= k // 2 else 2 ** 31 - 1
        b = B[k // 2 - 1] if len(B) >= k // 2 else 2 ** 31 - 1
        
        if a < b:
            return self._find_kth_largest(A[k // 2:], B, k - k // 2)
        else:
            return self._find_kth_largest(A, B[k // 2:], k - k // 2)
```

#### 943. 区间和查询 - Immutable
```
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self._presum = [0]
        for num in nums:
            self._presum.append(self._presum[-1] + num)

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self._presum[j + 1] - self._presum[i]
```

#### 165. 合并两个排序链表
```
class Solution:
    """
    @param l1: ListNode l1 is the head of the linked list
    @param l2: ListNode l2 is the head of the linked list
    @return: ListNode head of linked list
    """
    def mergeTwoLists(self, l1, l2):
        # write your code here
        dummy_node = ListNode(-1)
        curr = dummy_node
        
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        if l1:
            curr.next = l1
        elif l2:
            curr.next = l2
        
        return dummy_node.next
```

#### 6. 合并排序数组 II
```
class Solution:
    """
    @param A: sorted integer array A
    @param B: sorted integer array B
    @return: A new sorted integer array
    """
    def mergeSortedArray(self, A, B):
        # write your code here
        # 这道题的followup是如果一个特别大一个特别小如何优化
        # 思路应该是可以用二分在大中找可以插入的位置
        # 然后用O(1)的方式将值插入
        i = j = 0
        len_A, len_B = len(A), len(B)
        
        res = []
        while i < len_A and j < len_B:
            if A[i] < B[j]:
                res.append(A[i])
                i += 1
            else:
                res.append(B[j])
                j += 1
        
        while i < len_A:
            res.append(A[i])
            i += 1
        
        while j < len_B:
            res.append(B[j])
            j += 1
        
        return res
```

#### 817. 范围矩阵元素和-可变的
```
class BinaryIndexTree:
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        m, n = len(matrix), len(matrix[0])
        self._mat = [[0] * (n + 1) for _ in range(m + 1)]
        self._presum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                self.setter(i + 1, j + 1, matrix[i][j])
    
    def _lowbit(self, x):
        return x & -x
    
    def setter(self, i, j, new_val):
        diff = new_val - self._mat[i][j]
        self._mat[i][j] = new_val
        mm, nn = len(self._mat), len(self._mat[0])
        ci = i
        while ci < mm:
            cj = j
            while cj < nn:
                self._presum[ci][cj] += diff
                cj += self._lowbit(cj)
            ci += self._lowbit(ci)
    
    def getter(self, i, j):
        res = 0
        ci = i
        while ci > 0:
            cj = j
            while cj > 0:
                res += self._presum[ci][cj]
                cj -= self._lowbit(cj)
            ci -= self._lowbit(ci)
        return res

class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        self.bit = BinaryIndexTree(matrix)

    def update(self, row, col, val):
        """
        :type row: int
        :type col: int
        :type val: int
        :rtype: void
        """
        self.bit.setter(row + 1, col + 1, val)

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        # 思路还是树状数组
        # 两层while循环
        # 分别用lowbit定位
        return self.bit.getter(row2 + 1, col2 + 1) \
            - self.bit.getter(row2 + 1, col1) \
            - self.bit.getter(row1, col2 + 1) \
            + self.bit.getter(row1, col1)
```

#### 793. 多个数组的交集
```
from collections import defaultdict

class Solution:
    """
    @param arrs: the arrays
    @return: the number of the intersection of the arrays
    """
    def intersectionOfArrays(self, arrs):
        # write your code here
        mapping = defaultdict(int)
        for arr in arrs:
            for each in arr:
                mapping[each] += 1
        
        res = 0
        for num, count in mapping.items():
            # 注意这里的要求是每个数组里的元素必须没有重复
            if count == len(arrs):
                res += 1
        
        return res
```

#### 665. 平面范围求和 -不可变矩阵
```
class NumMatrix:
    """
    @param: matrix: a 2D matrix
    """
    def __init__(self, matrix):
        if not matrix or not matrix[0]:
            return
        
        m, n = len(matrix), len(matrix[0])
        self._presum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self._presum[i][j] = self._presum[i][j - 1] \
                    + self._presum[i - 1][j] \
                    - self._presum[i - 1][j - 1] \
                    + matrix[i - 1][j - 1]
        
    def sumRegion(self, row1, col1, row2, col2):
        return self._presum[row2 + 1][col2 + 1] \
            - self._presum[row2 + 1][col1] \
            - self._presum[row1][col2 + 1] \
            + self._presum[row1][col1]
```

#### 548. 两数组的交集 II
```
from collections import defaultdict

class Solution:
    """
    @param nums1: an integer array
    @param nums2: an integer array
    @return: an integer array
    """
    def intersection(self, nums1, nums2):
        # write your code here
        mapping = defaultdict(int)

        for num in nums1:
            mapping[num] += 1
        
        res = []
        for num in nums2:
            mapping[num] -= 1
            if mapping[num] >= 0:
                res.append(num)
        
        return res
```

#### 405. 和为零的子矩阵
```
class Solution:
    """
    @param: matrix: an integer matrix
    @return: the coordinate of the left-up and right-down number
    """
    def submatrixSum(self, matrix):
        # write your code here
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        presum = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                presum[i][j] = matrix[i - 1][j - 1] \
                    + presum[i - 1][j] \
                    + presum[i][j - 1] \
                    - presum[i - 1][j - 1]
        
        # 核心思路很巧妙
        # 先确定两条平行线i1和i2
        # 然后用一条垂直线j去从左到右扫描
        # 看每次的diff = presum[i2][j] - presum[i1][j]是否出现过
        # 如果出现过 说明出现了0矩阵
        
        # 这道题有点坑
        # 习惯于写presum[r + 1] - presum[l]的形式
        # 但是对于二维的矩阵来说
        # 是有可能只有一行的
        # 所以当如果用`for i2 in range(i1, m)`和`presum[i2 + 1][j + 1]`的话
        # 就会被跳过因为此时m == 1
        for i1 in range(m):
            for i2 in range(i1 + 1, m + 1):
                mapping = {}
                for j in range(n + 1):
                    diff = presum[i2][j] - presum[i1][j]
                    if diff in mapping:
                        k = mapping[diff]
                        return [(i1, k), (i2 - 1, j - 1)]
                    mapping[diff] = j

        return [[0, 0], [0, 0]]
```

#### 149. 买卖股票的最佳时机
```
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        # 这道题是只买卖一次
        if not prices:
            return 0
        
        global_min = 2 ** 31 - 1
        res = 0
        # 注意最后一天可以直接丢掉
        # 不可能最后一天买
        for i in range(1, len(prices)):
            global_min = min(global_min, prices[i - 1])
            res = max(res, prices[i] - global_min)
        
        return res
```

#### 139. 最接近零的子数组和
```
class Solution:
    """
    @param: nums: A list of integers
    @return: A list of integers includes the index of the first 
    number and the index of the last number
    """
    def subarraySumClosest(self, nums):
        # write your code here
        # 在数组中找子数组
        # 使的这个子数组之和的绝对值最小
        # 基本思路就是前缀和并排序
        # 然后看相邻的哪一对儿之差最小
        # 很好的思路！！！
        presum = [[0, -1]]
        for inx, num in enumerate(nums):
            last_sum, _ = presum[-1]
            presum.append([last_sum + num, inx])
        
        presum.sort()
        min_diff = 2 ** 31 - 1
        
        res = [0, 0]
        for i in range(1, len(presum)):
            last_presum, last_inx = presum[i - 1]
            curr_presum, curr_inx = presum[i]
            if abs(last_presum - curr_presum) < min_diff:
                min_diff = abs(last_presum - curr_presum)
                # 注意这里的last_inx和curr_inx不是presum的索引
                # 而是原始nums里的索引
                # 而这个索引对应的presum值指的是从0开始到包括这个索引的presum
                # 而最终的min开始要从这个索引之后
                # 所以min索引要加1
                res = [
                    min(last_inx, curr_inx) + 1,
                    max(last_inx, curr_inx),
                ]
        
        return res
```

#### 104. 合并k个排序链表
```
class Solution:
    """
    @param lists: a list of ListNode
    @return: The head of one sorted list.
    """
    def mergeKLists(self, lists):
        # write your code here
        if not lists:
            return
        
        if len(lists) == 1:
            return lists[0]
        
        if len(lists) == 2:
            return self._merge(lists[0], lists[1])
        
        n = len(lists)
        half1 = self.mergeKLists(lists[:n // 2])
        half2 = self.mergeKLists(lists[n // 2:])
        return self._merge(half1, half2)
    
    def _merge(self, l1, l2):
        dummy_node = ListNode(-1)
        curr = dummy_node
        
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        
        if l1:
            curr.next = l1
        if l2:
            curr.next = l2
        
        return dummy_node.next
```

#### 41. 最大子数组
```
class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def maxSubArray(self, nums):
        # write your code here
        # 这道题里的nums是有正有负的
        if not nums:
            return -2 ** 31
        
        local_max = global_max = -2 ** 31
        for num in nums:
            # nums中遍历的顺序
            # 要么先正后负
            # 要么先负后正
            local_max = max(local_max + num, num)
            global_max = max(global_max, local_max)
        
        return global_max
```

#### 620. 最大子序列的和IV
```
class Solution:
    """
    @param nums: an array of integer
    @param k: an integer
    @return: the largest sum
    """
    def maxSubarray4(self, nums, k):
        # write your code here
        if len(nums) < k:
            return 0
        
        presum = [0] * (len(nums) + 1)
        # 这里的min_presum是在遍历i时候
        # 跟i间距k的最小值
        # 这样就能保证了一次循环中
        # 能够定位到长度大于k的之前最小的presum
        min_presum = 0
        res = -2 ** 31
        for i in range(1, len(nums) + 1):
            presum[i] = presum[i - 1] + nums[i - 1]
            
            if i < k:
                continue
            
            if presum[i] - min_presum > res:
                res = presum[i] - min_presum
            
            min_presum = min(min_presum, presum[i - k + 1])
        
        return res
```

#### 617. 最大平均值子数组 II
```
class Solution:
    """
    @param nums: an array with positive and negative numbers
    @param k: an integer
    @return: the maximum average
    """
    def maxAverage(self, nums, k):
        # write your code here
        # 基本思路：平均值一定小于等于数组中的最大值，并且大于等于数组中的最小值
        low, high = min(nums), max(nums)
        while high - low > 1e-7:
            mid = (low + high) / 2
            if self._check(nums, k, mid):
                low = mid
            else:
                high = mid
        
        # 此时return low或者high都是一样的
        return low
    
    # check函数作用是检查nums中是否存在长度大于k
    # 并且和大于mid的子数组
    # 为什么呢？因为如果我们已经算出来了最大的平均值是max_mean,
    # 则这个数组中任何的子数组中每个元素减去这个max_mean之后的sum和一定小于0
    # 只有满足max_mean条件的那个子数组的和这么操作后的sum和是等于0的
    # 这样我们就能找到折半方向了
    def _check(self, nums, k, mean):
        min_sum = 2 ** 31 - 1
        # 核心之一：
        # 注意这里的pre_sum和一般定义的前缀和数组不同
        # 是去mean之后的前缀和!!!!!
        pre_sum = [0] * (len(nums) + 1)
        
        for i in range(1, len(nums) + 1):
            pre_sum[i] = pre_sum[i - 1] + nums[i - 1] - mean
            if i >= k:
                min_sum = min(min_sum, pre_sum[i - k])
                if pre_sum[i] - min_sum > 0:
                    # 这里的min_sum一定前面遍历过的
                    # 长度大于k的子数组的pre_sum中的最小值
                    # 此时如果发现大于0
                    # 说明nums中是存在长度大于k并且均值大于mean的子数组的
                    # 可以说明此时的mean有点小了
                    # 在二分中可以通过增大low来增大mean
                    return True
        
        return False
```

#### 191. 乘积最大子序列
```
class Solution:
    """
    @param nums: An array of integers
    @return: An integer
    """
    def maxProduct(self, nums):
        # write your code here
        # 这道题本质上是dp
        # pos[i]表示子数组[0, i]范围内并且一定包含nums[i]数字的最大子数组乘积
        # neg[i]表示子数组[0, i]范围内并且一定包含nums[i]数字的最小子数组乘积
        # 最后更新的res是全局，可以包括nums[i]也可以不包括nums[i]
        if not nums:
            return 0
        
        if len(nums) == 1:
            return nums[0]
        
        n = len(nums)
        pos, neg = [0] * n, [0] * n
        
        if nums[0] > 0:
            pos[0] = nums[0]
        if nums[0] < 0:
            neg[0] = nums[0]
        
        res = neg[0] if nums[0] < 0 else pos[0]
        for i in range(1, n):
            if nums[i] > 0:
                pos[i] = max(pos[i - 1] * nums[i], nums[i])
                neg[i] = neg[i - 1] * nums[i]
            if nums[i] < 0:
                neg[i] = min(pos[i - 1] * nums[i], nums[i])
                pos[i] = neg[i - 1] * nums[i]
            res = max(res, pos[i])
        
        return res
```

#### 151. 买卖股票的最佳时机 III
```
class Solution:
    """
    @param prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        if not prices:
            return 0
        
        # 核心思想
        # 两次交易必定是错开的
        # 所以某天我们要看两个值：
        # 1是某天卖出之前买入的股票
        # 2是某天买入之后卖出的股票
        # 二者之和的最大值就是答案!!!
        n = len(prices)
        prev = [0] * n
        future = [0] * n
        
        min_v = prices[0]
        for i in range(1, n):
            min_v = min(min_v, prices[i])
            prev[i] = max(prev[i - 1], prices[i] - min_v)
        
        max_v = prices[-1]
        for i in range(n - 2, -1, -1):
            max_v = max(max_v, prices[i + 1])
            future[i] = max(future[i + 1], max_v - prices[i])
        
        res = 0
        for i in range(n):
            res = max(res, prev[i] + future[i])
        
        return res
```

#### 150. 买卖股票的最佳时机 II
```
class Solution:
    """
    @param: prices: Given an integer array
    @return: Maximum profit
    """
    def maxProfit(self, prices):
        # write your code here
        # 股票2本质就是贪心
        res = 0
        for i in range(len(prices) - 1):
            res += max(0, prices[i + 1] - prices[i])
        return res
```

#### 45. 最大子数组差
```
class Solution:
    """
    @param nums: A list of integers
    @return: An integer indicate the value of maximum difference between two substrings
    """
    _INT_MAX = 2 ** 31 - 1
    _INT_MIN = -2 ** 31
    def maxDiffSubArrays(self, nums):
        # write your code here
        if not nums:
            return 0
        n = len(nums)
    
        # left pass
        left_min = [self._INT_MAX] * n
        left_max = [self._INT_MIN] * n
        local_min = self._INT_MAX
        local_max = self._INT_MIN

        for i in range(n):
            local_min = min(local_min + nums[i], nums[i])
            local_max = max(local_max + nums[i], nums[i])

            if i == 0:
                left_min[i] = local_min
                left_max[i] = local_max
                continue

            left_min[i] = min(left_min[i - 1], local_min)
            left_max[i] = max(left_max[i - 1], local_max)

        # right pass
        right_min = [self._INT_MAX] * n
        right_max = [self._INT_MIN] * n
        local_min = self._INT_MAX
        local_max = self._INT_MIN
        
        for i in range(n - 1, -1, -1):
            local_min = min(local_min + nums[i], nums[i])
            local_max = max(local_max + nums[i], nums[i])
            
            if i == n - 1:
                right_min[i] = local_min
                right_max[i] = local_max
                continue
    
            right_min[i] = min(right_min[i + 1], local_min)
            right_max[i] = max(right_max[i + 1], local_max)
        
        res = self._INT_MIN
        
        # 注意这里i坐标都是闭区间坐标
        for i in range(n - 1):
            res = max(
                res,
                abs(left_min[i] - right_max[i + 1]),
                abs(left_max[i] - right_min[i + 1]),
            )

        return res
```

#### 42. 最大子数组 II
```
class Solution:
    """
    @param: nums: A list of integers
    @return: An integer denotes the sum of max two non-overlapping subarrays
    """
    def maxTwoSubArrays(self, nums):
        # write your code here
        n = len(nums)
        
        left = [-2 ** 31] * n
        right = [-2 ** 31] * n
        
        local_max = global_max = -2 ** 31
        for i in range(n):
            local_max = max(local_max + nums[i], nums[i])
            global_max = max(global_max, local_max)
            left[i] = global_max
        
        local_max = global_max = - 2 ** 31
        for i in range(n - 1, -1, -1):
            local_max = max(local_max + nums[i], nums[i])
            global_max = max(global_max, local_max)
            right[i] = global_max
        
        res = -2 ** 31
        for i in range(n - 1):
            res = max(res, left[i] + right[i + 1])
        
        return res
```

#### 404. 子数组求和 II
```
class Solution:
    """
    @param A: An integer array
    @param start: An integer
    @param end: An integer
    @return: the number of possible answer
    """
    def subarraySumII(self, A, start, end):
        # write your code here
        # 理解的不好！再回头看看
        l = r = 0
        s1 = s2 = 0
        res = 0

        for i in range(len(A)):
            # 优先让s1和s2分别加到刚刚大于start和刚刚小于end
            while l <= i or (l < len(A) and s1 < start):
                s1 += A[l]
                l += 1
            while r <= i or (r < len(A) and s2 <= end):
                s2 += A[r]
                r += 1
            if start <= s1 <= end:
                if s2 > end:
                    res += r - l
                else:
                    res += r - l + 1
            s1 -= A[i]
            s2 -= A[i]
        
        return res
```

#### 393. 买卖股票的最佳时机 IV
```
class Solution:
    """
    @param K: An integer
    @param prices: An integer array
    @return: Maximum profit
    """
    def maxProfit(self, K, prices):
        # write your code here
        n = len(prices)
        
        # 比如10天 6次交易
        # 而且每天只能买入或者卖出一次
        # 实际上就是在10天里最多可以有12次操作
        # 即根本没有限制
        if K > n // 2:
            return self._quick_solver(prices)
        
        dp = [-2 ** 31] * (2 * K + 1)
        dp[0] = 0
        for i in range(n):
            # 题目给出了K次交易
            # 则一共有2 * K次操作
            # 由于每天最多只能有一次操作
            # 所以这里的j上限肯定是min(2 * K, i + 1)
            for j in range(min(2 * K, i + 1), 0, -1):
                if j % 2 == 0:
                    dp[j] = max(dp[j], dp[j - 1] + prices[i])
                else:
                    dp[j] = max(dp[j], dp[j - 1] - prices[i])
        
        return max(dp)
    
    def _quick_solver(self, prices):
        res = 0
        for i in range(len(prices) - 1):
            if prices[i + 1] - prices[i] > 0:
                res += prices[i + 1] - prices[i]
        return res
```

#### 43. 最大子数组 III
```
class Solution:
    """
    @param nums: A list of integers
    @param k: An integer denote to find k non-overlapping subarrays
    @return: An integer denote the sum of max k non-overlapping subarrays
    """
    def maxSubArray(self, nums, k):
        # write your code here
        # 理解的不好！回头再仔细看
        if k > len(nums):
            return 0

        n = len(nums)
        local_max = [[0] * (n + 1) for _ in range(k + 1)]
        global_max = [[0] * (n + 1) for _ in range(k + 1)]
        
        for i in range(1, k + 1):
            local_max[i][i - 1] = -2 ** 31
            for j in range(i, n + 1):
                local_max[i][j] = max(
                    local_max[i][j - 1],
                    global_max[i - 1][j - 1],
                ) + nums[j - 1]
                
                if i == j:
                    global_max[i][j] = local_max[i][j]
                else:
                    global_max[i][j] = max(global_max[i][j - 1], local_max[i][j])
        
        return global_max[-1][-1]
```

### 10 - Additional - Dynamic Programming

#### 115. 不同的路径 II
```
class Solution:
    """
    @param obstacleGrid: A list of lists of integers
    @return: An integer
    """
    def uniquePathsWithObstacles(self, obstacleGrid):
        # write your code here
        if not obstacleGrid or not obstacleGrid[0]:
            return 0
        
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if m == 1 and n == 1:
            return 1 if obstacleGrid[0][0] == 0 else 1

        dp = [[0] * n for _ in range(m)]
        
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                break
            dp[i][0] = 1
        
        for j in range(n):
            if obstacleGrid[0][j] == 1:
                break
            dp[0][j] = 1

        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[-1][-1]
```

#### 114. 不同的路径
```
class Solution:
    """
    @param m: positive integer (1 <= m <= 100)
    @param n: positive integer (1 <= n <= 100)
    @return: An integer
    """
    def uniquePaths(self, m, n):
        # write your code here
        if m == 1 or n == 1:
            return 1
        
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        
        return dp[-1][-1]
```

#### 111. 爬楼梯
```
class Solution:
    """
    @param n: An integer
    @return: An integer
    """
    def climbStairs(self, n):
        # write your code here
        if n == 0:
            return 0
        if n <= 2:
            return n

        a, b = 1, 2
        for i in range(2, n):
            a, b = b, a + b

        return b
```

#### 110. 最小路径和
```
class Solution:
    """
    @param grid: a list of lists of integers
    @return: An integer, minimizes the sum of all numbers along its path
    """
    def minPathSum(self, grid):
        # write your code here
        if not grid or not grid[0]:
            return 2 ** 31 - 1
        
        m, n = len(grid), len(grid[0])
        dp = [[2 ** 31 - 1] * n for _ in range(m)]
        dp[0][0] = grid[0][0]
        
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        
        for j in range(1, n):
            dp[0][j] = dp[0][j - 1] + grid[0][j]
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        
        return dp[-1][-1]
```

#### 603. 最大整除子集
```
class Solution:
    """
    @param: nums: a set of distinct positive integers
    @return: the largest subset 
    """
    def largestDivisibleSubset(self, nums):
        # write your code here
        # 给一个由 无重复的正整数 组成的集合
        # 找出满足任意两个元素 (Si, Sj) 都有 Si % Sj = 0 
        # 或 Sj % Si = 0 成立的最大子集
        if not nums:
            return []
        
        nums.sort()
        n = len(nums)
        # dp[i]表示nums的[0-i]子数组的最大divisible subset长度
        dp = [1] * n
        # father[i]表示nums[i]可以整除的最大数字下标（注意nums是sort过的）
        # 这里理解小的数字为father
        father = [None] * n
        
        for i in range(n):
            # 注意这里j是大于等于i的递增
            for j in range(i):
                if nums[i] % nums[j] == 0:
                    # 说明此时外层的dp[i]长度小了
                    # 可以被更新
                    if dp[i] < dp[j] + 1:
                        dp[i] = dp[j] + 1
                        father[i] = j
        
        res = []
        max_inx = dp.index(max(dp))
        while max_inx is not None:
            res.append(nums[max_inx])
            # 往回倒推小的数字
            max_inx = father[max_inx]
        
        return res
```

#### 611. 骑士的最短路线
```
from collections import deque

class Solution:
    """
    @param grid: a chessboard included 0 (false) and 1 (true)
    @param source: a point
    @param destination: a point
    @return: the shortest path 
    """
    # 这道题问的是从起点到终点的最短路线
    # 注意这里的棋盘是有障碍物的
    _DIRS = [
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
    ]
    
    def shortestPath(self, grid, source, destination):
        # write your code here
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        
        queue = deque()
        queue.append((source.x, source.y))
        visited[source.x][source.y] = True
        
        res = 0
        while queue:
            q_len = len(queue)
            for _ in range(q_len):
                ci, cj = queue.popleft()
                if ci == destination.x and cj == destination.y:
                    return res
                for di, dj in self._DIRS:
                    newi, newj = ci + di, cj + dj
                    if not 0 <= newi < m or \
                        not 0 <= newj < n or \
                        grid[newi][newj] == 1 or \
                        visited[newi][newj]:
                        continue
                    queue.append((newi, newj))
                    visited[newi][newj] = True
            res += 1
        
        return -1
```

#### 513. 完美平方
```
class Solution:
    """
    @param n: a positive integer
    @return: An integer
    """
    def numSquares(self, n):
        # write your code here
        dp = [2 ** 31 - 1] * (n + 1)
        i = 1
        # 初始化
        while i * i <= n:
            # 此时数字i ** 2自己就是完美平方数
            dp[i * i] = 1
            i += 1
        
        for i in range(1, n + 1):
            # 核心之一：
            # 主循环先fix i
            # 再去尽可能的更新dp[i + j * j]的值
            j = 1
            while i + j * j <= n:
                dp[i + j * j] = min(dp[i + j * j], dp[i] + 1)
                j += 1
        
        return dp[-1]
```

#### 116. 跳跃游戏
```
class Solution:
    """
    @param nums: A list of integers
    @return: A boolean
    """
    def canJump(self, nums):
        # write your code here
        # 标准的贪心 O(n)
        reach = 0
        n = len(nums)
        for i in range(n):
            if i > reach or reach >= n - 1:
                break
            reach = max(reach, i + nums[i])
        
        return reach >= n - 1
        
        # DP解法 O(n ^ 2)
        # n = len(nums)
        # dp = [False] * n
        # dp[0] = True
        
        # for i in range(1, n):
        #     for j in range(i):
        #         if dp[j] and j + nums[j] >= i:
        #             dp[i] = True
        #             break
        
        # return dp[-1]
```

#### 109. 数字三角形
```
class Solution:
    """
    @param triangle: a list of lists of integers
    @return: An integer, minimum path sum
    """
    def minimumTotal(self, triangle):
        # write your code here
        if not triangle or not triangle[0]:
            return 2 ** 31 - 1
        
        m = len(triangle)
        if m == 1:
            return min(triangle[0])

        dp = [triangle[0]]
        # 这道题思路是从最底层向上
        # 实际上最后塔尖的数字就是解答
        for i in range(m - 2, -1, -1):
            for j in range(len(triangle[i])):
                # 比如j = 2
                # 只能到达下一行索引为2或者3的点
                triangle[i][j] += min(triangle[i + 1][j], triangle[i + 1][j + 1])
        
        return triangle[0][0]
```

#### 76. 最长上升子序列
```
class Solution:
    """
    @param nums: An integer array
    @return: The length of LIS (longest increasing subsequence)
    """
    def longestIncreasingSubsequence(self, nums):
        # write your code here
        # 非常典型的DP题目，好题，需要多看
        if not nums:
            return 0
        
        n = len(nums)
        # dp[i]定义：nums[0]到nums[i]左闭右闭区间的LIS是多少
        dp = [1] * n
        
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        
        # 注意：这里是的dp不是递增的！！！
        # 因为中间可能出现1(即dp[i]之前全部的数字都比它小，没有更新，所以
        # dp[i]还是初始值1)
        # 例子： dp最终可能是[1,2,3,4,5,1]
        # nums是[100, 200, 300, 400, 500, 1]
        # 所以要返回一个全局的最大
        return max(dp)
```

#### 272. 爬楼梯 II
```
class Solution:
    """
    @param n: An integer
    @return: An Integer
    """
    def climbStairs2(self, n):
        # write your code here
        # 跟爬楼梯I几乎一样
        if n < 2:
            return 1
        
        if n == 2:
            return 2
        
        a, b, c = 1, 1, 2
        for i in range(3, n + 1):
            a, b, c = b, c, a + b + c
        
        return c
```

#### 630. 骑士的最短路径II
```
class Solution:
    """
    @param grid: a chessboard included 0 and 1
    @return: the shortest path
    """
    def shortestPath2(self, grid):
        # write your code here
        # 这道题问的是从0 0点走到m - 1 n - 1点
        # 在有障碍物的情况下
        # 最少的步数
        if not grid or not grid[0]:
            return -1
        
        m, n = len(grid), len(grid[0])
        # 注意这里的棋子只能从左往右走
        dirs = [(-2, -1), (-1, -2), (1, -2), (2, -1)]
        # dp[i][j]指从起点走到i j点最短需要几步
        dp = [[2 ** 31 - 1] * n for _ in range(m)]
        dp[0][0] = 0
        
        # 疑问：
        # 为什么交换下面的循环次序不能AC？
        # for i in range(m):
        #     for j in range(n):
        for j in range(n):
            for i in range(m):
                if grid[i][j] == 0:
                    for di, dj in dirs:
                        newi, newj = i + di, j + dj
                        if self._valid_pos(newi, newj, grid):
                            # 说明这个点之前从来没有走过
                            # 可以更新
                            if dp[newi][newj] != 2 ** 31 - 1:
                                dp[i][j] = min(dp[i][j], dp[newi][newj] + 1)

        # 说明m - 1 n - 1这个点能被走到
        if dp[-1][-1] != 2 ** 31 - 1:
            return dp[-1][-1]
        return -1
    
    def _valid_pos(self, i, j, grid):
        if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == 0:
            return True
        return False
```

#### 117. 跳跃游戏 II
```
class Solution:
    """
    @param nums: A list of integers
    @return: An integer
    """
    def jump(self, nums):
        # write your code here
        steps = 0
        n = len(nums)
        curr_i = curr_most_i = 0
        while curr_most_i < n - 1:
            steps += 1
            pre_most_i = curr_most_i
            while curr_i <= pre_most_i:
                curr_most_i = max(curr_most_i, curr_i + nums[curr_i])
                curr_i += 1
            if curr_most_i == pre_most_i:
                return -1
        
        return steps
        
        
        # dp解法 会TLE
        # n = len(nums)
        # # dp[i]指跳到i位置最短的steps是多少
        # dp = [2 ** 31 - 1] * n
        # dp[0] = 0
        
        # for i in range(1, n):
        #     for j in range(i):
        #         # 后一个条件说明当前从j是可以跳过i的
        #         # 所以此时可以去更新dp了
        #         if dp[j] != 2 ** 31 - 1 and j + nums[j] >= i:
        #             dp[i] = min(dp[i], dp[j] + 1)
        
        # return dp[-1]
```

#### 602. 俄罗斯套娃信封
```
class Solution:
    """
    @param: envelopes: a number of envelopes with widths and heights
    @return: the maximum number of envelopes
    """
    def maxEnvelopes(self, envelopes):
        # write your code here
        # 二分法
        envelopes_copy = [(i[0], -i[1]) for i in envelopes]
        envelopes_copy.sort()
        n = len(envelopes_copy)
        
        res = []
        for i in range(n):
            l, r = 0, len(res)
            w, h = envelopes_copy[i][0], -envelopes_copy[i][1]
            while l < r:
                mid = l + (r - l) // 2
                if h > res[mid][1]:
                    l = mid + 1
                else:
                    r = mid
            if r == len(res):
                res.append((w, h, i))
            else:
                res[r] = (w, h, i)
        
        return len(res)
        
        # DP解法 会TLE
        # if not envelopes:
        #     return 0

        # envelopes.sort()
        # n = len(envelopes)
        # # dp定义是对于一个排好序的envelopes
        # # 第i个信封里最多能装多少封信
        # dp = [1] * n
        # for i in range(n):
        #     for j in range(i):
        #         if envelopes[i][0] > envelopes[j][0] and envelopes[i][1] > envelopes[j][1]:
        #             dp[i] = max(dp[i], dp[j] + 1)
        
        # return max(dp)
```

#### 622. 青蛙跳
```
class Solution:
    """
    @param stones: a list of stones' positions in sorted ascending order
    @return: true if the frog is able to cross the river or false
    """
    def canCross(self, stones):
        # write your code here
        stones_hash = {}
        for stone in stones:
            stones_hash[stone] = set()
        
        stones_hash[0].add(0)
        
        for stone in stones:
            for jumps in stones_hash[stone]:
                if stone + jumps - 1 != stone \
                    and stone + jumps - 1 > 0 \
                    and stone + jumps - 1 in stones_hash:
                    stones_hash[stone + jumps - 1].add(jumps - 1)
                if stone + jumps in stones_hash:
                    stones_hash[stone + jumps].add(jumps)
                if stone + jumps + 1 in stones_hash:
                    stones_hash[stone + jumps + 1].add(jumps + 1)
        
        return len(stones_hash[stones[-1]]) > 0
```

#### 254. 丢鸡蛋
```
class Solution:
    """
    @param: n: An integer
    @return: The sum of a and b
    """
    def dropEggs(self, n):
        # write your code here
        # 等间距扔鸡蛋的最坏情况：
        # 出现在最后一个区间的最后一个数
        # 可以每过一个区间把区间长度-1
        # 这样就能够达到最佳的方案
        # x + (x - 1) + (x - 2) + ... + 1 = n
        # 等差数列求和的问题
        x = 1
        while x * (x + 1) // 2 < n:
            x += 1
        return x
```

#### 584. 丢鸡蛋II
```
class Solution:
    """
    @param m: the number of eggs
    @param n: the number of floors
    @return: the number of drops in the worst case
    """
    # 理解的不是很好
    # 回头再看看
    def dropEggs2(self, m, n):
        # write your code here
        # 这道题是说m个鸡蛋 n层建筑
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 如果只有1层楼
        # 不管有多少个鸡蛋 我只需要1个就够了
        for i in range(1, m + 1):
            dp[i][1] = 1
        
        # 如果我们只要1个鸡蛋
        # 有几层楼 就扔几次这个鸡蛋
        for j in range(1, n + 1):
            dp[1][j] = j
        
        for i in range(2, m + 1):
            for j in range(2, n + 1):
                dp[i][j] = 2 ** 31 - 1
                for k in range(1, j + 1):
                    # 最终只有两种情况(碎还是没有碎)
                    # 一是如果鸡蛋在k层扔了以后碎了，那么我们只需要去用剩下的i - 1个鸡蛋
                    # 去检查小于k层楼的情况
                    # 二是如果鸡蛋仔k层扔了以后没有碎，我们只需要去检查大于k层
                    # 所以问题简化成了j - k层楼外加i个鸡蛋的问题
                    dp[i][j] = min(
                        dp[i][j],
                        max(dp[i - 1][k - 1], dp[i][j - k]) + 1,
                    )
        
        return dp[-1][-1]
```