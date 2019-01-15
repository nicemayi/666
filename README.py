#### 1. Two Sum
```
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        mapping = {}
        for i, num in enumerate(nums):
            if target - num in mapping:
                return [mapping[target - num], i]
            mapping[num] = i
        
        return []
```

#### 2. Add Two Numbers
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy_node = ListNode(-1)
        curr = dummy_node
        carry = 0
        
        while l1 or l2:
            n1 = l1.val if l1 else 0
            n2 = l2.val if l2 else 0
            total = n1 + n2 + carry
            carry = total // 10
            curr.next = ListNode(total % 10)
            curr = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        
        if carry:
            curr.next = ListNode(carry)
        
        return dummy_node.next
```

#### 3. Longest Substring Without Repeating Characters
```
class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 追赶型双指针
        mapping = set()
        start = inx = 0
        res = 0
        
        while inx < len(s):
            if s[inx] in mapping:
                mapping.remove(s[start])
                start += 1
            else:
                res = max(res, inx - start + 1)
                mapping.add(s[inx])
                inx += 1
        
        return res
```

#### 4. Median of Two Sorted Arrays, Hard, Facebook
```
class Solution:
    def findMedianSortedArrays(self, A, B):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        # A, B两个排序过的数组，如果A的中位数比B的中位数小
        # 则总的A + B这个长的数组的中位数一定在A的左半部分
        # 如果B的长度太小，中位数也不会在A的左半部分。
        n = len(A) + len(B)
        
        if n % 2 == 0:
            smaller = self.find_kth_largest(A, B, n // 2)
            larger = self.find_kth_largest(A, B, n // 2 + 1)
            return (smaller + larger) / 2
        else:
            return self.find_kth_largest(A, B, n // 2 + 1)
        
    
    def find_kth_largest(self, A, B, k):
        if not A:
            return B[k - 1]
        
        if not B:
            return A[k - 1]
        
        if k == 1:
            return min(A[0], B[0])
        
        a = A[k // 2 - 1] if len(A) >= k // 2 else 2 ** 31 - 1
        b = B[k // 2 - 1] if len(B) >= k // 2 else 2 ** 31 - 1
        
        if a < b:
            return self.find_kth_largest(A[k // 2:], B, k - k // 2)
        else:
            return self.find_kth_largest(A, B[k // 2:], k - k // 2)
```

#### 5. Longest Palindromic Substring
```
class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
            return ''
        
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        max_len = 0
        l = r = 0

        for i in range(n):
            for j in range(i):
                if s[i] == s[j] and (i - j < 2 or dp[j + 1][i - 1]):
                    dp[j][i] = True
                    if max_len < i - j + 1:
                        max_len = i - j + 1
                        l, r = j, i
            dp[i][i] = True
        
        return s[l:r + 1]
```

#### 6. ZigZag Conversion
```
class Solution:
    """
    P   A   H   N
    A P L S I I G
    Y   I   R
    And then read line by line: "PAHNAPLSIIGYIR"
    这道题是说的输入如果是正常字符串例如"PAYPALISHIRING"，给定一个num of rows
    输出"PAHNAPLSIIGYIR"
    """
    def convert(self, s, num_rows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if num_rows <= 1:
            return s
        
        res = ''
        # 这道题的核心
        step = 2 * num_rows - 2
        for i in range(num_rows):
            for j in range(i, len(s), step):
                res += s[j]
                temp = j + step - 2 * i 
                # 下面是指除了首行和尾行
                if i != 0 and i != num_rows - 1 and temp < len(s):
                    res += s[temp]
        
        return res
```

#### 7. Reverse Integer
```
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
#         if x == 0:
#             return 0
        
#         result = ''
#         if x<0:
#             result = '-'
#             x = -x

#         while x > 0:
#             tmp = x % 10
#             if tmp != 0:
#                 result += str(tmp)
#             x = x // 10
            
#         result = int(result)

#         return result
        if x == 0:
            return 0
        
        sign = 1
        if x < 0:
            sign = -1
            x = -x
        
        reverse = 0
        while x > 0:
            reverse = 10 * reverse + x % 10
            x //= 10
        reverse *= sign
        
        if not -2 ** 31 <= reverse <= 2 ** 31 - 1:
            return 0
        return reverse
```

#### 8. String to Integer (atoi), Medium, Facebook
```
class Solution:
    def myAtoi(self, s):
        """
        :type str: str
        :rtype: int
        """
        s = s.strip()
        if not s:
            return 0
        
        sign = 1
        if s[0] == '-':
            sign = -1
            s = s[1:]
        elif s[0] == '+':
            s = s[1:]
        
        res = 0
        for i in range(len(s)):
            if not '0' <= s[i] <= '9':
                break
            res = 10 * res + int(s[i])
        
        res *= sign
        if -2 ** 31 <= res <= 2 ** 31 - 1:
            return res
        
        return 2 ** 31 - 1 if sign == 1 else -2 ** 31
```

#### 9. Palindrome Number
```
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return False
        
        temp = x
        y = 0
        
        while temp:
            y = y * 10 + temp % 10
            temp = temp // 10

        return y == x
```

#### 10. Regular Expression Matching, Hard, Facebook
```
# 递归
class Solution(object):
    def isMatch(self, text, pattern):
        if not pattern:
            return not text

        first_match = bool(text) and pattern[0] in {text[0], '.'}

        if len(pattern) >= 2 and pattern[1] == '*':
            return (self.isMatch(text, pattern[2:]) or
                    first_match and self.isMatch(text[1:], pattern))
        else:
            return first_match and self.isMatch(text[1:], pattern[1:])

# DP
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        for i in range(1, n + 1):
            # 当s为空的时候
            # p的前i个字符如果能和空字符串匹配的话
            # 必须p[i - 1] == '*'并且p的前i - 2个要能和空字符串匹配上
            # 这样显然不管p[i - 2]个字符是什么都可以（可以为空，所以可以不考虑）
            dp[0][i] = p[i - 1] == '*' and dp[0][i - 2]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    # 三种条件：
                    # 1是类似前面的初始条件（这样这次可用可不用）
                    # 2是前j - 1个字符匹配上（这样这次可以强制不用）
                    # 3是p的第j - 2个字符要么匹配上s当前的[i - 1]个字符
                    # 要么为点'.'
                    dp[i][j] = (
                        dp[i][j - 2] or
                        dp[i][j - 1] or (
                            dp[i - 1][j] and 
                            (s[i - 1] == p[j - 2] or p[j - 2] == '.')
                        )
                    )
                elif p[j - 1] == '.':
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = s[i - 1] == p[j - 1] and dp[i - 1][j - 1]
        
        return dp[-1][-1]
```

#### 11. Container With Most Water
```
class Solution:
    def maxArea(self, heights):
        """
        :type height: List[int]
        :rtype: int
        """
        res = 0
        i, j = 0, len(heights) - 1
        
        while i < j:
            res = max(res, min(heights[i], heights[j]) * (j - i))
            
            if heights[i] < heights[j]:
                i += 1
            else:
                j -= 1
        
        return res
```

#### 12. Integer to Roman
```
class Solution:
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        # 一共7种类型
        # 除了最后一个I
        # 每一种类型还有左边减去下一位的类型
        # 相当于一共处理13种情况
        vals = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        strs = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        res = ''
        for i in range(len(vals)):
            while num >= vals[i]:
                num -= vals[i]
                res += strs[i]
        return res
```

#### 13. Roman to Integer
```
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        罗马字符串变int数字
        """
        roman = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10, 'V': 5, 'I': 1}
        
        res = 0
        for i in range(len(s) - 1):
            if roman[s[i]] < roman[s[i + 1]]:
                res -= roman[s[i]]
            else:
                res += roman[s[i]]
        
        return res + roman[s[-1]]
```

#### 14. Longest Common Prefix
```
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return ''
        if len(strs) == 1:
            return strs[0]

        min_len = min(len(s) for s in strs)
        res = ''
        for i in range(min_len):
            pivot_ch = strs[0][i]
            if any(word[i] != pivot_ch for word in strs[1:]):
                break
            res += pivot_ch
        
        return res
```

#### 15. 3Sum
```
class Solution:
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if not nums:
            return []

        nums.sort()
        res = []

        for i in range(len(nums) - 2):
            
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            current = nums[i]
            left, right = i + 1, len(nums) - 1
            while left < right:
                total = current + nums[left] + nums[right]
                if total == 0:
                    res.append([current, nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif total > 0:
                    right -= 1
                else:
                    left += 1
            
        return res
```

#### 17. Letter Combinations of a Phone Number
```
class Solution:
    
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
        """
        :type digits: str
        :rtype: List[str]
        """
        res = []
        
        if not digits:
            return []
        
        self.find_combination(digits, 0, '', res)
        return res
        
    def find_combination(self, digits, index, current, results):
        if index == len(digits):
            results.append(current)
            return
        
        letters = self.LETTER_MAP[digits[index]]
        for c in letters:
            self.find_combination(digits, index + 1, current + c, results)
```

#### 18. 4Sum
```
class Solution:
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        nums.sort()
        n = len(nums)
        res = set()
        for i in range(n - 3):
            for j in range(i + 1, n - 2):
                left, right = j + 1, n - 1
                while left < right:
                    val = nums[i] + nums[j] + nums[left] + nums[right]
                    if val == target:
                        res.add((nums[i], nums[j], nums[left], nums[right]))
                        left += 1
                        right -= 1
                    elif val < target:
                        left += 1
                    else:
                        right -= 1
        
        return list(res)
```

#### 19. Remove Nth Node From End of List
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy_node = ListNode(-1)
        dummy_node.next = head
        p1 = p2 = dummy_node
        
        # 思路就是先将p1从dummy_node移动n位
        # 然后再将p1和p2同时移动
        # 直到p1.next为None时候，此时p1指向链表中的最后一位
        # 而p2正好是移动到想要删除的节点的前一位
        for _ in range(n):
            p1 = p1.next
        
        while p1.next:
            p1 = p1.next
            p2 = p2.next
        
        p2.next = p2.next.next
        return dummy_node.next
```

#### 20. Valid Parentheses
```
class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        bracket_hash = {'}': '{', ']': '[', ')': '('}
        bracket_stack = []
        
        for ch in s:
            if ch not in bracket_hash:
                bracket_stack.append(ch)
            else:
                if not bracket_stack or bracket_hash[ch] != bracket_stack[-1]:
                    return False
                bracket_stack.pop()
        
        return not bracket_stack
```

#### 21. Merge Two Sorted Lists
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if not l1:
            return l2
        
        if not l2:
            return l1
        
        dummy = ListNode(0)
        curr = dummy
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
        
        return dummy.next
```

#### 22. Generate Parentheses
```
class Solution:
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        Given number of n, meaning n pairs of parentheses
        write a function to generate all combinations of well-formed parentheses.
        """
        res = []
        self._dfs(n, n, '', res)
        return res
    
    ## left表示当前剩余的左括号数目
    ## right表示当前剩余的右括号数目
    ## 如果当前剩余的左括号数目大于当前剩余的右括号数目
    ## 肯定是出现了不合法的情况（例如：先一个")..."的情况）
    def _dfs(self, left, right, current, results):
        if left > right:
            return
        
        if left == 0 and right == 0:
            results.append(current)
            return
        
        if left > 0:
            self._dfs(left - 1, right, current + '(', results)
        if right > 0:
            self._dfs(left, right - 1, current + ')', results)
```

#### 23. Merge k Sorted Lists, Hard, Facebook, Linkedin
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
from heapq import heappush
from heapq import heappop

class Solution:
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        
        # 下面3行仅仅是为了证明本算法是正确的
        def cmp(self, other):
            return self.val < other.val
        ListNode.__lt__ = cmp
        
        hp = []
        for each_head in lists:
            if each_head:
                heappush(hp, (each_head.val, each_head))
        
        dummy_head = ListNode(0)
        curr = dummy_head
        while hp:
            _, node = heappop(hp)
            if node.next:
                heappush(hp, (node.next.val, node.next))
            curr.next = node
            curr = curr.next
        
        return dummy_head.next
```

#### 24. Swap Nodes in Pairs
```
class Solution:
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy_head = ListNode(-1)
        dummy_head.next = head        
        pre = dummy_head
        
        while pre.next and pre.next.next:
            temp = pre.next.next
            pre.next.next = temp.next
            temp.next = pre.next
            pre.next = temp
            pre = temp.next
        
        return dummy_head.next
```

#### 25. Reverse Nodes in k-Group, Hard, Facebook
```
class Solution:
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        # 这道题记住有三个关键变量：
        # 左边界left， 原头节点last，当前指针curr
        # last就是原头节点（需要翻转部分的第一个节点）
        # 每次循环都将原头节点的next指向当前curr的next
        # 然后curr的next指向左边界（left）的next
        # 左边界的next指向当前curr
        # 最后移动curr

        # 翻转链表不太好理解
        # http://blog.csdn.net/feliciafay/article/details/6841115
        # 这里面的第三种方法比较好理解！！！
        # 链表翻转需要用到3根指针：
        # 1.当前链表头curr
        # 2.当前链表头的前一个元素pre
        # 3.当前立案表头的下一个元素post
        # 每次循环做以下4件事：
        # 1. curr.next = pre
        # 2. pre = curr
        # 3. curr = post
        # 4. post = post.next
        # 直到curr为None为止
        if not head or k < 2:
            return head
        
        dummy_node = ListNode(0)
        dummy_node.next = head
        
        pre = dummy_node
        curr = head
        
        cnt = 0
        while curr:
            cnt += 1
            if cnt % k == 0:
                pre = self._reverse_one_group(pre, curr.next)
                # 注意这里不应该再是curr.next了
                # 因为curr需要一个一个的遍历
                # 当前如果需要翻转，翻转后的新头就是pre
                # 所以一个一个来，curr就应该是pre的next
                curr = pre.next
            else:
                curr = curr.next
        
        return dummy_node.next
    
    
    def _reverse_one_group(self, left, right):
        new_head = left.next
        curr = new_head.next
        
        while curr is not right:
            new_head.next = curr.next
            curr.next = left.next
            left.next = curr
            curr = new_head.next
        
        return new_head
```

#### 26. Remove Duplicates from Sorted Array
```
class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        if len(nums) == 1:
            return 1
        
        last_pos = 0
        for i in range(1, len(nums)):
            if nums[i] != nums[last_pos]:
                last_pos += 1
                # 说明此时nums[i]是一个新数字
                nums[last_pos] = nums[i]
        
        return last_pos + 1
```

#### 28. Implement strStr()
```
class Solution:
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if not haystack:
            return 0 if not needle else -1
        if not needle:
            return 0
        
        for i in range(len(haystack) - len(needle) + 1):
            j = 0
            while j < len(needle):
                if haystack[i + j] != needle[j]:
                    break
                j += 1
            # 表示没有break，则肯定是j的长度此时等于needle的长度，说明完全匹配上了
            else:
                return i
        
        return -1
```

#### 29. Divide Two Integers
###### Medium, Facebook
```
class Solution:
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        if divisor == 0:
            raise Exception('Divisor can not be zeor!')
        
        if dividend == 0:
            return 0
        
        sign = 1
        if (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0):
            sign = -1
        
        res = 0
        dividend, divisor = abs(dividend), abs(divisor)
        while dividend >= divisor:
            shift = 1
            while dividend >= divisor << shift:
                shift += 1
            dividend -= divisor << (shift - 1)
            res += 1 << (shift - 1)
        
        if sign * res > 2 ** 31 - 1:
            return 2 ** 31 - 1
        
        if sign * res < - 2 ** 31:
            return -2 ** 31
        
        return sign * res
```

#### 30. Substring with Concatenation of All Words, Hard, Facebook
```
from collections import defaultdict

class Solution:
    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        You are given a string, s, and a list of words, words, 
        that are all of the same length. 
        Find all starting indices of substring(s) in s 
        that is a concatenation of each word in words exactly 
        once and without any intervening characters.
        """
        if not s or not words:
            return []
        
        mapping = defaultdict(int)
        for word in words:
            mapping[word] += 1
        
        res = []
        len_s = len(s)
        n, m = len(words), len(words[0])
        for i in range(len_s - n * m + 1):
            temp_mapping = defaultdict(int)
            for j in range(n):
                new_str = s[i + j * m:i + j * m + m]
                if new_str not in mapping:
                    break
                temp_mapping[new_str] += 1
                if temp_mapping[new_str] > mapping[new_str]:
                    break
            else:
                res.append(i)
        
        return res
```

#### 31. Next Permutation
```
class Solution:
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """

        # 1.pos1: 从后往前遍历寻找第一个值变小的index（num[i] < num[i + 1]）。
        # 2.pos2: 从后往前遍历寻找第一个比num[pos1]大的index。
        # 例子：[..., 8, 9], pos1 = 值8的位置，pos2=值9的位置。
        # 3. 交换pos1和pos2的值
        # 4. 将pos1之后的子数组reverse。
        if not nums or len(nums) <= 1:
            return
        
        if len(nums) == 2:
            nums[1], nums[0] = nums[0], nums[1]
            return
        
        n = len(nums)
        for i in range(n - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                break
        else:
            start = 0
            end = n - 1
            while start < end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1
            return
        
        for j in range(n - 1, i, -1):
            if nums[j] > nums[i]:
                break
        
        nums[i], nums[j] = nums[j], nums[i]
        start = i + 1
        end = n - 1
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
```

#### 32. Longest Valid Parentheses, Hard, Facebook
```
class Solution:
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 当前的stack只有长度大于等于2的时候才有意义
        stack = [0]
        res = 0
        
        for ch in s:
            if ch == '(':
                stack.append(0)
            elif ch == ')':
                if len(stack) > 1:
                    top = stack.pop()
                    stack[-1] += top + 2
                    res = max(res, stack[-1])
                else:
                    stack = [0]
        
        return res
```

#### 33. Search in Rotated Sorted Array
```
class Solution:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return -1
        
        n = len(nums)
        start, end = 0, n - 1
        
        ## 这道题核心：用nums[end]作为判断哪段是递增序列
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return mid
            
            if nums[mid] < nums[start]:
                if nums[mid] < target and nums[end] >= target:
                    start = mid + 1
                else:
                    end = mid - 1
            else:
                if nums[mid] > target and nums[start] <= target:
                    end = mid - 1
                else:
                    start = mid + 1
        
        if nums[start] == target:
            return start
        
        if nums[end] == target:
            return end
        
        return -1
```

#### 34. Find First and Last Position of Element in Sorted Array
```
class Solution:
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
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
        
        ## 小陷阱：注意下面两个if的顺序
        startPos = -1
        if nums[end] == target:
            startPos = end
        
        if nums[start] == target:
            startPos = start
        
        start, end = 0, n - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] <= target:
                start = mid
            else:
                end = mid - 1
        
        endPos = -1
        if nums[start] == target:
            endPos = start
        
        if nums[end] == target:
            endPos = end
        
        if startPos != -1 and endPos != -1:
            return [startPos, endPos]
        
        return [-1, -1]
```

#### 36. Valid Sudoku
```
class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # create 9 empty set, python2-> set([]), python3-> set()
        row = [set() for _ in range(9)]
        col = [set() for _ in range(9)]
        grid = [set() for _ in range(9)]
        
        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    continue
                if board[i][j] in row[i]:
                    return False
                if board[i][j] in col[j]:
                    return False
                
                # for 3*3 grid matrix 
                # if we convert it to 1*9 matrix, index will be i*3+j
                g = i // 3 * 3 + j // 3  
                # for board,we have 3 grids through the row, 
                # 3 grids through the columns
                if board[i][j] in grid[g]: 
                    #so i divide by 3, 
                    # j divided by 3 to get the index for each grid
                    return False         
                        
                row[i].add(board[i][j]) 
                col[j].add(board[i][j])
                grid[g].add(board[i][j])
        
        return True
```

#### 37. Sudoku Solver
```
class Solution:
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if (
            not board
            or not board[0]
            or len(board) != 9
            or len(board[0]) != 9
        ):
            return
        
        self._dfs(board, 0, 0)
        
    def _dfs(self, board, i, j):
        if i == 9:
            return True
        if j == 9:
            return self._dfs(board, i + 1, 0)
        
        if board[i][j] == '.':
            for k in range(1, 10):
                board[i][j] = str(k)
                if self._is_valid(board, i, j) and self._dfs(board, i, j + 1):
                    return True
                board[i][j] = '.'
        else:
            return self._dfs(board, i, j + 1)
        
        return False
    
    def _is_valid(self, board, i, j):
        for col in range(9):
            if col != j and board[i][j] == board[i][col]:
                return False
        
        for row in range(9):
            if row != i and board[i][j] == board[row][j]:
                return False
        
        for row in range(i // 3 * 3, i // 3 * 3 + 3):
            for col in range(j // 3 * 3, j // 3 * 3 + 3):
                if (row != i or col != j) and board[i][j] == board[row][col]:
                    return False
        
        return True
```

#### 38. Count and Say
```
class Solution:
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        if n <= 0:
            return '1'
        # 对于前一个数
        # 找出相同元素的个数
        # 把个数和该元素存到curr里
        # 最后在每次循环里更新res
        # 核心：数相同的数字！(这叫count)
        res = '1'
        i = 0
        for _ in range(n - 1):
            curr = ''
            i = 0
            while i < len(res):
                cnt = 1
                while i + 1 < len(res) and res[i] == res[i + 1]:
                    cnt += 1
                    i += 1
                curr += str(cnt) + res[i]
                i += 1
            res = curr

        return res
```

#### 39. Combination Sum
```
class Solution:
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        
        ret = []
        if not candidates:
            return ret
        
        candidates.sort()
        self.helper(candidates, target, 0, [], ret)
        return ret
    
    
    def helper(self, candidates, target, start, current, results):
        
        if target == 0:
            results.append(current[:])
            return
        
        for i in range(start, len(candidates)):
            if target - candidates[i] < 0:
                break
            if i != 0 and candidates[i] == candidates[i - 1]:
                continue
            current.append(candidates[i])
            self.helper(candidates, target - candidates[i], i, current, results)
            current.pop()
```

#### 41. First Missing Positive
```
class Solution:
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)

        for i in range(n):
            while nums[i] > 0 and nums[i] <= n and nums[i] != nums[nums[i] - 1]:
                # python交换操作
                # 先将等号左边的第一个数字存为temp
                # 然后所有对左边第一个数字的赋值操作（对应右边第一个数字）
                # 是使用temp的
                # 此时对左边第二个数字再进行赋值操作
                # 但是这时候引用的nums[i]是之前变化过的！！！！！
                # 这是个大坑要注意！！！
                # 这道题下面的交换顺序是通不过的！！！
                # nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        return n + 1
```

#### 42. Trapping Rain Water, Hard, Facebook
```
class Solution:
    def trap(self, heights):
        """
        :type height: List[int]
        :rtype: int
        """
        # 这道题和histogram不同
        # 这道题是维护一个单调递减栈
        # 而histogram那道题是维护一个单调递增栈
        stack = []
        res = 0
        i = 0
        while i < len(heights):
            if not stack or heights[i] <= heights[stack[-1]]:
                stack.append(i)
                i += 1
            else:
                curr = stack.pop()
                if not stack:
                    continue
                res += (min(heights[i], heights[stack[-1]]) - heights[curr]) * (i - stack[-1] - 1)
        return res
    
## Two pointers AC
# class Solution:
#     """
#     @param: heights: a list of integers
#     @return: a integer
#     """
#     def trap(self, heights):
#         # write your code here
        
#         if not heights:
#             return 0
        
#         res = 0
#         lp, rp = 0, len(heights) - 1
#         while lp < rp:
#             mn = min(heights[lp], heights[rp])
#             if mn == heights[lp]:
#                 lp += 1
#                 while lp < rp and mn > heights[lp]:
#                     res += mn - heights[lp]
#                     lp += 1
#             else:
#                 rp -= 1
#                 while lp < rp and mn > heights[rp]:
#                     res += mn - heights[rp]
#                     rp -= 1
        
#         return res
```

#### 43. Multiply Strings
```
class Solution:
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        
        len_num1, len_num2 = len(num1), len(num2)
        
        ## 假设num1是长度大的那个字符串
        if len_num1 < len_num2:
            return self.multiply(num2, num1)
        
        # 这道题思路就是先求出一个不带进位的list
        # 这个list的最大长度就是len_num1 + len_num2
        # 不可能更大了
        num3 = [0] * (len_num1 + len_num2) 
        
        for i in range(len_num2 - 1, -1, -1):
            for j in range(len_num1 - 1, -1, -1):
                ## 注意坐标：这里是i + j + 1， 而且是+=
                num3[i + j + 1] += int(num1[j]) * int(num2[i])

        carry = 0
        for k in range(len_num1 + len_num2 - 1, -1, -1):
            temp = carry + num3[k]
            carry = temp // 10
            num3[k] = temp % 10

        left = 0
        while left < len_num1 + len_num2 - 1 and num3[left] == 0:
            left += 1
        
        return ''.join(str(i) for i in num3[left:])
```

#### 44. Wildcard Matching, Hard, Facebook
```
class Solution:
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        
        n = len(s)
        m = len(p)
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        
        if n == 0 and p.count('*') == m:
            return True
        
        # dp[i][j]定义：前i个s里的字符和前j个p里的字符是否match
        # dp[i][j]取决于三个值：dp[i - 1][j - 1], dp[i - 1][j]和dp[i][j - 1]
        # 注意：对于dp来说dp里的i和j是表示"前"几个字符
        # 但是对于s[i - 1]就是代表当前处理的字符
        dp[0][0] = True
        for i in range(n + 1):
            for j in range(m + 1):
                if i > 0 and j > 0:
                    # 前i-1个s中的字符和前j-1个p中的字符已经匹配
                    # 下面只要看p的当前字符p[j - 1]是不是?或者*即可 
                    dp[i][j] |= dp[i - 1][j - 1] and \
                        (s[i - 1] == p[j - 1] or p[j - 1] in ('?', '*'))
                if i > 0 and j > 0:
                    # 前i-1个s中的字符和前j个p中的字符已经匹配
                    # 下面如果还能匹配，p中的当前字符一定必须是*
                    # 在这种情况下p中的这个*就表示可以没有
                    dp[i][j] |= dp[i - 1][j] and p[j - 1] == '*'
                if j > 0:
                    # 前i个s中的字符和前j - 1个p中的字符已经匹配
                    # 下面如果还能匹配，p中的当前字符也一定是*
                    # 在这种情况下p中的这个*就表示可以是任意一个字符
                    # 为什么不可以是下面这样?
                    # dp[i][j] |= dp[i][j - 1] and p[j - 1] in ('?', '*')
                    dp[i][j] |= dp[i][j - 1] and p[j - 1] == '*'
        
        return dp[n][m]
```

#### 45. Jump Game II
```
class Solution:
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        这道题是问最少几步能跳到最后
        贪心法
        """
        steps = 0
        n = len(nums)
        i = curr = 0
        # curr表示当前能跳到的最远距离
        while curr < n - 1:
            steps += 1
            # 因为在下面的循环里要更新curr
            # 所以预先保存下pre
            pre = curr
            while i <= pre:
                # 注意：这道题保证了i一定是nums中valid index
                curr = max(curr, i + nums[i])
                i += 1
            if pre == curr:
                return -1
        
        return steps
```

#### 46. Permutations
```
class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if not nums:
            return []
        
        res = []
        self.dfs(nums, [], res)
        return res
    
    def dfs(self, numbers, current, results):
        
        if not numbers:
            results.append(current[:])
        
        for i in range(len(numbers)):
            current.append(numbers[i])
            self.dfs(numbers[:i] + numbers[i + 1:], current, results)
            current.pop()
```

#### 47. Permutations II
```
class Solution:
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        
        if not nums:
            return []
        
        nums.sort()
        res = []
        self.dfs(nums, [], res)
        return res
    
    def dfs(self, numbers, current, results):
        
        if not numbers:
            results.append(current[:])
            return
        
        for i in range(len(numbers)):
            if i > 0 and numbers[i] == numbers[i - 1]:
                continue
            current.append(numbers[i])
            self.dfs(numbers[:i] + numbers[i + 1:], current, results)
            current.pop()
```

#### 48. Rotate Image
```
class Solution:
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        
        # 先水平翻转（将image水平flip）
        for i in range(row):
            for j in range(i + 1, row):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        # 再垂直翻转（将image垂直flip）
        for i in range(row):
            matrix[i].reverse()
```

#### 49. Group Anagrams
```
from collections import defaultdict

class Solution:
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        ## 用mapping存每一个词的pattern
        mapping = defaultdict(list)
        for each in strs:
            mapping[''.join(sorted(each))].append(each)
        
        res = []
        for _, val in mapping.items():
            res.append(sorted(val))
        
        return res
```

#### 50. Pow(x, n)
```
class Solution:
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """

        if n < 0:
            return 1 / self.myPow(x, -n)
        
        if n == 0:
            return 1
        
        half = self.myPow(x, n // 2)
        if n % 2 == 1:
            return x * half * half
        return half * half
```

#### 51. N-Queens, Hard, Facebook
```
class Solution:
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        res = []
        self._dfs([-1] * n, 0, [], res)
        return res
    
    # positions是每行Q的位置（因为每行只能放一个Q）
    # positions的长度就是一共有多少行
    # curr_row_inx表示当前递归处理第几行
    def _dfs(self, positions, curr_row_inx, path, res):
        total_length = len(positions)
        if curr_row_inx == total_length:
            res.append(path)
            return
        
        for col in range(total_length):
            positions[curr_row_inx] = col
            if self._valid(positions, curr_row_inx):
                temp = '.' * total_length
                self._dfs(
                    positions,
                    curr_row_inx + 1,
                    path + [temp[:col] + 'Q' + temp[col + 1:]],
                    res
                )
    
    def _valid(self, positions, curr_row_inx):
        for i in range(curr_row_inx):
            # 当前行之差等于当前列之差
            # curr_row_inx就是第几行
            # positions[curr_row_inx]就是curr_inx行上第几列
            if curr_row_inx - i == abs(positions[curr_row_inx] - positions[i]) or \
                positions[curr_row_inx] == positions[i]:
                return False
        return True
```

#### 52. N-Queens II
```
class Solution:
    def totalNQueens(self, n):
        """
        :type n: int
        :rtype: int
        """
        self._res = 0
        self._dfs(0, [None] * n)
        return self._res

    def _dfs(self, row, pos):
        n = len(pos)

        if row == n:
            self._res += 1
            return
        
        for col in range(n):
            if self._is_valid(row, col, pos):
                pos[row] = col
                self._dfs(row + 1, pos)
                pos[row] = None
    
    def _is_valid(sefl, row, col, pos):
        for i in range(row):
            if pos[i] == col or abs(row - i) == abs(col - pos[i]):
                return False
        
        return True
```

#### 53. Maximum Subarray
```
class Solution:
    INT_MIN = -2147483648
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        ## 读题意：这道题是要返回最大的和，不是最大和的子序列长度！！！
        
        if not nums:
            return self.INT_MIN
        
        localMax = globalMax = self.INT_MIN
        for num in nums:
            localMax = max(localMax + num, num)
            globalMax = max(globalMax, localMax)
        
        return globalMax
```

#### 55. Jump Game
```
class Solution:
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # reach表示当前能到达的最远坐标
        reach = 0
        n = len(nums)
        for i in range(n):
            if i > reach or reach >= n - 1:
                break
            # i就是当前位置
            # 加上在i位置的max dump就是从i开始能走多远
            # reach是全局的最大坐标
            reach = max(reach, i + nums[i])
        
        return reach >= n - 1
```

#### 56. Merge Intervals
```
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

from functools import cmp_to_key

def cmp(a, b):
    if a.start != b.start:
        return a.start - b.start
    else:
        return a.end - b.end

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        
        if not intervals:
            return []
        
        intervals.sort(key=cmp_to_key(cmp))
        res = []
        for each in intervals:
            if not res:
                res.append(each)
                continue
            ## 只有两种情况需要往res里添加或者修改新的interval
            
            ## 一是当前有新间隔了（需要添加新间隔）
            if each.start > res[-1].end:
                res.append(each)
            ## 二是当前结尾的间隔大于之前结尾的间隔（需要修改上一个间隔的结束时间）
            else:
                if each.end > res[-1].end:
                    last = res.pop()
                    res.append(Interval(last.start, each.end))
        
        return res
```

#### 57. Insert Interval, Hard, Facebook, Linkedin
```
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        
        if not intervals:
            return [newInterval]
        
        res = []
        insertPos = 0
        n = len(intervals)
        for i in range(n):
            ## 说明newInterval在current interval的左边（不重叠）
            if intervals[i].start > newInterval.end:
                res.append(intervals[i])
            ## 说明newInterval在current interval的右边（不重叠）
            elif intervals[i].end < newInterval.start:
                ## 注意：这个地方应该是+=1,而不是i,因为i是不准的!!!!!
                insertPos += 1
                res.append(intervals[i])
            ## 说明newInterval和current interval有重叠
            ## 在这种情况下重新update newInterval
            else:
                newInterval.start = min(newInterval.start, intervals[i].start)
                newInterval.end = max(newInterval.end, intervals[i].end)
            
        res.insert(insertPos, newInterval)
        return res
```

#### 65. Valid Number, Hard, Facebook, Linkedin
```
class Solution:
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """
        
        s = s.strip()
        n = len(s)
        num = dot = expr = sign = False
        allow_e = True
        
        # 当前字符串经过trim以后，每个字符有6种情况：
        # 1. 空格（不可能是首尾位置，因为已经trim过了）
        # 2. 数字
        # 3. 小数点
        # 4. e
        # 5. 加减号
        # 6. 其他，直接返回False
        # 口诀：数点e sign （15.13）
        for i in range(n):
            if s[i] == ' ':
                if num or dot or expr or sign:
                    return False
            elif '0' <= s[i] <= '9':
                num = True
                allow_e = True
            elif s[i] == '.':
                if dot or expr:
                    return False
                dot = True
            elif s[i] == 'e':
                if expr or not num:
                    return False
                expr = True
                allow_e = False
            elif s[i] == '+' or s[i] == '-':
                if i > 0 and s[i - 1] != 'e':
                    return False
                sign = True
            else:
                return False
        
        return num and allow_e
```

#### 66. Plus One
```
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] + carry == 10:
                digits[i] = 0
            else:
                digits[i] += carry
                carry = 0
                break
        
        if carry == 1:
            digits.insert(0, 1)
        
        return digits
```

#### 67. Add Binary
```
class Solution:
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        two binary strings, return their sum
        """
        res = ''
        pa = len(a) - 1
        pb = len(b) - 1
        carry = 0
        
        while pa >= 0 or pb >= 0:
            da = 1 if pa >= 0 and int(a[pa]) == 1 else 0
            db = 1 if pb >= 0 and int(b[pb]) == 1 else 0
            ab_sum = da + db + carry
            res = str(ab_sum % 2) + res
            carry = ab_sum // 2
            pa -= 1
            pb -= 1
        
        if carry == 1:
            res = '1' + res
        
        return res
```

#### 68. Text Justification, Hard, Facebook, Linkedin
```
class Solution:
    ## 抄的九章答案
    ## http://www.jiuzhang.com/solution/text-justification/#tag-highlight-lang-python
    def _format(self, line, maxWidth):
        if len(line) == 1:
            return line[0] + " " * (maxWidth - len(line[0]))
        length = sum([len(w) for w in line])
        s, gaps = line[0], len(line) - 1
        for index, w in enumerate(line[1:]):
            if index < (maxWidth - length) % gaps:
                s = s + " " + " " * ((maxWidth - length) // gaps) + w
            else:
                s = s + " " * ((maxWidth - length) // gaps) + w
        return s
        
        
    def _formatLast(self, line, maxWidth):
        s = ' '.join(line)
        return s + " " * (maxWidth - len(s))

    
    def fullJustify(self, words, maxWidth):
        """
        :type words: List[str]
        :type maxWidth: int
        :rtype: List[str]
        """
        line, length = [], 0
        results = []
        for w in words:
            if length + len(w) + len(line) <= maxWidth:
                length += len(w)
                line.append(w)
            else:
                results.append(self._format(line, maxWidth))
                length = len(w)
                line = [w]
        if len(line):
            results.append(self._formatLast(line, maxWidth))
        return results
```

#### 69. Sqrt(x)
```
class Solution:
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        
        start, end = 0, x
        while start + 1 < end:
            mid = start + (end - start) // 2
            if mid * mid == x:
                return mid
            elif mid * mid < x:
                start = mid
            else:
                end = mid
        
        if end * end <= x:
            return end
        
        return start
```

#### 71. Simplify Path
```
class Solution:
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        For example,
        path = "/home/", => "/home"
        path = "/a/./b/../../c/", => "/c"
        path = "/a/../../b/../c//.//", => "/c"
        path = "/a//b////c/d//././/..", => "/a/b/c"
        """

        res = []
        i = 0
        
        while i < len(path):
            end = i + 1
            while end < len(path) and path[end] != '/':
                end += 1
            sub_str = path[i + 1:end]
            if sub_str:
                if sub_str == '..':
                    if res:
                        res.pop()
                elif sub_str != '.':
                    res.append(sub_str)
            i = end
        
        if not res:
            return '/'
        
        return '/' + '/'.join(res)
```

#### 72. Edit Distance, Hard, Linkedin
```
class Solution:
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        
        if not word1:
            return len(word2)
        
        if not word2:
            return len(word1)
        
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j],
                        dp[i][j - 1],
                        dp[i - 1][j - 1],
                    ) + 1
        
        return dp[-1][-1]
```

#### 73. Set Matrix Zeroes
```
class Solution:
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        Given a m x n matrix, if an element is 0, 
        set its entire row and column to 0. Do it in-place.
        """
        if not matrix or not matrix[0]:
            return
        
        # 因为要确定第一行（第一列）的零到底是由于中间
        # 元素变成的零还是原来就有的零
        # 所以开始就记录一下
        # 这样如果是开始就有零
        # 在最后就需要将第一行（第一列）全部置零
        rows, cols = len(matrix), len(matrix[0])
        empty_row = empty_col = False
        
        for i in range(cols):
            if matrix[0][i] == 0:
                empty_row = True
                break
        
        for i in range(rows):
            if matrix[i][0] == 0:
                empty_col = True
                break
        
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        if empty_row:
            for i in range(cols):
                matrix[0][i] = 0
        
        if empty_col:
            for i in range(rows):
                matrix[i][0] = 0
```

#### 74. Search a 2D Matrix
```
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        Integers in each row are sorted from left to right.
        The first integer of each row is greater than the last integer 
        of the previous row.
        """
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        row, col = m - 1, 0
        while row >= 0 and col < n:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                row -= 1
            else:
                col += 1
        
        return False
```

#### 75. Sort Colors
```
class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        0, 1, 2
        """
        # 三根指针 在大循环中遍历one_pos
        # zero_pos含义是0的最后一个位置
        # two_pos含义是2的第一个位置
        # one_pos含义是1的第一个位置
        zero_pos = -1
        one_pos = 0
        two_pos = len(nums)

        while one_pos < two_pos:
            if nums[one_pos] == 0:
                zero_pos += 1
                nums[zero_pos], nums[one_pos] = nums[one_pos], nums[zero_pos]
                one_pos += 1
            elif nums[one_pos] == 1:
                one_pos += 1
            # 此时nums[one_pos] == 2
            else:
                two_pos -= 1
                nums[one_pos], nums[two_pos] = nums[two_pos], nums[one_pos]
```

#### 76. Minimum Window Substring, Hard, Facebook, Linkedin
```
class Solution:
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        Given a string S and a string T,
        find the minimum window in S which will contain all the characters in T 
        in complexity O(n).
        Input: S = "ADOBECODEBANC", T = "ABC"
        Output: "BANC"
        """

        ## 最后一个test case通不过
        ## TLE
        
        if not s or len(t) > len(s):
            return ''
        
        sHash = [0] * 128
        tHash = [0] * 128
        
        for ch in t:
            tHash[ord(ch)] += 1
        
        minLen = len(s)
        left_index = right_index = 0
        right = 0
        
        for left in range(len(s)):
            while right < len(s):
                if not self.valid(sHash, tHash):
                    sHash[ord(s[right])] += 1
                    right += 1
                else:
                    break
            if self.valid(sHash, tHash):
                if minLen > right - left - 1:
                    left_index, right_index = left, right - 1
                    minLen = right - left - 1
        
            sHash[ord(s[left])] -= 1
        
        return s[left_index:right_index + 1] if minLen < len(s) else ''
    
    
    def valid(self, s, t):
        
        for i in range(128):
            if s[i] < t[i]:
                return False
        
        return True


    ## 九章解法能通过
    
    # def minWindow(self, source, target):
    #     if (target == ""):
    #         return ""
    #     S , T = source, target
    #     d, dt = {}, dict.fromkeys(T, 0)
    #     for c in T: d[c] = d.get(c, 0) + 1
    #     pi, pj, cont = 0, 0, 0
    #     if (source =="" or target ==""):
    #         return ""
    #     ans = ""
    #     while pj < len(S):
    #         if S[pj] in dt:
    #             if dt[S[pj]] < d[S[pj]]:
    #                 cont += 1
    #             dt[S[pj]] += 1;
    #         if cont == len(T):
    #             while pi < pj:
    #                 if S[pi] in dt:
    #                     if dt[S[pi]] == d[S[pi]]:
    #                         break;
    #                     dt[S[pi]] -= 1;
    #                 pi+= 1
    #             if ans == '' or pj - pi < len(ans):
    #                 ans = S[pi:pj+1]
    #             dt[S[pi]] -= 1
    #             pi += 1
    #             cont -= 1
    #         pj += 1
    #     return ans
```

#### 77. Combinations
```
class Solution:
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        1到n中选取不重复的k个数字凑成combination
        """
        res = []
        self._dfs(n, k, 0, [], res)
        return res
    
    # df定义：从start开始到n，在现有curr基础上遍历
    # 往curr里添加数字
    def _dfs(self, n, k, start, curr, res):
        if len(curr) == k:
            res.append(curr[:])
            return
        
        for i in range(start + 1, n + 1):
            curr.append(i)
            self._dfs(n, k, i, curr, res)
            curr.pop()
```

#### 78. Subsets
```
class Solution:
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        
        res = []
        self.dfs(nums, 0, [], res)
        return res
    
    def dfs(self, nums, start, current, results):
        
        results.append(current[:])
        
        if start == len(nums):
            return
        
        for i in range(start, len(nums)):
            current.append(nums[i])
            self.dfs(nums, i + 1, current, results)
            current.pop()
```

#### 79. Word Search
```
class Solution:
    
    _DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        
        if not board or not board[0]:
            return False
        
        if not word:
            return True

        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]

        # 这个corner case开始就没考虑到！！！
        # 说明下面的DFS并没有cover住这个case
        if m == 1 and n == 1 and len(word) == 1:
            return board[0][0] == word[0]

        for i in range(m):
            for j in range(n):
                if self._dfs(board, word, i, j, visited):
                    return True
        
        return False
    
    
    def _dfs(self, board, word, x, y, visited):
        if not word:
            return True
        
        for di, dj in self._DIRECTIONS:
            newi, newj = di + x, dj + y
            if 0 <= newi < len(board) and \
                0 <= newj < len(board[0]) and \
                not visited[newi][newj] and \
                board[newi][newj] == word[0]:
                visited[newi][newj] = True
                if self._dfs(board, word[1:], newi, newj, visited):
                    return True
                visited[newi][newj] = False
        
        return False
```

#### 81. Search in Rotated Sorted Array II
```
class Solution:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        有重复元素
        """
        
        if not nums:
            return False
        
        n = len(nums)
        start, end = 0, n - 1
        
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] == target:
                return True
            
            if nums[mid] == nums[start]:
                start += 1
            elif nums[mid] > nums[start]:
                if nums[mid] > target >= nums[start]:
                    end = mid - 1
                else:
                    start = mid
            else:
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid
        
        if nums[start] == target or nums[end] == target:
            return True
        
        return False
```

#### 84. Largest Rectangle in Histogram, Hard, Facebook
```
class Solution:
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        # 核心：维护一个单调递增栈
        # 栈中存的是heights数组中的index
        # 当遇见比当前栈顶（最后一位）小的值的时候
        # 触发出栈操作，维护单调递减栈的性质
        heights.append(0)
        stack = [-1]
        res = 0
        for i in range(len(heights)):
            # 这里挺tricky的
            # 因为初始为了我们对heights append了一个0
            # 即heights[-1] == 0
            # 所以在第一次循环时候heights[i]一定是大于等于heights[-1]的
            # 因为题目指定了non-negative
            # 所以第一次循环会跳过while直接对stack append操作
            while heights[i] < heights[stack[-1]]:
                h = heights[stack.pop()]
                w = i - stack[-1] - 1
                res = max(res, h * w)
            stack.append(i)
        
        return res
```

#### 85. Maximal Rectangle
```
class Solution:
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        # 一行一行的生成直方图
        # 然后在每次的循环中调用直方图里的方法
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        res = 0
        height = [0] * n
        for i in range(m):
            for j in range(n):
                height[j] = 0 if matrix[i][j] == '0' else 1 + height[j]
            res = max(res, self._helper(height))
        
        return res
    
    # 84题最大直方图
    def _helper(self, height):
        # 这里用的是一个单调递增栈
        stack = [-1]
        height.append(0)
        res = 0
        for i in range(len(height)):
            while height[i] < height[stack[-1]]:
                h = height[stack.pop()]
                w = i - stack[-1] - 1
                res = max(res, h * w)
            stack.append(i)
        height.pop()
        return res
```

#### 87. Scramble String
```
class Solution:
    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if s1 == s2:
            return True
        
        if len(s1) != len(s2):
            return False
        
        n = len(s1)
        counts = [0] * 26
        for i in range(n):
            counts[ord(s1[i]) - ord('a')] += 1
            counts[ord(s2[i]) - ord('a')] -= 1
        for each_count in counts:
            if each_count != 0:
                return False
        
        for i in range(1, n):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                return True
            if self.isScramble(s1[:i], s2[n - i:]) and self.isScramble(s1[i:], s2[:n - i]):
                return True
        
        return False
```

#### 88. Merge Sorted Array
```
class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i = m - 1
        j = n - 1
        k = m + n - 1
        
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        
        # 此时nums1中的数字先使用完了（说明nums1中大的数字比较多）
        # nums2还没使用完
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
```

#### 91. Decode Ways
```
class Solution:
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        这里实际上是decode way II的解法
        """
        
        if not s:
            return 1
        
        if s[0] == '0':
            return 0

        n = len(s)
        # Python 1e9是一个float number
        M = int(1e9 + 7)
        # dp[i]定义：前i个字符有多少种解码方式
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 9 if s[0] == '*' else 1
        
        for i in range(2, n + 1):
            if s[i - 1] == '0':
                # 前一个有一个固定选择，就延续加一种
                if s[i - 2] == '1' or s[i - 2] == '2':
                    dp[i] = dp[i - 2]
                # 当前一个是*，当前就有两种选择
                elif s[i - 2] == '*':
                    dp[i] = 2 * dp[i - 2]
                # 因为单独一个0是invalid （A=1, ...Z=26）
                else:
                    return 0
            elif '1' <= s[i - 1] <= '9':
                dp[i] = dp[i - 1]
                if (s[i - 2] == '1') or (s[i - 2] == '2' and '0' <= s[i - 1] <= '6'):
                    dp[i] += dp[i - 2]
                elif s[i - 2] == '*':
                    # 此时s[i - 2]可以有两种选择，可以为1也可以为2
                    if '0' <= s[i - 1] <= '6':
                        dp[i] += 2 * dp[i - 2]
                    # 但是当s[i - 1]为大于6的数字的时候
                    # s[i - 2]只剩一种选择，就是1
                    else:
                        dp[i] += dp[i - 2]
            else: # now s[i - 1] == '*'
                dp[i] = 9 * dp[i - 1]
                if s[i - 2] == '1':
                    dp[i] += 9 * dp[i - 2]
                elif s[i - 2] == '2':
                    dp[i] += 6 * dp[i - 2]
                # 当**的时候，此时dp[i]的结果就应该是上面两种情况的和
                elif s[i - 2] == '*':
                    dp[i] += 15 * dp[i - 2]
            dp[i] %= M
        
        return dp[n]        
```

#### 92. Reverse Linked List II
```
class Solution:
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if m == n:
            return head
        
        dummy_node = ListNode(-1)
        dummy_node.next = head
        
        pre = dummy_node
        for i in range(m - 1):
            pre = pre.next
        
        reverse = None
        curr = pre.next
        i = 0
        while i < n - m + 1:
            temp = curr.next
            curr.next = reverse
            reverse = curr
            curr = temp
            i += 1
        
        # 这里不太清晰
        # 实际上，此时pre.next还没有被改变
        # 它应该指向的是反转后的最右边（即现在的cur）
        pre.next.next = curr
        # 这里把pre的next指向了反转后的头
        pre.next = reverse
        
        return dummy_node.next
```

#### 94. Binary Tree Inorder Traversal
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        
        res = []
        stack = []
        curr = root
        
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            top = stack.pop()
            res.append(top.val)
            curr = top.right
        
        return res
```

#### 97. Interleaving String
```
class Solution:
    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        这道题是问s3是不是s1和s2保持序列的循序交织在一起的字符串
        """
        # dp[i][j]定义：
        # s1的前i个字符和s2的前j个字符能否组成s3的前i+j个字符
        # 递推公式：
        # dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[i - 1 + j]) or (dp[i][j - 1] and s2[j - 1] == s3[j - 1 + i])
        
        len_s1 = len(s1)
        len_s2 = len(s2)
        len_s3 = len(s3)
        if len_s1 + len_s2 != len_s3:
            return False
        
        dp = [[False] * (len_s2 + 1) for _ in range(len_s1 + 1)]
        dp[0][0] = True
        
        for i in range(1, len_s1 + 1):
            dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
        
        for i in range(1, len_s2 + 1):
            dp[0][i] = dp[0][i - 1] and s2[i - 1] == s3[i - 1]

        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                dp[i][j] = (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1]) or (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
        
        return dp[-1][-1]
```

#### 98. Validate Binary Search Tree
```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        infinity = 10**10
        return self.isValid(root, -infinity,infinity)
    
    def isValid(self, root,minvalue,maxvalue):
        if root is None:
            return True
        
        if not minvalue < root.val < maxvalue:
            return False
        
        return self.isValid(root.left, minvalue, root.val) and \
            self.isValid(root.right,root.val, maxvalue)
```

#### 99. Recover Binary Search Tree, Hard, Facebook
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# Two elements of a binary search tree (BST) are swapped by mistake.

class Solution:
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        self._first = None
        self._second = None
        self._prev = TreeNode(-2 ** 31)
        
        self._in_order(root)
        self._first.val, self._second.val = self._second.val, self._first.val
    
    def _in_order(self, root):
        if not root:
            return
        
        self._in_order(root.left)
        
        # If first element has not been found
        # assign it to _prev
        # 正常按照中序遍历前一个节点的值应该小于当前节点的值
        # 如果发现前一个节点的值大于当前节点的值
        # 就说明我们发现了第一个不符合条件的节点（self._first）
        if self._first is None and self._prev.val >= root.val:
            self._first = self._prev
        # If first element is found
        #   assign the second element to the root
        # 不好理解的地方就在这里
        # 因为我们已经找到了第一个节点
        # 而且根据题意这个树里有且只有一对儿（两个节点的值）是需要调整的
        # 所以毫无疑问此时如果再次发现一个需要调整的节点
        # 这个节点就是所谓的self._second
        if self._first is not None and self._prev.val >= root.val:
            self._second = root
        self._prev = root
        
        self._in_order(root.right)
```

#### 100. Same Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        
        if p and not q:
            return False
        
        if q and not p:
            return False
        
        if not p and not q:
            return True
        
        if p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) and \
            self.isSameTree(p.right, q.right)
```

#### 101. Symmetric Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root):
        # Write your code here
        if not root:
            return True
        
        return self.helper(root.left, root.right)
    
    
    def helper(self, left, right):
        
        if left and not right:
            return False
        
        if not left and right:
            return False
        
        if not left and not right:
            return True

        if left.val != right.val:
            return False
        
        return self.helper(left.left, right.right) and \
            self.helper(left.right, right.left)
```

#### 102. Binary Tree Level Order Traversal
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        
        if not root:
            return []
        
        queue = [root]
        res = []
        while queue:
            qLen = len(queue)
            temp = []
            for _ in range(qLen):
                curr = queue.pop(0)
                temp.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            res.append(temp)
        
        return res
```

#### 103. Binary Tree Zigzag Level Order Traversal
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        
        if not root:
            return []
        
        queue = [root]
        res = []
        reverse = False
        while queue:
            qLen = len(queue)
            temp = []
            for _ in range(qLen):
                curr = queue.pop(0)
                temp.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)

            if not reverse:
                res.append(temp)
            else:
                res.append(temp[::-1])
            reverse = not reverse
            
        return res
```

#### 104. Maximum Depth of Binary Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if not root:
            return 0
        
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

#### 105. Construct Binary Tree from Preorder and Inorder Traversal
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        return self._helper(deque(preorder), inorder)
    
    def _helper(self, preorder, inorder):
        if not inorder:
            return None
        
        root_val = preorder.popleft()
        root = TreeNode(root_val)
        
        index = inorder.index(root_val)
        root.left = self._helper(preorder, inorder[:index])
        root.right = self._helper(preorder, inorder[index + 1:])
        
        return root
```

#### 106. Construct Binary Tree from Inorder and Postorder Traversal
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if not inorder:
            return
        
        root = TreeNode(postorder[-1])
        split_index = inorder.index(postorder[-1])
        
        # 核心！！！
        # 中序： [...A, x, ...B]
        # 后序： [...A, ...B, x]
        # 即后序的前半部分就是中序的前半部分
        # 后序的后半部分（除去最后一个点）就是中序的后半部分
        root.left = self.buildTree(inorder[:split_index], postorder[:split_index])
        root.right = self.buildTree(inorder[split_index + 1:], postorder[split_index:-1])
        
        return root
```

#### 109. Convert Sorted List to Binary Search Tree
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return head
        
        if not head.next:
            return TreeNode(head.val)
        
        prev = mid = fast = head
        while fast and fast.next:
            prev = mid
            mid = mid.next
            fast = fast.next.next
        
        if prev:
            prev.next = None
        
        root = TreeNode(mid.val)
        root.left = self.sortedListToBST(head)
        # 这里的mid是左边链表的最后一位
        # 所以要用mid.next来表示右边链表的开始位置
        root.right = self.sortedListToBST(mid.next)
        
        return root
```

#### 110. Balanced Binary Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        
        left_height = self.get_max_height(root.left)
        right_height = self.get_max_height(root.right)
        
        return -1 <= left_height - right_height <= 1 and \
            self.isBalanced(root.left) and \
            self.isBalanced(root.right)
    
    # 树的高度就是左右的最大高度再加1
    def get_max_height(self, root):
        
        if not root:
            return 0
        
        return max(
            self.get_max_height(root.left),
            self.get_max_height(root.right),
        ) + 1
```

#### 111. Minimum Depth of Binary Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        if not root:
            return 0
        
        # 左边最小和右边最小中的最小值，再加1即最终的答案
        return self.get_min(root)
    
    def get_min(self, root):
        
        if not root:
            return 2 ** 31
        
        if not root.left and not root.right:
            return 1
        
        return 1 + min(self.get_min(root.left), self.get_min(root.right))

# 解法2：跟上面类似，但是简洁一些
class Solution:
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 这道题还可以用层序遍历来解
        # 如果访问到一个叶子节点
        # 则直接返回当前的深度
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

#### 113. Path Sum II
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        self._helper(root, sum, [], res)
        return res
    
    def _helper(self, node, sum, curr, res):
        if not node:
            return
        
        curr.append(node.val)
        
        if sum == node.val and not node.left and not node.right:
            res.append(curr[:])
        
        self._helper(node.left, sum - node.val, curr, res)
        self._helper(node.right, sum - node.val, curr, res)
        
        curr.pop()
```

#### 114. Flatten Binary Tree to Linked List
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        # 这道题是先序遍历
        # 第一个if： 首先在root的左子树中找到最右节点node
        # 然后将root右子树当成找到的node的右子树
        # 这样root的左子树就处理完了
        # 第二个if： 将处理完的root的左子树当成root的右子树
        # 最后循环处理root的右子树
        # 例子：
        # A：
        #              1
        #             / \
        #            2   5
        #           / \   \
        #          3   4   6
        # B：
        #            1
        #             \
        #              2
        #             / \
        #            3   4
        #                 \
        #                  5
        #                   \
        #                    6
        # C：  
        #            1
        #             \
        #              2
        #               \
        #                3
        #                 \
        #                  4
        #                   \
        #                    5
        #                     \
        #                      6
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

#### 115. Distinct Subsequences
```
class Solution:
    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        # 典型dp题
        ns, nt = len(s), len(t)
        dp = [[0] * (ns + 1) for _ in range(nt + 1)]
        for i in range(ns + 1):
            # 当t为空串时候
            # 则不管s是什么答案都是1
            # 因为空串是任意字符串的一个子序列
            dp[0][i] = 1
        for i in range(1, nt + 1):
            # 当s为空串的时候
            # 除了第一个位置，剩下的答案都是0
            # 因为任何非空字符串都不可能是空串的子序列
            dp[i][0] = 0
        
        for i in range(1, nt + 1):
            for j in range(1, ns + 1):
                dp[i][j] = dp[i][j - 1]
                if t[i - 1] == s[j - 1]:
                    dp[i][j] += dp[i - 1][j - 1]
        
        return dp[-1][-1]
```

#### 116. Populating Next Right Pointers in Each Node
```
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        """
        Populate each next pointer to point to its next right node.
        If there is no next right node, the next pointer should be set to NULL.
        """
        if not root:
            return
        # 注意这道题能这么做很重要的一个前提是这个树是一个完全树！！！
        # 否则就得考虑层序遍历
        # 最优解，不需要额外的空间
        start = root
        curr = None
        while start.left:
            # 实际上每个while循环就是curr的层序
            # 每个while循环开始的curr都是从每层的最左边开始
            curr = start
            while curr:
                # 核心之一：在同一棵树上处理，将curr为根的树的左子树指向右子树
                curr.left.next = curr.right
                if curr.next:
                    # 核心之二：在兄弟树上处理，当前以curr为根的树的右子树的next指向curr兄弟树的左孩子
                    curr.right.next = curr.next.left
                curr = curr.next
            start = start.left
```

#### 117. Populating Next Right Pointers in Each Node II
```
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        """
        这道题不一定是完全树了
        """
        # 这道题并没有要求每一层的最右节点的next要指向下一层的最左节点
        # 让next指针保持初始值（None）即可
        while root:
            dummy_head = TreeLinkNode(-1)
            curr = dummy_head
            # 隐含着是从第二层开始的！！
            # 这道题实际上虽然while里的是root
            # 但是做事情的是root的下一层！！！
            # 用 2 - 1 - 3为例
            # root为1
            # 但是dummy_head实际上是第二层2 3的dummy_head
            # curr走的是2到3
            # 最终dummy_head的next就是2
            # 本次循环结束
            # 再将root指向2
            while root:
                if root.left:
                    curr.next = root.left
                    curr = curr.next
                if root.right:
                    curr.next = root.right
                    curr = curr.next
                root = root.next
            root = dummy_head.next
```

#### 118. Pascal's Triangle
```
class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        res = []
        if numRows <= 0:
            return res
        
        res.append([1])
        
        for i in range(1, numRows):
            temp = []
            temp.append(1)
            for j in range(1, i):
                temp.append(res[i - 1][j - 1] + res[i - 1][j])
            temp.append(1)
            res.append(temp)
        
        return res
```

#### 121. Best Time to Buy and Sell Stock
```
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        Say you have an array for which the ith element 
        is the price of a given stock on day i.
        If you were only permitted to complete at most 
        one transaction (i.e., buy one and sell one share of the stock), 
        design an algorithm to find the maximum profit. Note that you 
        cannot sell a stock before you buy one.
        这道题是最多只能买卖一次
        """
        ## 在当前天找到在当前天之前的最低价格，再update一下result
        if not prices:
            return 0
        
        global_min = 2 ** 31 - 1
        res = 0
        for i in range(1, len(prices)):
            global_min = min(global_min, prices[i - 1])
            res = max(res, prices[i] - global_min)
        
        return res
```

#### 122. Best Time to Buy and Sell Stock II
```
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        这道题总体可以买进卖出多次
        是指多天里可以多次
        每一天最多只有一次买入一次卖出
        因为多了没必要
        """
        profit = 0
        if len(prices) <= 1:
            return profit
        
        for i in range(1, len(prices)):
            # 典型贪心问题
            # 只要当天比前一天价格低
            # 就交易能赚一笔钱
            if prices[i] - prices[i - 1] > 0:
                profit += prices[i] - prices[i - 1]
        
        return profit
```

#### 123. Best Time to Buy and Sell Stock III, Hard, Facebook
***
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions.

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).
***
```
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
    """
    @param: prices: Given an integer array
    @return: Maximum profit
    """
    ## 股票类型题目一次交易(transaction)意思是：
    ## A. 买入一只股票
    ## 或者
    ## B. 卖出一只股票
    ## 在本题中，每天是可以两笔交易的
    ## 所以需要分别考虑每天买入或者卖出两种情况
    ## 其中，第i天卖出则需要在i天之前必须已经买入过这只股票
    ## 所以卖出应该对应的是[0, i],因此需要从左到右遍历
    ## 而买入则相反，需要在未来某天卖出才能获益
    ## 所以买入对应的是在[i, len(prices) - 1]内的收益，因此从右往左
    ## 所以定义previous和future:
    ##  1. previous[i] 表示在第i天<<卖出>>的话，
    ##     在[0, i]范围内能够获得的最大收益
    ##  2. future[i] 表示在第i天<<买入>>的话，
    ##     在[i, len(prices) - 1]范围能能够获得的最大收益
    ## 这道题是两个dp数组的问题
    def maxProfit(self, prices):
        # write your code here
        n = len(prices)
        
        if n == 0:
            return 0
        
        previous = [0 for _ in range(n)]
        future = [0 for _ in range(n)]
        
        min_in_past = prices[0]
        max_in_future = prices[-1]
        
        ## 卖出从左到右遍历时一定是从第二天，因为第一天没有东西可卖
        ## 当然第一天也可以左手买入右手卖出，收益为0
        for i in range(1, n):
            min_in_past = min(min_in_past, prices[i])
            previous[i] = max(previous[i - 1], prices[i] - min_in_past)
        
        ## 同理，考虑买入情况时不需要考虑最后一天
        ## 因为最后一天买入会亏本（不会再有机会卖出了）
        for j in range(n - 2, -1, -1):
            max_in_future = max(max_in_future, prices[j])
            future[j] = max(future[j + 1], max_in_future - prices[j])
            
        ## 最后，综合考虑每天，看看previous[k]和future[k]和在哪一天最大
        ## 注意: 这里的previous[k]和future[k]已经分别包含了在k天不买入或者不卖出的情况
        res = 0
        for k in range(n):
            res = max(res, previous[k] + future[k])
        
        return res
```

#### 124. Binary Tree Maximum Path Sum, Hard, Facebook
```
class Solution:
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.res = -2 ** 31
        self._helper(root)
        return self.res
    
    # _helper函数返回的是以当前node为根
    # 到叶子节点的最大path sum
    def _helper(self, node):
        if not node:
            return 0

        # 这道题的核心就是要判定是否要进入左右子树
        # 如果当前子树的和已经为负值了
        # 就不要进入该子树
        left = max(self._helper(node.left), 0)
        right = max(self._helper(node.right), 0)
        
        self.res = max(self.res, left + right + node.val)
        return max(left, right) + node.val
```

#### 125. Valid Palindrome
```
class Solution:
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if not s:
            return True
    
        left, right = 0, len(s) - 1
        
        while left < right:
            while left < right and not (s[left].isalpha() or s[left].isdigit()):
                left += 1
            while left < right and not (s[right].isalpha() or s[right].isdigit()):
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1
        
        return True
```

#### 126. Word Ladder II, Hard, Facebook
```
from collections import deque
from collections import defaultdict

class Solution:
    def findLadders(self, start, end, wordList):
        """
        :type start: str
        :type end: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        word_set = set(wordList)
        word_set.add(start)
        
        remove_one_char = self._remove_one_char_mapping(word_set)
        
        # distance to begin word
        dist = {}
        self._bfs(end, start, dist, remove_one_char)
        
        if start not in dist:
            return []
        
        res = []
        self._dfs(start, end, dist, remove_one_char, [start], res)
        return res

    # bfs做的事情是更新每个在word_set里的词到begin word的距离
    # 存到distance字典中
    def _bfs(self, start, end, dist, remove_one_char):
        dist[start] = 0
        queue = deque()
        queue.append(start)
        while queue:
            word = queue.popleft()
            for next_word in self._get_next_word(word, remove_one_char):
                if next_word not in dist:
                    dist[next_word] = dist[word] + 1
                    queue.append(next_word)
                    
    def _dfs(self, start, end, dist, remove_one_char, curr_path, res):
        if start == end:
            res.append(curr_path[:])
            return
        for word in self._get_next_word(start, remove_one_char):
            if dist[word] != dist[start] - 1:
                continue
            curr_path.append(word)
            self._dfs(word, end, dist, remove_one_char, curr_path, res)
            curr_path.pop()

    # 任意一个word替换掉一个字符都对应自身
    # 相当于变换的距离为1
    def _remove_one_char_mapping(self, word_set):
        remove_one_char = defaultdict(set)
        for word in word_set:
            for i in range(len(word)):
                remove_one_char[word[:i] + '%' + word[i + 1:]].add(word)
        return remove_one_char
    
    def _get_next_word(self, word, remove_one_char):
        next_words = set()
        for i in range(len(word)):
            for next_word in remove_one_char[word[:i] + '%' + word[i + 1:]]:
                next_words.add(next_word)
        return next_words
```

#### 127. Word Ladder
```
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """

        wordSet = set(wordList)
        n = len(beginWord)
        
        queue = [(beginWord, 1)]
        while queue:
            curr, count = queue.pop(0)
            if curr == endWord:
                return count
            for i in range(n):
                for ch in 'abcdefghijklmnopqrtsuvwxyz':
                    newWord = curr[:i] + ch + curr[i + 1:]
                    if newWord in wordSet:
                        wordSet.remove(newWord)
                        queue.append((newWord, count + 1))
        
        return 0
```

#### 128. Longest Consecutive Sequence, Hard, Facebook
```
class Solution:
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ## hash表中每次被访问到的数字代表是该数字在全局情况下被cover住
        ## 所以说每次只要该数字被访问过就可以被删除，这样下次就不会被重复访问
        ## 这种思路很重要
        
        if not nums:
            return 0
        
        hash = set(nums)
        res = 0
        for i in range(len(nums)):
            curr = nums[i]
            ## 如果curr不在当前的hash里
            ## 就说明这个curr一定之前是被访问过的(所以才会被remove掉)
            ## 注意：hash里只会存nums里的数字，所以remove就意味着
            ## 不再访问了
            if curr not in hash:
                continue
            
            hash.remove(curr)
            
            asc = curr + 1
            while asc in hash:
                hash.remove(asc)
                asc += 1
                
            desc = curr - 1
            while desc in hash:
                hash.remove(desc)
                desc -= 1
            res = max(res, asc - desc - 1)
        
        return res
```

#### 129. Sum Root to Leaf Numbers
***
Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.

An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers.
***
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self._dfs(root, 0)
    
    def _dfs(self, node, curr_sum):
        if not node:
            return 0
        
        # 重点之一
        curr_sum = 10 * curr_sum + node.val
        
        if not node.left and not node.right:
            return curr_sum
        
        # 需要把当前的curr_sum带入到左右子树的递归中去
        return self._dfs(node.left, curr_sum) + self._dfs(node.right, curr_sum)
```

#### 130. Surrounded Regions
```
class Solution:
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if (i in (0, m - 1) or j in (0, n - 1)) and board[i][j] == 'O':
                    self._dfs(board, i, j)
        
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == '#':
                    board[i][j] = 'O'
    
    def _dfs(self, board, i, j):
        board[i][j] = '#'

        m, n = len(board), len(board[0])
        if i > 0 and board[i - 1][j] == 'O':
            self._dfs(board, i - 1, j)
        if j > 0 and board[i][j - 1] == 'O':
            self._dfs(board, i, j - 1)
        if i < m - 1 and board[i + 1][j] == 'O':
            self._dfs(board, i + 1, j)
        if j < n - 1 and board[i][j + 1] == 'O':
            self._dfs(board, i, j + 1)
```

#### 133. Clone Graph
```
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        return self._helper(node, {})
    
    def _helper(self, node, node_map):
        if not node:
            return node
        # 这道题所有节点值不同
        # 所以我们可以使用哈希表来对应节点值和新生成的节点
        if node.label in node_map:
            return node_map[node.label]
        else:
            new_node = UndirectedGraphNode(node.label)
            node_map[node.label] = new_node
            for neighbor_node in node.neighbors:
                new_node.neighbors.append(self._helper(neighbor_node, node_map))
            return new_node
```

#### 135. Candy
```
class Solution:
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        这道题是说
        每个小朋友都至少有一个糖果
        分数高的小朋友要保证比分数低的小朋友糖果多
        问最少需要多少糖果
        """
        # 初始化每个人一个糖果，然后需要遍历两遍
        # 第一遍从左向右遍历，如果右边的小盆友的等级高，等加一个糖果，这样保证了一个方向上高等级的糖果多。
        # 然后再从右向左遍历一遍，如果相邻两个左边的等级高，而左边的糖果又少的话，则左边糖果数为右边糖果数加一。
        # 最后再把所有小盆友的糖果数都加起来返回即可。
        n = len(ratings)
        res = 0
        nums = [1] * n

        for i in range(n - 1):
            if ratings[i] < ratings[i + 1]:
                nums[i + 1] = nums[i] + 1
        
        for i in range(n - 1, 0, -1):
            if ratings[i] < ratings[i - 1]:
                # 反过来遍历时
                # 如果左边的rating高，而左边的糖果一定最少是右边的糖果再加1
                # 下面的max相当于确定了左边的最小值
                nums[i - 1] = max(nums[i - 1], nums[i] + 1)
        
        return sum(nums)
```

#### 136. Single Number
```
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        数组里面只有一个出现1次的数，其他都是两次
        """
        mapping = set()
        for num in nums:
            if num in mapping:
                mapping.remove(num)
            else:
                mapping.add(num)
        
        assert len(mapping) == 1
        return mapping.pop()
```

#### 137. Single Number II
```
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        每个数字都出现3次，只有一个数字出现了1次
        """
        # 当5第一次出现的时候，b = 5, a=0,  b记录这个数字
        # 当5第二次出现的时候，b = 0, a=5， a记录了这个数字
        # 当5第三次出现的时候，b = 0, a=0， 都清空了，可以去处理其他数字了
        # 所以，如果有某个数字出现了1次，就存在b中，出现了两次，就存在a中，所以返回 a|b
        a = b = 0
        for num in nums:
            a = a ^ num & ~b
            b = b ^ num & ~a
        return a | b
```

#### 138. Copy List with Random Pointer
```
# Definition for singly-linked list with a random pointer.
# class RandomListNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        if not head:
            return head
        
        ## 先生成新Node，放在原来Node的后面
        curr = head
        while curr:
            new_node = RandomListNode(curr.label)
            new_node.next = curr.next
            curr.next = new_node
            # 这里要注意curr要跳一位的
            # curr的下一个是新节点
            # 所以curr要更新为新节点的下一个 
            curr = new_node.next

        ## 将新生成的Node的random指针指向该指向的新Node
        ## curr.random是老节点，但是curr.random.next是新节点
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
        
        ## 最后将新链表拆除出来
        curr = head
        new_head = curr.next
        while curr:
            temp = curr.next
            curr.next = temp.next
            if temp.next:
                temp.next = temp.next.next
            curr = curr.next
        
        return new_head
```

#### 139. Word Break
```
class Solution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """

        if not s:
            return True
        
        if not wordDict:
            return False
        
        ## 思路：分隔板问题
        ## dp[i]表示字符串0到i是否可以用wordDict里的词表示
        ## j遍历所有0到i - 1的情况，如果dp[j]已经为True而且字符串j + 1到i也是wordDict中的某个词
        ## 则说明dp[i]可以用wordDict里的词表示了
        
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True
        
        for i in range(1, n + 1):
            for j in range(i + 1):
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = True
                    break
        
        return dp[-1]

## DFS解法（MLE）
# class Solution:
#     def wordBreak(self, s, wordDict):
#         """
#         :type s: str
#         :type wordDict: List[str]
#         :rtype: bool
#         """

#         return self.dfs(s, wordDict, 0)
    
#     def dfs(self, string, dict, start):
        
#         if start == len(string):
#             return True
        
#         for i in range(start, len(string)):
#             if string[start:i + 1] in dict:
#                 if self.dfs(string, dict, i + 1):
#                     return True
        
#         return False
```

#### 140. Word Break II, Hard, Facebook
```
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        # 剪枝思路：
        # 定义一个一位数组rangeValid（长度为len(s)
        # 其中rangeValid[i] = True表示在[i, n - 1]左闭右闭区间上有解
        # （相当于i的右半部分）
        # 如果某个区间之前被判定了无解
        # 下次循环时就会跳过这个区间，从而大大减少了运行时间。
        # code细节：
        # ```
        # oldResultsLength = len(results)
        # self.dfs(s, wordDict, i + 1, current, rangeValid, results)
        # if len(results) == oldResultsLength:
        #     rangeValid[i] = False
        # ````
        # 在下一次递归之前先保存下当前results的长度
        # 如果递归回来results没有变化，说明从i开始往后的区间是无解的
        # 所以可以把validRange[i]置为False，下次就不会去s的[i, n - 1]区间找答案了。
        res = []
        valid_range = [True] * len(s)
        self._dfs(s, 0, [], set(wordDict), valid_range, res)
        return res
    
    def _dfs(self, s, start, curr, words, valid_range, res):
        if start == len(s):
            res.append(' '.join(curr))
            return
        
        for i in range(start, len(s)):
            if s[start:i + 1] not in words:
                continue
            if not valid_range[i]:
                continue
            # 重要小技巧
            # 利用res数组在递归前后长度的变化来更新valid_range数组
            # 用于剪枝
            old_res_length = len(res)
            self._dfs(s, i + 1, curr + [s[start:i + 1]], words, valid_range, res)
            if len(res) == old_res_length:
                valid_range[i] = False
```

#### 141. Linked List Cycle
***
Given a linked list, determine if it has a cycle in it.
***
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        
        slow = fast = head
        
        while True:
            if fast.next:
                slow = slow.next
                fast = fast.next.next
                if not fast or not slow:
                    return False
                elif fast == slow:
                    return True
            else:
                return False
```

#### 143. Reorder List
***
Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
***
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        if not head or not head.next or not head.next.next:
            return
            
        slow = fast = head
        # 快慢指针
        # 如果链表是奇数
        # 则最后fast指针指向的是倒数第一个位置(被fast.next终止)
        # 如果链表是偶数
        # 则最后fast指针指向的是倒数第二个位置(被fast.next.next终止)
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        mid = slow.next
        slow.next = None
        last = mid
        pre = None
        while last:
            temp = last.next
            last.next = pre
            pre = last
            last = temp
        
        # pre就是最终反转后的新头
        # 理解是最终将以mid开头的链表接到原来的以head为头的链表上去
        while head and pre:
            head_temp = head.next
            head.next = pre
            pre = pre.next
            head.next.next = head_temp
            # 下面写成head = head_temp是一样的
            head = head.next.next
```

#### 145. Binary Tree Postorder Traversal, Hard, Facebook
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # 先序遍历是根-左-右，需要一个栈来实现
        # 后序遍历是左-右-根，所以和先序遍历正好相反，把根入队头即可
        if not root:
            return []

        res = deque()
        stack = [root]
        
        while stack:
            curr = stack.pop()
            res.appendleft(curr.val)
            if curr.left:
                stack.append(curr.left)
            if curr.right:
                stack.append(curr.right)
        
        return list(res)
```

#### 146. LRU Cache, Hard, Facebook, Linkedin
```
from collections import OrderedDict

class LRUCache(object):

    ## 这道题核心就是用python的OrderedDict
    ## 每一次操作都先pop一遍，再添加一次
    ## 这样就保证了最后被访问的（包括插入，查询）都会被放到最后一位
    ## 换句话说第一位就是最久没被访问过的，可以被pop掉
    ## OrderedDict是按照插入顺序保存的
    ## 支持的API：
    ## pop(key)
    ## popitem() 这里和普通的dict不同，普通的dict的popitem()不支持参数，
    ## OrderedDict的popitem(last = False)默认last = True,即默认从插入的最后一位pop
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.cache:
            val = self.cache.pop(key)
            self.cache[key] = val
            return val
        else:
            return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.cache:
            self.cache.pop(key)
        else:
            if len(self.cache) == self.capacity:
                self.cache.popitem(last = False)
        self.cache[key] = value
```

#### 148. Sort List
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 典型merge sort问题
        # 用到快慢指针找中点的技巧
        if not head or not head.next:
            return head
        
        slow = fast = head
        pre_slow = None
        while fast and fast.next:
            pre_slow = slow
            slow = slow.next
            fast = fast.next.next
        pre_slow.next = None

        return self._merge(self.sortList(head), self.sortList(slow))
    
    def _merge(self, node1, node2):
        dummy_node = ListNode(-1)
        curr = dummy_node
        while node1 and node2:
            if node1.val < node2.val:
                curr.next = node1
                node1 = node1.next
            else:
                curr.next = node2
                node2 = node2.next
            curr = curr.next
        
        if node1:
            curr.next = node1
        elif node2:
            curr.next = node2
        
        return dummy_node.next
```

#### 149. Max Points on a Line, Hard, Linkedin
```
# Definition for a point.
# class Point:
#     def __init__(self, a=0, b=0):
#         self.x = a
#         self.y = b
from collections import defaultdict
class Solution:

    def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """
        if not points:
            return 0
        
        n = len(points)
        
        res = 0
        for i in range(n):
            hash = defaultdict(int)
            same = 0
            for j in range(i + 1, n):
                if self.isEqual(points[i], points[j]):
                    same += 1
                else:
                    k = self.getK(points[i], points[j])
                    hash[k] += 1
            
            val = 0
            if hash:
                val = max(hash.values())
            res = max(res, val + same + 1)
        
        return res
    
    def isEqual(self, p1, p2):
        
        return p1.x == p2.x and p1.y == p2.y
    
    def getK(self, p1, p2):
        
        if p1.x == p2.x:
            return 2147483647
        
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        ## 单纯返回 dx / dy 会因为精度问题通过不了OJ
        ## 详情看
        ## http://www.cnblogs.com/grandyang/p/4579693.html
        gcd = self.gcd(dx, dy)
        return (dx // gcd, dy // gcd)

    def gcd(self, a, b):
        
        if b == 0:
            return a
        
        return self.gcd(b, a % b)
```

#### 150. Evaluate Reverse Polish Notation
```
class Solution:
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        
        number_stack = []
        for s in tokens:
            
            if s not in '+-*/':
                number_stack.append(int(s))
                continue
            
            a = number_stack.pop()
            # 为何可以直接pop两次？
            # 因为根据题意，肯定会有连续两个数字
            b = number_stack.pop()
            
            if s == '+':
                number_stack.append(b + a)
            elif s == '-':
                number_stack.append(b - a)
            elif s == '*':
                number_stack.append(b * a)
            elif s == '/':
                if a == 0:
                    number_stack.append(0)
                else:
                    number_stack.append(int(b / a))

        assert len(number_stack) == 1
        return number_stack.pop()
```

#### 151. Reverse Words in a String
```
class Solution(object):
    def reverseWords(self, s):
        words = s.split()
        return ' '.join([word for word in words[::-1]])
```

#### 152. Maximum Product Subarray
```
class Solution:
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
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
        
        ## 出了一个小bug：
        ## res注意在循环里是没有和pos[0]比较过的
        ## 所以要增加一次比较
        ## 因为最终答案可能就是第一个数字nums[0]
        res = -2147483648 if nums[0] < 0 else pos[0]
        for i in range(1, n):
            if nums[i] > 0:
                ## 因为此时pos[i]可能为0
                pos[i] = max(pos[i - 1] * nums[i], nums[i])
                neg[i] = neg[i - 1] * nums[i]
            if nums[i] < 0:
                neg[i] = min(pos[i - 1] * nums[i], nums[i])
                pos[i] = neg[i - 1] * nums[i]
            res = max(res, pos[i])
        
        return res
```

#### 153. Find Minimum in Rotated Sorted Array
```
class Solution:
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return
        
        # 好题
        # 重点：和nums的开头元素比较
        if len(nums) == 1 or nums[0] < nums[-1]:
            return nums[0]
        
        n = len(nums)
        start, end = 0, n - 1

        while start + 1 < end:
            mid = start + (end - start) // 2
            # 下面两个if都说明了找到了分界点
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            if nums[mid] < nums[mid - 1]:
                return nums[mid]
            if nums[mid] > nums[0]:
                start = mid + 1
            if nums[mid] < nums[0]:
                end = mid - 1
        
        if nums[start] > nums[end]:
            return nums[end]
        
        return nums[start]
```

#### 154. Find Minimum in Rotated Sorted Array II, Hard, Facebook
```
class Solution:
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 核心：用end比较确定折半方向
        if not nums:
            return 2 ** 31 - 1
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if nums[mid] > nums[end]:
                start = mid + 1
            # 此时我们并不清楚mid是否是解
            # 所以不能轻易去掉mid
            # 但是我们确定的是mid+1以后的肯定不是解
            # 因为题目要求的是最小值
            elif nums[mid] < nums[end]:
                end = mid
            else:
                end -= 1

        if nums[start] < nums[end]:
            return nums[start]
        
        return nums[end]
```

#### 156. Binary Tree Upside Down
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def upsideDownBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """

        ## 这道题是要干啥？
        ## 相当于将旧root的最左孩子变成新root，
        ## 旧root的最左孩子的兄弟右孩子变成新root的左孩子，
        ## 旧root的最左孩子的祖先节点们变成新root的右孩子，
        ## 顺时针旋转180度
        
        ##     1
        ##    / \
        ##   2   3
        ##  / \
        ## 4   5
        ## 变成
        ##   4
        ##  / \
        ## 5   2
        ##    / \
        ##   3   1 

        if not root or not root.left:
            return root
        
        newRoot = self.upsideDownBinaryTree(root.left)
        
        ## 持续对root的左孩子操作
        ## 下面是重点
        root.left.left = root.right
        root.left.right = root
        root.left = root.right = None
        
        return newRoot
        ## 1. 对于一个parent来说，假如有right node，必须得有left node。而有left node，right node可以为空。而right node必须为叶子节点。所以该树               每层至多有2个节点，并且2节点有共同的parent。
        ## 2. 所以对于最底层来说，必有一个left node，而这个left node则为整个新树的根。
        ## 3. 原树的根节点，变为了新树的最右节点。
        ## 4. 对于子树1 2 3来说，需要在以2为根的子树2 4 5建立成新树4 5 2后，插入到新树的最右节点2下面。原树的根节点root为left child，原树root->right为新树的left node
```

#### 157. Read N Characters Given Read4
```
# The read4 API is already defined for you.
# @param buf, a list of characters
# @return an integer
# def read4(buf):

class Solution(object):
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Maximum number of characters to read (int)
        :rtype: The number of characters read (int)
        """
        # 输入的buf就是一个能承装n个字符的array的引用
        # 输入的buf4就是一个能承装4个字符的array的引用
        buf4 = [None] * 4
        offset = 0
        
        while True:
            read_size = read4(buf4)
            i = 0
            while i < read_size and offset < n:
                buf[offset] = buf4[i]
                offset += 1
                i += 1
            # 如果读不出来东西了（read_size == 0）或者当前的buf满了（offset==n）
            # 说明可以return了
            if read_size == 0 or offset == n:
                return offset
```

#### 158. Read N Characters Given Read4 II - Call multiple times, Hard, Facebook
```
# The read4 API is already defined for you.
# @param buf, a list of characters
# @return an integer
# def read4(buf):
    def __init__(self):
        self.buf4 = [None] * 4
        self.read_pos = 0
        self.write_pos = 0
    
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Maximum number of characters to read (int)
        :rtype: The number of characters read (int)
        """
        i = 0
        while i < n:
            # 初始的时候read_pos是和write_pos重合的
            # 其实每次只要这两个位置重合
            # 就意味着buf4满了
            # write_pos意义是从read4中读到了多少个字符
            # 而read_pos是服务于往结果的buf里存数据的
            # 表示从当前的buf4里的哪个位置开始读
            # 换句话说，if里每次读4个字符
            # 而外层的while循环其实每次是往buf里存一个字符
            if self.read_pos == self.write_pos:
                self.read_pos = 0
                self.write_pos = read4(self.buf4)
                if self.write_pos == 0:
                    return i
            buf[i] = self.buf4[self.read_pos]
            self.read_pos += 1
            i += 1
        return i
```

#### 159. Longest Substring with At Most Two Distinct Characters
```
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s):
        """
        :type s: str
        :rtype: int
        """
        mapping = {}
        l = r = 0
        res = 0
        while r < len(s):
            if s[r] not in mapping:
                mapping[s[r]] = 1
            else:
                mapping[s[r]] += 1
            while l < r and len(mapping) > 2:
                mapping[s[l]] -= 1
                if mapping[s[l]] == 0:
                    del mapping[s[l]]
                l += 1
            res = max(res, r - l + 1)
            r += 1
        
        return res
```

#### 160. Intersection of Two Linked Lists
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """

        if not headA or not headB:
            return None
        
        lenA = 0
        curr = headA
        while curr:
            lenA += 1
            curr = curr.next
        
        lenB = 0
        curr = headB
        while curr:
            lenB += 1
            curr = curr.next
        
        currA, currB = headA, headB
        if lenA > lenB:
            steps = lenA - lenB
            while steps:
                headA = headA.next
                steps -= 1
        elif lenA < lenB:
            steps = lenB - lenA
            while steps:
                headB = headB.next
                steps -= 1
        
        while headA:
            if headA is headB:
                return headA
            headA = headA.next
            headB = headB.next
        
        return None
```

#### 161. One Edit Distance
```
class Solution:
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) < len(t):
            # 默认第一个参数s的长度是大于第二个参数t的长度的
            # 这样可以简化很多判断的逻辑
            return self.isOneEditDistance(t, s)
        
        length_s, length_t = len(s), len(t)
        if length_s - length_t > 1:
            return False
        
        for i in range(length_t):
            if s[i] != t[i]:
                if length_s == length_t:
                    return s[i + 1:] == t[i + 1:]
                # 长度大的字符串s抛弃当前的字符s[i], 看后面的字符串是否和t的子串是否相等
                return s[i + 1:] == t[i:]
        return length_s - length_t == 1
```

#### 162. Find Peak Element
```
class Solution:
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l, r = 0, len(nums) - 1
        
        while l <= r:
            mid = l + (r - l) // 2
            
            if (mid == 0 or nums[mid] > nums[mid - 1]) and \
                (mid == len(nums) - 1 or nums[mid] > nums[mid + 1]):
                return mid
            elif mid > 0 and nums[mid] < nums[mid - 1]:
                r = mid - 1
            else:
                l = mid + 1
        
        return -1
```

### 167. Two Sum II - Input array is sorted
```
class Solution:
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(numbers) < 2:
            return []
        
        start, end = 0, len(numbers) - 1
        while start + 1 < end:
            if numbers[start] + numbers[end] == target:
                return [start + 1, end + 1]
            elif numbers[start] + numbers[end] > target:
                end -= 1
            else:
                start += 1
        
        if numbers[start] + numbers[end] == target:
            return [start + 1, end + 1]
        
        return []
```

#### 170. Two Sum III - Data structure design
```
from collections import defaultdict
class TwoSum:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = defaultdict(int)
        

    def add(self, number):
        """
        Add the number to an internal data structure..
        :type number: int
        :rtype: void
        """
        self.data[number] += 1
        

    def find(self, value):
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        :type value: int
        :rtype: bool
        """
        
        for curr in self.data:
            # 支持a + a == 2a的情况
            if curr * 2 == value and self.data[curr] > 1:
                return True
            
            if curr * 2 != value and value - curr in self.data:
                return True

        return False
```

#### 173. Binary Search Tree Iterator
```
# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.stack = []
        self.pushLeft(root)

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) > 0
        
    def next(self):
        """
        :rtype: int
        """
        top = self.stack.pop()
        self.pushLeft(top.right)
        return top.val
    
    def pushLeft(self, node):
        while node:
            self.stack.append(node)
            node = node.left
```

#### 179. Largest Number
```
from functools import cmp_to_key

def _cmp(a, b):
    ab = str(a) + str(b)
    ba = str(b) + str(a)
    # 比如a=9, b = 30
    # ab=930, ba=309
    # 我们希望大的ab排在前面，所以返回-1
    # 在比较器里-1会将大于号左边的排在前面
    if ab > ba:
        return -1
    if ab == ba:
        return 0
    return 1

class Solution:
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        # 这样排过序以后 '9' > '30'会排在前面
        nums.sort(key=cmp_to_key(_cmp))
        res = ''.join(str(i) for i in nums)
        return res if res[0] != '0' else '0'
```

#### 187. Repeated DNA Sequences
```
from collections import defaultdict
class Solution:
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        
        if not s or len(s) < 10:
            return []

        ## 基本思路就是遍历所有的10个字符长度的子串，将其添加到hash里
        ## 如果此时该key对应在hash里的值大于1，说明出现了重复，将其添加到一个结果set里
        hash = defaultdict(int)
        res = set()
        for i in range(len(s) - 9):
            hash[s[i:i + 10]] += 1
            if hash[s[i:i + 10]] > 1:
                res.add(s[i:i + 10])
        
        return list(res)
```

#### 188. Best Time to Buy and Sell Stock IV
```
class Solution:
    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        一次交易是指一次完整的买入和卖出
        """
        # 虽然说k次完整交易
        # 我们可以把k次拆成2 * k次动作
        # 问题的实质是从长度为n的prices数组中挑选出至多2 * k个元素
        # 组成一个交易（买卖）序列。
        # 交易序列中的首次交易为买入，其后卖出和买入操作交替进行
        # (因为题目限制了手里最多只能持有一只股票，必须先卖出才能买入)。
        # 总收益为交易序列中的偶数项之和 - 奇数项之和。
        n = len(prices)
        if k > n // 2:
            return self._quick_solve(prices)
        
        # dp[j]表示完成j次交易时的最大收益
        # dp[j] = max(dp[j], dp[j - 1] + prices[i] * [1, -1][j % 2])
        dp = [-2 ** 31] * (2 * k + 1)
        dp[0] = 0
        for i in range(n):
            for j in range(min(2 * k, i + 1), 0, -1):
                dp[j] = max(dp[j], dp[j - 1] + prices[i] * [1, -1][j % 2])
        
        return max(dp)
    
    def _quick_solve(self, prices):
        res = 0
        for i in range(len(prices) - 1):
            if prices[i + 1] > prices[i]:
                res += prices[i + 1] - prices[i]
        return res
```

#### 189. Rotate Array
```
class Solution:
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        k %= len(nums)

        self._reverse(nums, 0, len(nums) - k - 1)
        self._reverse(nums, len(nums) - k, len(nums) - 1)
        self._reverse(nums, 0, len(nums) - 1)
    
    def _reverse(self, nums, left, right):
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1
```

#### 191. Number of 1 Bits
```
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 跟汉明距离那道题很像
        res = 0
        for i in range(32):
            if n & (1 << i) != 0:
                res += 1
        
        return res
```
#### 192. Word Frequency
```
# Write a bash script to calculate the frequency of each word in a text file words.txt.
cat words.txt | tr -s ' ' '\n' | sort | uniq -c | sort -r | awk '{ print $2, $1 }'
```

#### 199. Binary Tree Right Side View
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        
        queue = deque()
        queue.append(root)
        res = []
        
        while queue:
            res.append(queue[-1].val)
            q_len = len(queue)
            for _ in range(q_len):
                curr = queue.popleft()
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            
        return res
```

#### 195. Tenth Line
```
# 这道题是说给一个file.txt的文件，print出第10行
# 用awk来解答
cnt=0
while read line && [ $cnt -le 10 ]; do
  let 'cnt = cnt + 1'
  if [ $cnt -eq 10 ]; then
    echo $line
    exit 0
  fi
done < file.txt

# Solution 2
awk 'FNR == 10 {print }'  file.txt
# OR
awk 'NR == 10' file.txt

# Solution 3
sed -n 10p file.txt

# Solution 4
tail -n+10 file.txt|head -1
```

#### 200. Number of Islands
```
# BFS （LEETCODE超过，LINTCODE可以通过，很奇怪）
# from collections import deque

# class Solution:
#     def numIslands(self, grid):
        
#         if not grid or not grid[0]:
#             return 0
        
#         n, m = len(grid), len(grid[0])
#         visited = [[False] * m for _ in range(n)]
#         res = 0
        
#         directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
#         for i in range(n):
#             for j in range(m):
#                 if grid[i][j] == '1' and not visited[i][j]:
#                     # 注意这里的括号
#                     queue = deque([(i, j)])
#                     res += 1
#                     while queue:
#                         ci, cj = queue.popleft()
#                         visited[ci][cj] = True
#                         for di, dj in directions:
#                             newi, newj = ci + di, cj + dj
#                             if 0 <= newi < n and \
#                                 0 <= newj < m and \
#                                 not visited[newi][newj] and \
#                                 grid[newi][newj] == '1':
#                                 queue.append((newi, newj))
        
#         return res
                    
# # DFS
class Solution:
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        
        if not grid or not grid[0]:
            return 0
        
        n, m = len(grid), len(grid[0])
        visited = [[False] * m for _ in range(n)]
        res = 0
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1' and not visited[i][j]:
                    self._dfs(grid, visited, i, j)
                    res += 1
        
        return res
    
    def _dfs(self, grid, visited, x, y):
        if not 0 <= x < len(grid) or \
            not 0 <= y < len(grid[0]):
            return
        
        if grid[x][y] != '1' or visited[x][y]:
            return
        
        visited[x][y] = True
        self._dfs(grid, visited, x - 1, y)
        self._dfs(grid, visited, x + 1, y)
        self._dfs(grid, visited, x, y - 1)
        self._dfs(grid, visited, x, y + 1)
```

#### 205. Isomorphic Strings
```
class Solution:
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        if len(s) != len(t):
            return False
        
        ## 注意这里ms和mt的初始值
        ## 因为0是一个reasonable的index，所以不要用0做初始值
        ## 用其他任何非0的做初始值都行
        ms = [-1] * 256
        mt = [-1] * 256
        
        n = len(s)
        for i in range(n):
            if ms[ord(s[i])] != mt[ord(t[i])]:
                return False
            # 更新最后出现的位置
            ms[ord(s[i])] = i
            mt[ord(t[i])] = i
        
        return True
```

#### 206. Reverse Linked List
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # 迭代
        if not head:
            return head
        
        dummy_head = ListNode(0)
        dummy_head.next = head
        
        # 以dummy_head -> 3 -> 2 -> 1举例，当前curr在3位置
        # A. 先保存2的引用为temp
        # B. 3的next指向1(temp.next为1，因为temp此时是2)
        # C. 2(即temp)的next指向3(因为此时dummy_node的next就是指向3的)
        # D. dummy_head的next指向2(即temp 2，注意dummy_head一直保持不变)
        # 这样第一次循环完就变成了
        # dummy_head -> 2 -> 3 -> 1
        # 第二次循环中
        # A. 先保存1的引用为temp
        # B. 3的next指向None（因为1的next为None）
        # C. 1(即temp)的next指向2(因为此时2是dummy_node的next)
        # D. dummy_head的next指向1(注意dummy_head仍然是一直保持不变的)
        # 第二次循环完就变成了
        # dummy_head -> 1 -> 2 -> 3
        curr = head
        while curr.next:
            temp = curr.next
            curr.next = temp.next
            temp.next = dummy_head.next
            dummy_head.next = temp
        
        return dummy_head.next
    
        # 递归
#         if not head or not head.next:
#             return head
        
#         # 相当于reverse两个nodes
#         curr = head
#         head = self.reverseList(curr.next)
#         curr.next.next = curr
#         curr.next = None
#         return head
```

#### 207. Course Schedule
```
from collections import defaultdict
from collections import deque

class Solution:
    def canFinish(self, num_courses, pre_requisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        # 课程代号就是0到num_courses - 1
        indegrees = [0] * num_courses
        edges = defaultdict(list)
        
        for course, pre_course in pre_requisites:
            indegrees[course] += 1
            edges[pre_course].append(course)
        
        queue = deque()
        for i in range(num_courses):
            if indegrees[i] == 0:
                queue.append(i)
        
        res = 0
        while queue:
            curr = queue.popleft()
            res += 1
            for each in edges[curr]:
                indegrees[each] -= 1
                if indegrees[each] == 0:
                    queue.append(each)

        return res == num_courses
```

#### 208. Implement Trie (Prefix Tree)
```
class Node:
    def __init__(self):
        self.is_word = False
        self.children = [None] * 26

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._root = Node()
        
    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curr = self._root
        for ch in word:
            index = ord(ch) - ord('a')
            if not curr.children[index]:
                curr.children[index] = Node()
            curr = curr.children[index]
        curr.is_word = True
        
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curr = self._root
        for ch in word:
            index = ord(ch) - ord('a')
            if not curr.children[index]:
                return False
            curr = curr.children[index]
        return curr.is_word

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curr = self._root
        for ch in prefix:
            index = ord(ch) - ord('a')
            if not curr.children[index]:
                return False
            curr = curr.children[index]
        
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```

#### 209. Minimum Size Subarray Sum
```
class Solution:
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        
        # 典型双指针题目
        # 左右指针从0开始
        # 追赶型双指针
        left = right = 0
        n = len(nums)
        
        res = n + 1
        total = 0
        while right < n:
            while total < s and right < n:
                total += nums[right]
                right += 1
            while total >= s:
                total -= nums[left]
                left += 1
                res = min(res, right - left + 1)
        
        return res if res != n + 1 else 0
```

#### 210. Course Schedule II
```
from collections import defaultdict
from collections import deque

class Solution:
    def findOrder(self, num_courses, pre_requisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        edges = defaultdict(list)
        indegrees = [0] * numCourses
        
        for course, pre_course in prerequisites:
            edges[pre_course].append(course)
            indegrees[course] += 1
        
        queue = deque()
        for course, indegree in enumerate(indegrees):
            if indegree == 0:
                queue.append(course)
        
        res = []
        while queue:
            curr = queue.popleft()
            res.append(curr)
            for each in edges[curr]:
                indegrees[each] -= 1
                if indegrees[each] == 0:
                    queue.append(each)
        
        # 此时表明并没有一个能够完成全部课程的路径
        if len(res) != numCourses:
            return []
        
        return res
```

#### 211. Add and Search Word - Data structure design
```
class WordDictionary:
    class _Node:
        def __init__(self):
            self.is_word = False
            self.children = [None] * 26
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._root = self._Node()
        

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        curr = self._root

        for ch in word:
            index = ord(ch) - ord('a')
            if not curr.children[index]:
                curr.children[index] = self._Node()
            curr = curr.children[index]
        
        curr.is_word = True
        

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        
        return self._dfs(self._root, word, 0)
    
    
    # 递归定义：在以node为根的字典树中搜索从index开始能否在字典树中找到一个word
    def _dfs(self, node, word, index):
        if index == len(word):
            return node.is_word
        
        ch = word[index]

        if ch != '.':
            if not node.children[ord(ch) - ord('a')]:
                # 剪枝，不需要在递归下去了
                return False
            return self._dfs(node.children[ord(ch) - ord('a')], word, index + 1)

        # 在此情况下，ch为'.'，所以需要递归下去了
        for each in node.children:
            if each is None:
                continue
            if self._dfs(each, word, index + 1):
                # 同样剪枝，不需要在递归下去了
                return True

        return False
                
# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```

#### 212. Word Search II, Hard, Facebook
```
class Node:
    def __init__(self):
        self.children = [None] * 26
        self.is_word = False
        self.word = ''


class Trie:
    def __init__(self):
        self.root = Node()
    
    def insert(self, word):
        curr = self.root
        for ch in word:
            if not curr.children[ord(ch) - ord('a')]:
                curr.children[ord(ch) - ord('a')] = Node()
            curr = curr.children[ord(ch) - ord('a')]
        curr.is_word = True
        curr.word = word


_DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class Solution:
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        if not board or not board[0]:
            return []
        
        trie = Trie()
        for word in words:
            trie.insert(word)
        
        res = set()
        m, n = len(board), len(board[0])
        visited = [[False] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                self._dfs(board, trie.root, res, visited, i, j)
        
        return list(res)
    
    # 递归定义：从棋盘的i,j点开始走，能找到多少个存在在给定list里的word
    def _dfs(self, board, root, results, visited, i, j):
        
        visited[i][j] = True

        # 递归核心：当前遍历的棋盘上的点是否在当前root的孩子节点中
        # 这决定了是否需要剪枝
        if root.children[ord(board[i][j]) - ord('a')]:
            root = root.children[ord(board[i][j]) - ord('a')]
            if root.is_word:
                results.add(root.word)
            # 因为aaa和aaab都是满足条件的答案
            # 所以上面不能直接return 
            for di, dj in _DIRECTIONS:
                newi, newj = i + di, j + dj
                if 0 <= newi < len(board) and 0 <= newj < len(board[0]) and \
                    not visited[newi][newj]:
                    self._dfs(board, root, results, visited, newi, newj)

        visited[i][j] = False
```

#### 214. Shortest Palindrome
```
class Solution:
    def shortestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        Given a string s, you are allowed to convert it to a palindrome 
        by adding characters in front of it. Find and return the shortest 
        palindrome you can find by performing this transformation.
        """
        # 这道题下面的brute force能AC
        # 还有two pointers解法
        # 最优解应该是O(n)的KMP
        n = len(s)
        rev_s = s[::-1]
        for i in range(n):
            if s[:n - i] == rev_s[i:]:
                return rev_s[:i] + s
        
        return ''
```

#### 215. Kth Largest Element in an Array
```
class Solution:
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        
        while True:
            candidate_pos = self._partition(nums, left, right)
            if candidate_pos == k - 1:
                return nums[candidate_pos]
            elif candidate_pos > k - 1:
                right = candidate_pos - 1
            else:
                left = candidate_pos + 1
    
    # partition定义为在left和right之间确定在nums中left最终的位置
    def _partition(self, nums, left, right):
        """
        对nums[left...right]前闭后闭部分进行partition操作，
        返回final_pivot_pos, 使得nums[left...final_pivot_pos-1] > nums[final_pivot_pos]
        并且nums[final_pivot_pos+1...right] <= nums[final_pivot_pos]
        (<=是因为可能在nums中有重复的元素)
        """
        
        # final_pivot_pos相当于当前nums数组的分界点
        # 在最终交换那一步（return之前的交换）之前
        # 都满足[left...final_pivot_pos]里都是大于pivot_value
        # 而[final_pivot_pos + 1...right]里都是小于等于pivot_value的
        # final_pivot_pos起始设置为left位置就好了
        final_pivot_pos = left
        pivot_value = nums[left]
        # 在i次循环时候
        # 肯定满足nums[left + 1...final_pivot_pos] > pivot_value
        # 而且nums[final_pivot_pos + 1...i - 1] <= pivot_value
        for i in range(left + 1, right + 1):
            # 此时i相当于整个数组的右半部分（应该小于pivot_value的部分）
            # 标准升序的quick sort中这里应该是小于号的
            # 但是这里是求第k大，相当于要降序
            # 所以用大于号
            if nums[i] > pivot_value:
                # 此时i位置不满足条件，需要调整时
                # 此时final_pivot_pos + 1位置的元素肯定是排好的
                # 所以safe to swap
                # 调整分界点加1即可
                nums[final_pivot_pos + 1], nums[i] = nums[i], nums[final_pivot_pos + 1]
                final_pivot_pos += 1
        
        nums[left], nums[final_pivot_pos] = nums[final_pivot_pos], nums[left]
        return final_pivot_pos
```

#### 218. The Skyline Problem
```
class Solution:
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        h = []
        for start, end, height in buildings:
            h.append((start, -height))
            h.append((end, height))
        h.sort()
        
        pre = curr = 0
        m = []
        m.append(0)
        res = []
        
        # 思路很简单：
        # 1. 先将(start, -height)和(end, height)都加入到h中，
        # 并且按照坐标排个序
        # 2. 使用一个最大堆m
        # 3. 遍历h，通过height的正负号判断是start还是end
        # 如果是start，二话不说加入堆，并判断当前堆顶(curr)是否前一次的堆顶元素(pre)一样
        # 如果不一样，说明这次操作堆顶发生了变化，加入到res中，并更新pre
        # 如果是end，说明需要从堆中pop掉
        
        # 但是这道题比较坑的是python没有支持remove任意元素的heap（类似c++ multiset）
        # 可以参考库sortedcontainers但是不是buildin的
        # LC判断也挺坑，多提交几次某次就通过了（尽管faster than大概小于1%的样子）
        # 可能是底层还是直接判断执行时间
        for start, height in h:
            if height < 0:
                m.append(-height)
                m.sort(reverse=True)
            else:
                m.remove(height)
                m.sort(reverse=True)
            curr = m[0]
            if curr != pre:
                res.append([start, curr])
                pre = curr
        
        return res
```

#### 220. Contains Duplicate III
```
class Solution:
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        问在num中是否存在两个index
        这两个index的距离小于等于k
        并且他们对应的值的差小于等于t
        """
        # 基本思路就是桶排序的方法
        # 当前桶号是m，num[i]通过整除桶的range w
        # 就能得到num[i]这个值对应的桶号
        # 满足值小于等于t的元素一定属于当前同一个桶或者临近桶
        if t < 0:
            return False
        n = len(nums)
        d = {}
        # w是桶自身的range
        w = t + 1
        for i in range(n):
            m = nums[i] // w
            # 注意：下面三个if都是处理值t的
            if m in d:
                return True
            if m - 1 in d and abs(nums[i] - d[m - 1]) <= t:
                return True
            if m + 1 in d and abs(nums[i] - d[m + 1]) <= t:
                return True
            d[m] = nums[i]
            # 这个if才是处理index距离k的
            if i >= k:
                del d[nums[i - k] // w]
        return False
```

#### 222. Count Complete Tree Nodes
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 下面简单的递归TLE
        # if not root:
        #     return 0
        # return 1 + self.countNodes(root.left) + self.countNodes(root.right)
        
        # 层序遍历 O(n)
        # if not root:
        #   return 0
        # queue = deque([root])
        # res = 0
        # while queue:
        #     curr = queue.popleft()
        #   res += 1
        #   if curr.left:
        #       queue.append(curr.left)
        #   if curr.right:
        #       queue.append(curr.right)
        # return res
        
        # 但是这道题由于是完全树
        # 所以有优化解法（O(logN * logN)）
        
        if not root:
            return 0
        left_height = right_height = 0
        left_node = right_node = root
        while left_node:
            left_height += 1
            left_node = left_node.left
        while right_node:
            right_height += 1
            right_node = right_node.right
        if left_height == right_height:
            return 2 ** left_height - 1
        # 这个1是root存在的意思
        return self.countNodes(root.left) + self.countNodes(root.right) + 1
```

#### 224. Basic Calculator
```
class Solution:
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = curr = 0
        stack = []
        sign = 1
        
        for ch in s:
            if '0' <= ch <= '9':
                curr = 10 * curr + int(ch)
            elif ch in '+-':
                res += sign * curr
                curr = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                # 遇到左括号就讲res入栈并重置res为0, sign为1
                # 这样在连续遇到左括号时候就不会额外加入什么值
                sign, res = 1, 0
            elif ch == ')':
                res += sign * curr
                res *= stack.pop()
                res += stack.pop()
                curr = 0
        
        return res + curr * sign

# # 下面的解法通不过最后一个大test case (35/36)
# # 实际上思路和计算器III带括号和加减乘除的一样
# class Solution:
#     def calculate(self, s):
#         """
#         :type s: str
#         :rtype: int
#         """
#         n = len(s)
#         num = curr_res = res = 0
#         op = '+'
#         i = 0
#         while i < n:
#             ch = s[i]
#             if '0' <= ch <= '9':
#                 num = 10 * num + int(ch)
#             elif ch == '(':
#                 cnt = 0
#                 j = i
#                 while j < n:
#                     if s[j] == '(':
#                         cnt += 1
#                     elif s[j] == ')':
#                         cnt -= 1
#                     if cnt == 0:
#                         break
#                     j += 1
#                 num = self.calculate(s[i + 1:j])
#                 i = j
            
#             if ch in '+-' or i == n - 1:
#                 if op == '+':
#                     curr_res += num
#                 elif op == '-':
#                     curr_res -= num
#                 res += curr_res
#                 curr_res = 0
#                 num = 0
#                 op = ch
#             i += 1
        
#         return res
```

#### 226. Invert Binary Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return root
    
        left = root.left
        right = root.right

        # 注意：这里不能用self.invertTree(root.right)和self.invertTree(root.left)
        # 因为在root.right那一行
        # 应该用的是原来的left node，此时的root.left已经被修改过了！！！
        root.left = self.invertTree(right)
        root.right = self.invertTree(left)

        return root
```

#### 227. Basic Calculator II
```
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        这道题是有加减乘除但是没有括号的
        """
        num = 0
        n = len(s)
        op = '+'
        stack = []
        
        # 默认第一次的operator是加好（从无到有就是加了一个初始的数字）
        # 这道题是假定了输入的字符串一定是valid
        # 比如不会出现" * 10" 这种乘号左边没有数字的非法字符串
        for i in range(n):
            if '0' <= s[i] <= '9':
                num = 10 * num + ord(s[i]) - ord('0')
            
            # 注意这里不能写成elif
            # 因为要保证最后一位即使是数字也要进入下面的循环
            # 如果用了elif，最后一位是数字的时候会进入上面的循环
            if s[i] in '+-*/' or i == n - 1:
                if op == '+':
                    stack.append(num)
                elif op == '-':
                    stack.append(-num)
                elif op == '*':
                    curr = stack.pop()
                    stack.append(curr * num)
                elif op == '/':
                    curr = stack.pop()
                    if curr < 0:
                        curr *= -1
                        stack.append(-1 * (curr // num))
                    else:
                        stack.append(curr // num)
                op = s[i]
                num = 0
        
        res = 0
        while stack:
            res += stack.pop()
        
        return res
```

#### 230. Kth Smallest Element in a BST
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        
        cnt = self._count(root.left)
        
        if k <= cnt:
            return self.kthSmallest(root.left, k)
        elif k > cnt + 1:
            return self.kthSmallest(root.right, k - cnt - 1)
        else: # now k == cnt + 1
            return root.val
    
    def _count(self, node):
        if not node:
            return 0
        return 1 + self._count(node.left) + self._count(node.right)
```

#### 234. Palindrome Linked List
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# 这道题可以用stack 空间复杂度O(n)
# 也可以如下用翻转链表，空间复杂度O(1)
class Solution:
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head or not head.next:
            return True
        
        slow = fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            
        curr = slow.next
        while curr.next:
            temp = curr.next
            curr.next = temp.next
            temp.next = slow.next
            slow.next = temp
        
        left = head
        right = slow.next
        
        while left and right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        
        return True
```

#### 235. Lowest Common Ancestor of a Binary Search Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        if not root:
            return
        
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        
        return root
```

#### 236. Lowest Common Ancestor of a Binary Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        # 由于这道题限制了p和q一定都在二叉树中存在
        # 那么如果当前结点不等于p或q，p和q要么分别位于左右子树中
        # 要么同时位于左子树，或者同时位于右子树
        
        # 如果当前node的值正好等于p或者q的值，直接将node返回
        if not root or p.val == root.val or q.val == root.val:
            return root
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left and right:
            return root
        
        return left if left else right
```

#### 238. Product of Array Except Self
```
class Solution:
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        
        n = len(nums)
        res = [1] * n
        
        for i in range(1, n):
            res[i] = res[i - 1] * nums[i - 1]
        
        right = 1
        for j in range(n - 1, -1, -1):
            res[j] *= right
            right *= nums[j]
        
        return res
```

#### 239. Sliding Window Maximum, Hard, Facebook
```
from collections import deque

class Solution:
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if not nums or k == 0:
            return []
        
        if k < 2:
            return nums
        # 这道题典型双端队列（递减队列，存的是坐标）
        # 队头（最左边）就是当前窗口里最大的值
        res = []
        desc_queue = deque()

        for i in range(len(nums)):
            if desc_queue and desc_queue[0] == i - k:
                desc_queue.popleft()
            while desc_queue and nums[desc_queue[-1]] < nums[i]:
                desc_queue.pop()
            desc_queue.append(i)
            # 这里是因为k是1-based-index的窗口大小
            # 比如k=3，说明窗口是3
            # 则index肯定大于等于2（0-based-index）
            if i > k - 2:
                res.append(nums[desc_queue[0]])
        
        return res
```

#### 240. Search a 2D Matrix II
```
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        i, j = m - 1, 0
        
        # 重点：从矩阵左下角做二分！！！
        # i是当前所在行，起始为m - 1
        # j是当前坐在列，起始为0
        while i >= 0 and j < n:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                i -= 1
            else:
                j += 1
        
        return False
```

#### 242. Valid Anagram
```
class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False

        n = len(s)
        mapping = [0] * 26
        for i in range(n):
            mapping[ord(s[i]) - ord('a')] += 1
        
        for j in range(n):
            mapping[ord(t[j]) - ord('a')] -= 1
            if mapping[ord(t[j]) - ord('a')] < 0:
                return False
        
        return True
```

#### 243. Shortest Word Distance
```
class Solution:
    def shortestDistance(self, words, word1, word2):
        """
        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        
        ## http://www.cnblogs.com/grandyang/p/5187041.html
        ## 我们用两个变量p1,p2初始化为-1
        ## 然后我们遍历数组，遇到单词1，就将其位置存在p1里
        ## 若遇到单词2，就将其位置存在p2里
        ## 如果此时p1, p2都不为-1了，那么我们更新结果
        if not words or (word1 not in words and word2 not in words):
            return -1
        
        p1 = p2 = -1
        n = len(words)
        res = n
        for i in range(n):
            if words[i] == word1:
                p1 = i
            if words[i] == word2:
                p2 = i
            
            if p1 != -1 and p2 != -1:
                res = min(res, abs(p1 - p2))
        
        return res
```

#### 244. Shortest Word Distance II
```
from collections import defaultdict

class WordDistance:

    _INT_MAX = 2 ** 31 - 1
    
    def __init__(self, words):
        """
        :type words: List[str]
        """
        self._mapping = defaultdict(list)
        for i in range(len(words)):
            self._mapping[words[i]].append(i)
        

    def shortest(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        
        if word1 not in self._mapping or \
            word2 not in self._mapping:
            return self._INT_MAX
        
        p1 = p2 = 0
        res = self._INT_MAX
        while p1 < len(self._mapping[word1]) and \
            p2 < len(self._mapping[word2]):
            word1_inx = self._mapping[word1][p1]
            word2_inx = self._mapping[word2][p2]
            res = min(res, abs(word1_inx - word2_inx))
            ## 因为self.hash[word]里存的是一个递增的index
            ## 所以每次尽可能的就把当前值小的index再增大一点
            ## 这样差值会变得更小
            ## res就会不停的在更新，往最小的差值方向逼近
            if word1_inx < word2_inx:
                p1 += 1
            else:
                p2 += 1
        
        return res


# Your WordDistance object will be instantiated and called as such:
# obj = WordDistance(words)
# param_1 = obj.shortest(word1,word2)
```

#### 246. Strobogrammatic Number
```
class Solution:
    def isStrobogrammatic(self, num):
        """
        :type num: str
        :rtype: bool
        """
        if not num:
            return True

        mapping = {
            '0': '0',
            '1': '1',
            '6': '9',
            '8': '8',
            '9': '6'
        }
        
        left, right = 0, len(num) - 1
        # 注意这里是小于等于号
        # 最终left和right重合的情况就是中点(当num长度为奇数)
        # 如果left=right+1就是num长度为偶数
        while left <= right:
            if num[left] not in mapping.keys() or \
                num[right] not in mapping.keys():
                return False
            if mapping[num[left]] != num[right]:
                return False
            left += 1
            right -= 1
        
        return True
    
    
        # python更好的写法
        # maps = {('0', '0'), ('1', '1'), ('6', '9'), ('8', '8'), ('9', '6')}
        # i, j = 0, len(num) - 1
        # while i <= j:
        #     if (num[i], num[j]) not in maps:
        #         return False
        #     i += 1
        #     j -= 1
        # return True
```

#### 247. Strobogrammatic Number II
```
class Solution:
    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        return self._dfs(n, n)
    
    def _dfs(self, inner_width, outer_width):
        if inner_width == 0:
            return ['']
        if inner_width == 1:
            return ['0', '1', '8']
        res = []
        inner_res = self._dfs(inner_width - 2, outer_width)
        for each in inner_res:
            # 如果当前是最外层的话
            # ‘010’不是一个valid number
            if inner_width < outer_width:
                res.append('0' + each + '0')
            res.append('1' + each + '1')
            res.append('6' + each + '9')
            res.append('9' + each + '6')
            res.append('8' + each + '8')
        
        return res
```

#### 248. Strobogrammatic Number III, Hard, Facebook
```
class Solution:
    def strobogrammaticInRange(self, low, high):
        """
        :type low: str
        :type high: str
        :rtype: int
        """
        self._res = 0 
        # len(high) + 1是防止出现'10'到'99'没有进入for循环的情况
        for i in range(len(low), len(high) + 1):
            # 初始化设置中心为'', '0', '1', '8'
            self._dfs(low, high, '', i)
            self._dfs(low, high, '0', i)
            self._dfs(low, high, '1', i)
            self._dfs(low, high, '8', i)
        return self._res
    
    # 定义为更新当前找到的valid path并且path的长度小于为str_len的解的个数
    def _dfs(self, low, high, last_path, valid_len):
        if len(last_path) > valid_len:
            return
        elif len(last_path) == valid_len:
            if valid_len == len(low) and last_path < low:
                return
            if valid_len == len(high) and last_path > high:
                return
            if valid_len > 1 and last_path[0] == '0':
                return
            self._res += 1
        else:
            # 下面第一个条件是必须的
            # 因为会有1001的情况
            # 需要从空开始，先加两个0，然后再加两个1变成1001
            self._dfs(low, high, '0' + last_path + '0', valid_len)
            self._dfs(low, high, '1' + last_path + '1', valid_len)
            self._dfs(low, high, '6' + last_path + '9', valid_len)
            self._dfs(low, high, '9' + last_path + '6', valid_len)
            self._dfs(low, high, '8' + last_path + '8', valid_len)
```

#### 251. Flatten 2D Vector
```
class Vector2D(object):

    def __init__(self, vec2d):
        """
        Initialize your data structure here.
        :type vec2d: List[List[int]]
        """
        self._data = [i[:] for i in vec2d]
        self._row = 0
        self._col = 0
        self._max_row = len(vec2d)

    def next(self):
        """
        :rtype: int
        """
        ret = self._data[self._row][self._col]
        self._col += 1
        return ret

    def hasNext(self):
        """
        :rtype: bool
        """
        # 陷阱：注意下面不能用if
        # 因为有test case是例如[[1], [], [3]]的情况
        # 在这种情况下需要跳过中间的空list
        while self._row < self._max_row and \
            self._col == len(self._data[self._row]):
            self._col = 0
            self._row += 1
        return self._row < self._max_row
        

# Your Vector2D object will be instantiated and called as such:
# i, v = Vector2D(vec2d), []
# while i.hasNext(): v.append(i.next())
```

#### 252. Meeting Rooms
```
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: bool
        """
        interval_list = [(i.start, i.end) for i in intervals]
        interval_list.sort()
        
        for i in range(1, len(interval_list)):
            curr_start, _ = interval_list[i]
            _, last_end = interval_list[i - 1]
            if curr_start < last_end:
                return False
        
        return True
```

#### 253. Meeting Rooms II
```
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

from collections import defaultdict

class Solution:
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        if not intervals:
            return 0
        
        ## 这道题核心需要一个有顺序的hash表(按照时间点从小到大排序)
        ## 里面需要存每个时间点上的room count
        rooms_at_time = defaultdict(int)
        for interval in intervals:
            rooms_at_time[interval.start] += 1
            rooms_at_time[interval.end] -= 1
        
        rooms_list = []
        for time_point, counts in rooms_at_time.items():
            rooms_list.append((time_point, counts))
        rooms_list.sort()
        
        ## localRes的含义是在遍历时候记录当前需要的room数目
        ## 然后每次都去更新全局最大的globalRes
        global_res = local_res = 0
        for _, counts in rooms_list:
            local_res += counts
            global_res = max(global_res, local_res)
        
        return global_res
```

#### 254. Factor Combinations
```
from math import sqrt
from math import floor

class Solution:
    def getFactors(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        
        res = []
        
        if n < 2:
            return []
        self._dfs(n, 2, [], res)
        return res
    
    def _dfs(self, n, start, curr, res):
        if n == 1:
            if len(curr) > 1:
                res.append(curr[:])
            return
        
        for i in range(start, n + 1):
            if n % i == 0:
                curr.append(i)
                self._dfs(n // i, i, curr, res)
                curr.pop()
```

#### 256. Paint House
```
class Solution:
    INT_MAX = 2 ** 31 - 1
    def minCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        # 这道题实际上是265 Paint House II的解法（k种房间颜色）
        if not costs or not costs[0]:
            return 0
        
        number_of_houses, number_of_colors = len(costs), len(costs[0])
        dp = [[self.INT_MAX] * number_of_colors for _ in range(number_of_houses)]
        dp[0] = costs[0][:]
        
        for i in range(1, number_of_houses):
            for j in range(number_of_colors):
                dp[i][j] = costs[i][j] + \
                    min(dp[i - 1][k] for k in range(number_of_colors) if k != j)
        
        return min(dp[-1])
```

#### 257. Binary Tree Paths
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        res = []
        if root:
            self._dfs(root, '', res)
        return res
    
    def _dfs(self, node, curr, res):
        if not node.left and not node.right:
            res.append(curr + str(node.val))
            # 其实这里的return可以不加
            # 加在这里是为了方便可读
            return

        # 这道题跟标准dfs稍微不同
        # 在函数内判断是否node为空，如果不空才可以进入下一次递归
        if node.left:
            self._dfs(node.left, curr + str(node.val) + '->', res)

        if node.right:
            self._dfs(node.right, curr + str(node.val) + '->', res)
```

#### 259. 3Sum Smaller
```
class Solution:
    def threeSumSmaller(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        nums中有多少个三元索引组合，sum和小于target
        """
        nums.sort()
        total = 0
        # 循环定义：固定nums[i]（即可得到target - nums[i]）
        # 然后从i+1到最后寻找多少组的和小于target - nums[i]
        for i in range(len(nums) - 2):
            total += self._two_sum_smaller(nums, i + 1, target - nums[i])
        return total
    
    def _two_sum_smaller(self, nums, start, target):
        total = 0
        l, r = start, len(nums) - 1
        while l < r:
            if nums[l] + nums[r] < target:
                # 注意这里不需要-1或者+1
                # 因为是找二元组的个数
                # -1是说l和r之间有多少个间隔
                # +1是说l和r之间有多少个数字（左闭右闭）
                total += r - l
                l += 1
            else:
                r -= 1
        return total
```

#### 260. Single Number III
```
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # 这道题有几个点要注意：
        # 异或的特点： （1）a ^ 0 = a, (2) a ^ a = 0
        # diff &= -diff得到的是一个2的阶乘（二进制00001000只有一个1）
        # 结果表示是原来diff最右边的1的位置
        diff = 0
        for num in nums:
            diff ^= num
        diff &= -diff

        res = [0] * 2
        for num in nums:
            # 最终的diff一定是两个目标数字的异或
            # 第一个if条件是能准确找到唯一的一个数字（两个解中某一个解, 这个解在diff位置有1，而另外一个解
            # 在diff位置上一定没有1，因为diff是异或过的！！！！这样就能区分出两个解了）
            # else条件就是把所有的其他数字都异或一次
            # 因为余下只剩一个出现一次的数字
            # 其他的数字都是出现两次
            # 就可以把另一个数字找出来了
            if num & diff != 0:
                res[0] ^= num
            else:
                res[1] ^= num
        
        return res
```

#### 261. Graph Valid Tree
```
from collections import deque

class Solution:
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """
        
        if not edges:
            # 1个节点，肯定没有边，为True
            # 0个节点，也肯定没有边，也为True
            return n <= 1 and True
        
        graph = [set() for _ in range(n)]
        
        for point1, point2 in edges:
            graph[point1].add(point2)
            graph[point2].add(point1)
        
        queue = deque([edges[0][0]])
        visited = set([edges[0][0]])
        while queue:
            curr_point = queue.popleft()
            for each in graph[curr_point]:
                # why?
                if each in visited:
                    return False
                visited.add(each)
                queue.append(each)
                # 重要， 一定要remove
                # 否则会反过来再遍历一边
                # why？
                graph[each].remove(curr_point)
        
        return len(visited) == n
```

#### 266. Palindrome Permutation
```
class Solution:
    def canPermutePalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        hash_set = set()
        for ch in s:
            if ch in hash_set:
                hash_set.remove(ch)
            else:
                hash_set.add(ch)
        return len(hash_set) <= 1
```

#### 269. Alien Dictionary, Hard, Facebook
```
from collections import defaultdict
from collections import deque

class Solution:
    def alienOrder(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        hash_map = defaultdict(set)
        hash_indegree = defaultdict(int)
        result = ''
        
        if not words:
            return result
        
        for word in words:
            for ch in word:
                hash_indegree[ch] = 0
        
        for i in range(len(words) - 1):
            curr_word = words[i]
            next_word = words[i + 1]
            word_length = min(len(curr_word), len(next_word))
            for j in range(word_length):
                small_ch = curr_word[j]
                large_ch = next_word[j]
                if small_ch != large_ch:
                    if not large_ch in hash_map[small_ch]:
                        hash_map[small_ch].add(large_ch)
                        hash_indegree[large_ch] += 1
                    break
        
        queue = deque()
        for key, val in hash_indegree.items():
            if val == 0:
                queue.append(key)
        
        while queue:
            # 队列每次pop出来的队头
            # 就是要处理的元素
            # 本题中就是解的一部分
            curr_ch = queue.popleft()
            result += curr_ch
            for ch in hash_map[curr_ch]:
                hash_indegree[ch] -= 1
                if hash_indegree[ch] == 0:
                    queue.append(ch)

        # hash_indegree里存的是理论上都应该能在res里的
        assert len(res) <= len(hash_indegree)
        return result if len(result) == len(hash_indegree) else ''
```

#### 270. Closest Binary Search Tree Value
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def closestValue(self, root, target):
        """
        :type root: TreeNode
        :type target: float
        :rtype: int
        """
        if not root:
            return 2 ** 31 - 1
        
        res = root.val
        while root:
            # 第一次会pass掉这个if
            # 因为res在外面初始过了
            # 此时左右相等
            if abs(res - target) > abs(root.val - target):
                res = root.val

            # 大于小于号这里都可以
            if target >= root.val:
                root = root.right
            else:
                root = root.left
        
        return res
```

#### 272. Closest Binary Search Tree Value II, Hard, Linkedin
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
# 最简单好想的思路
# 就是中序遍历
class Solution:
    def closestKValues(self, root, target, k):
        """
        :type root: TreeNode
        :type target: float
        :type k: int
        :rtype: List[int]
        """
        
        if not root:
            return
        
        arr = []
        self._in_order(root, arr)
        
        arr_list = []
        for i, val in enumerate(arr):
            arr_list.append((abs(val - target), i))

        arr_list.sort()
        return [arr[i[1]] for i in arr_list[:k]]
    
    def _in_order(self, node, arr):
        if not node:
            return
        
        self._in_order(node.left, arr)
        arr.append(node.val)
        self._in_order(node.right, arr)
```

#### 273. Integer to English Words, Hard, Facebook, Linkedin
```
from collections import deque

class Solution:
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        
        level1 = ("Zero One Two Three Four Five Six Seven Eight Nine Ten" + \
                 " Eleven Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen").split()
        level2 = "Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety".split()
        level3 = "Hundred"
        level4 = "Thousand Million Billion".split()
        
        words, digits = deque(), 0
        while num:
            token = num % 1000
            num //= 1000
            word = ''
            if token > 99:
                word = level1[token // 100] + ' ' + level3 + ' '
                token %= 100
            if token > 19:
                word += level2[token // 10 - 2] + ' '
                token %= 10
            if token:
                word += level1[token] + ' '
            word = word.strip()
            if word:
                if digits:
                    word += ' ' + level4[digits - 1]
                words.appendleft(word)
            digits += 1
        
        return ' '.join(words) or 'Zero'
```

#### 277. Find the Celebrity
```
# The knows API is already defined for you.
# @param a, person a
# @param b, person b
# @return a boolean, whether a knows b
# def knows(a, b):

class Solution(object):
    def findCelebrity(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        res = 0
        for i in range(1, n):
            if knows(res, i):
                res = i
        
        for i in range(n):
            if res == i:
                continue
            if knows(res, i) or not knows(i, res):
                return -1
        
        return res
```

#### 278. First Bad Version
```
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1:
            return 1
        
        start, end = 1, n
        while start + 1 < end:
            mid = start + (end - start) // 2
            if isBadVersion(mid):
                end = mid
            else:
                start = mid + 1
        
        if isBadVersion(start):
            return start
        return end
```

#### 280. Wiggle Sort
```
class Solution:
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) <= 1:
            return
        # 根据题目要求
        # 当i为奇数时，nums[i] >= nums[i - 1]
        # 当i为偶数时，nums[i] <= nums[i - 1]
        # 类似贪心的思路，时间复杂度O(n)
        n = len(nums)
        for i in range(1, n):
            if (i % 2 == 0 and nums[i] > nums[i - 1]) or \
                (i % 2 == 1 and nums[i] < nums[i - 1]):
                nums[i], nums[i - 1] = nums[i - 1], nums[i]
```

#### 281. Zigzag Iterator
```
from collections import deque

class ZigzagIterator(object):

    def __init__(self, v1, v2):
        """
        Initialize your data structure here.
        :type v1: List[int]
        :type v2: List[int]
        """
        # 好题！！用双端队列来解题
        self._queue = deque([deque(v) for v in (v1, v2) if v])

    def next(self):
        """
        :rtype: int
        """
        curr_v = self._queue.popleft()
        value = curr_v.popleft()
        if curr_v:
            self._queue.append(curr_v)
        return value

    def hasNext(self):
        """
        :rtype: bool
        """
        return bool(self._queue)

# Your ZigzagIterator object will be instantiated and called as such:
# i, v = ZigzagIterator(v1, v2), []
# while i.hasNext(): v.append(i.next())
```

#### 282. Expression Add Operators, Hard, Facebook
```
class Solution:
    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]
        """
        if not num:
            return []
        
        res = []
        self._dfs(num, target, 0, '', 0, 0, res)
        return res
    
    def _dfs(self, num, target, start_pos, last_str, last_val, last_diff, res):
        if start_pos == len(num):
            if last_val == target:
                res.append(last_str)
            return

        for i in range(start_pos, len(num)):
            curr_str = num[start_pos:i + 1]
            if i != start_pos and num[start_pos] == '0':
                break
            
            if start_pos == 0:
                self._dfs(
                    num, target, i + 1,
                    curr_str, int(curr_str), int(curr_str), res,
                )
            else:
                self._dfs(
                    num, target, i + 1,
                    # 比如上次是"1+2",这次是"*5"
                    # 上次的diff是2，curr_sum是3（1+2=3）
                    # 所以这次要先3-2再加上2*5
                    last_str + '*' + curr_str,
                    last_val - last_diff + last_diff * int(curr_str),
                    last_diff * int(curr_str),
                    res,
                )
                self._dfs(
                    num, target, i + 1,
                    last_str + '+' + curr_str,
                    last_val + int(curr_str),
                    int(curr_str),
                    res,
                )
                self._dfs(
                    num, target, i + 1,
                    last_str + '-' + curr_str,
                    last_val - int(curr_str),
                    -int(curr_str),
                    res,
                )
```

#### 283. Move Zeroes
```
class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        available_pos_for_0 = 0
        # i指针始终在available_pos_for_0之后的
        # 不停的将i指针指向的非零数字移到available_pos_for_0上去
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[available_pos_for_0] = nums[available_pos_for_0], nums[i]
                available_pos_for_0 += 1
```

#### 285. Inorder Successor in BST
```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
        """
        # 实际上等价于寻找bst中第一个大于p值得节点
        res = None
        while root:
            if root.val > p.val:
                res = root
                root = root.left
            else:
                root = root.right
        
        return res
```

#### 286. Walls and Gates
```
# # DFS
# class Solution:
#     def wallsAndGates(self, rooms):
#         """
#         :type rooms: List[List[int]]
#         :rtype: void Do not return anything, modify rooms in-place instead.
#         """
#         if not rooms or not rooms[0]:
#             return
        
#         for i in range(len(rooms)):
#             for j in range(len(rooms[i])):
#                 if rooms[i][j] == 0:
#                     self._dfs(rooms, i, j, 0)
    
#     def _dfs(self, rooms, i, j, current_depth):
#         if not 0 <= i < len(rooms) or \
#             not 0 <= j < len(rooms[i]) or \
#             rooms[i][j] < current_depth:
#             return
#         rooms[i][j] = current_depth
#         self._dfs(rooms, i + 1, j, current_depth + 1)
#         self._dfs(rooms, i - 1, j, current_depth + 1)
#         self._dfs(rooms, i, j + 1, current_depth + 1)
#         self._dfs(rooms, i, j - 1, current_depth + 1)


# BFS
class Solution:
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: void Do not return anything, modify rooms in-place instead.
        """
        if not rooms or not rooms[0]:
            return
        
        # 这道题思路是先找门（0的点）然后从门出发更新最近的空房间的距离
        queue = []
        for i in range(len(rooms)):
            for j in range(len(rooms[i])):
                if rooms[i][j] == 0:
                    queue.append((i, j))
        
        DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        while queue:
            curr_i, curr_j = queue.pop(0)
            for dir_i, dir_j in DIRECTIONS:
                x = curr_i + dir_i
                y = curr_j + dir_j
                if not 0 <= x < len(rooms) or not 0 <= y < len(rooms[0]) or rooms[x][y] < rooms[curr_i][curr_j] + 1:
                    continue
                rooms[x][y] = rooms[curr_i][curr_j] + 1
                queue.append((x, y))
```

#### 287. Find the Duplicate Number
```
class Solution:
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 这道题可以排序，可以用set，可以二分，可以用找环的思想
        # 比如：1 -> 3 -> 2 -> 4 -> 2 （2, 4, 2）就形成了环 
        # 已知0肯定是不在list里的
        # 所以0可以用来做初始化
        slow = fast = temp = 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        # 此时slow就是(2, 4, 2)环中的某个点
        # 下面需要找到环里的重复值
        while True:
            slow = nums[slow]
            temp = nums[temp]
            if slow == temp:
                break
        
        return slow
        
        # 丧心病狂的三行解法：
        # while nums[nums[0]] != nums[0]:
        #     nums[nums[0]], nums[0] = nums[0], nums[nums[0]]
        # return nums[0]
```

#### 291. Word Pattern II, Hard, Facebook
```
class Solution:
    def wordPatternMatch(self, pattern, string):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        return self._dfs(pattern, string, {}, set())
    
    # 递归定义：每次以pattern的首字母为目标
    # 去判断剩下的pattern[1:]能不能组成一个valid pattern
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

            if self._dfs(pattern[1:], string[i + 1:], mapping, used):
                return True

            del mapping[ch]
            used.remove(word)

        return False
```

#### 295. Find Median from Data Stream, Hard, Facebook
```
from heapq import heappush
from heapq import heappop

class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self._max_heap = []
        self._min_heap = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        # 维护两个堆
        # 最大堆的最大值小于最小堆里的任何元素
        # 同时保证最大堆的长度等于最小堆的长度或者+1
        # 这样就保证了当前的中位数应该是最大堆堆顶元素与最小堆
        # 堆顶元素的平均值
        heappush(self._max_heap, -num)
        heappush(self._min_heap, -heappop(self._max_heap))
        if len(self._max_heap) < len(self._min_heap):
            heappush(self._max_heap, -heappop(self._min_heap))
            

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self._max_heap) == len(self._min_heap):
            return (-self._max_heap[0] + self._min_heap[0]) / 2
        return -self._max_heap[0]


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

#### 296. Best Meeting Point, Hard, Facebook
```
class Solution:
    def minTotalDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # 注意：这道题要求的是最短的距离, 而不是最短距离的点！！！
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        rows = []
        cols = []
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    rows.append(i)
                    cols.append(j)
        
        return self._helper(rows) + self._helper(cols)
    
    def _helper(self, v):
        res = 0
        v.sort()
        i, j = 0, len(v) - 1
        while i < j:
            # 最佳点肯定在最内层的i，j之间
            # 对应的距离就是v[j] - v[i]
            # 所以要不停的缩小i和j
            res += v[j] - v[i]
            j -= 1
            i += 1
        return res
```

#### 297. Serialize and Deserialize Binary Tree, Hard, Facebook, Linkedin
```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        :type root: TreeNode
        :rtype: str
        """
        ret = []
        self._serialize(root, ret)
        return ' '.join(ret)

    def _serialize(self, node, results):
        if node:
            results.append(str(node.val))
            self._serialize(node.left, results)
            self._serialize(node.right, results)
        else:
            results.append('#')

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :type data: str
        :rtype: TreeNode
        """
        inputs = data.split(' ')
        return self._deserialize(inputs)

    def _deserialize(self, inputs):
        if not inputs:
            return
        curr = inputs.pop(0)
        if curr == '#':
            return
        node = TreeNode(int(curr))
        node.left = self._deserialize(inputs)
        node.right = self._deserialize(inputs)
        return node
            
# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```

#### 300. Longest Increasing Subsequence
```
class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        n = len(nums)
        # dp[i]是以i结尾的最长递增子序列
        # 子序列subsequence是可以不连续的
        # subarray是连续的
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], 1 + dp[j])
        
        return max(dp)
```

#### 301. Remove Invalid Parentheses, Hard, Facebook
```
from collections import deque

# BFS解法
# 基本思路就是remove 1个，remove 2个 etc。。。
class Solution:
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        
        res = []
        visited = set()
        visited.add(s)
        done = False
        queue = deque([s])

        while queue:
            curr = queue.popleft()
            if self.is_valid(curr):
                done = True
                res.append(curr)
            # 根据BFS的性质
            # 当首次从队列中发现有效字符串时
            # 其去掉的括号数一定是最小的。
            # 所以只要发现curr已经合法
            # 后面不需要再向队列里添加字符串了
            # 因为一定不是remove minimum parenthese
            if done:
                continue
            for i in range(len(curr)):
                if curr[i] in ('(', ')'):
                    # 去掉一个括号
                    new_str = curr[:i] + curr[i + 1:]
                    if new_str not in visited:
                        visited.add(new_str)
                        queue.append(new_str)
        
        return res
    
    def is_valid(self, s):
        count = 0
        for c in s:
            if c == '(':
                count += 1
            elif c == ')':
                count -= 1
            if count < 0:
                return False
        # 注意：最后要检查count是否为0的
        return count == 0
```

#### 304. Range Sum Query 2D - Immutable
```
class NumMatrix:

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        if not matrix or not matrix[0]:
            return
        m, n = len(matrix), len(matrix[0])
        self._sum = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                self._sum[i][j] = self._sum[i][j - 1] + \
                    self._sum[i - 1][j] - \
                    self._sum[i - 1][j - 1] + \
                    matrix[i - 1][j - 1]

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        return self._sum[row2 + 1][col2 + 1] - \
        self._sum[row2 + 1][col1] - \
        self._sum[row1][col2 + 1] + \
        self._sum[row1][col1]

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```

#### 305. Number of Islands II, Hard, Facebook
```
class Union:
    def __init__(self, n):
        self.father = [i for i in range(n)]
        self.count = 0
    
    def find(self, a):
        if self.father[a] == a:
            return self.father[a]
        self.father[a] = self.find(self.father[a])
        return self.father[a]

    def connect(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
            self.count -= 1
            

class Solution:
    
    _DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def numIslands2(self, m, n, positions):
        """
        :type m: int
        :type n: int
        :type positions: List[List[int]]
        :rtype: List[int]
        这道题是给出一个空board
        然后要支持往里加岛的操作
        """
        board = [[0] * n for _ in range(m)]
        union = Union(m * n)

        res = []
        for ci, cj in positions:
            if board[ci][cj] == 0:
                board[ci][cj] = 1
                union.count += 1
                for di, dj in self._DIRECTIONS:
                    newi, newj = ci + di, cj + dj
                    if 0 <= newi < m \
                        and 0 <= newj < n \
                        and board[newi][newj] == 1:
                        # 注意坐标：n是列数！！！
                        union.connect(ci * n + cj, newi * n + newj)
            res.append(union.count)
        
        return res
```

#### 307. Range Sum Query - Mutable
```
class BinaryIndexTree:
    def __init__(self, nums):
        n = len(nums)
        self._nums = [0] * (n + 1)
        self._pre_sum = [0] * (n + 1)
        for inx, val in enumerate(nums):
            self.setter(inx + 1, val)
    
    def _lowbit(self, x):
        # 核心：非常重要！！！
        # lowbit函数计算的是2 ** k的值
        return x & -x

    def setter(self, inx, new_val):
        diff = new_val - self._nums[inx]
        self._nums[inx] = new_val
        while inx < len(self._pre_sum):
            self._pre_sum[inx] += diff
            inx += self._lowbit(inx)
    
    def getter(self, inx):
        res = 0
        while inx > 0:
            res += self._pre_sum[inx]
            inx -= self._lowbit(inx)
        return res
        

class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.bit = BinaryIndexTree(nums)

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        self.bit.setter(i + 1, val)

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.bit.getter(j + 1) - self.bit.getter(i)


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)
```

#### 308. Range Sum Query 2D - Mutable
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

class NumMatrix:

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
        return self.bit.getter(row2 + 1, col2 + 1) - \
            self.bit.getter(row2 + 1, col1) - \
            self.bit.getter(row1, col2 + 1) +
            self.bit.getter(row1, col1)


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# obj.update(row,col,val)
# param_2 = obj.sumRegion(row1,col1,row2,col2)
```

#### 311. Sparse Matrix Multiplication
```
class Solution:
    def multiply(self, A, B):
        """
        :type A: List[List[int]]
        :type B: List[List[int]]
        :rtype: List[List[int]]
        """
        row = len(A)
        col = len(B[0])
        assert len(A[0]) == len(B)
        common = len(A[0])
        
        mat = [[0] * col for _ in range(row)]
        for i in range(row):
            for j in range(common):
                if A[i][j] == 0:
                    continue
                for k in range(col):
                    if B[j][k] == 0:
                        continue
                    # 核心
                    # 所以可以当A[i][j] == 0或者B[j][k] == 0
                    # 时候直接continue掉
                    mat[i][k] += A[i][j] * B[j][k]
        
        return mat
```

#### 312. Burst Balloons
```
class Solution:
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        nums = [1] + nums + [1]
        # dp[i][j]定义：
        # nums中i到j气球全部打爆能有多少分数
        dp = [[0] * (n + 2) for _ in range(n + 2)]
        
        # i遍历的意义是当前剩下几个气球
        for i in range(1, n + 1):
            # left是从下标1开始的一直到n - i + 1
            for left in range(1, n - i + 2):
                right = left + i - 1
                for k in range(left, right + 1):
                    dp[left][right] = max(
                        dp[left][right],
                        # 判断left到right之间所气球都打爆能有多少分数
                        # 实际上到最后最剩下一个气球k
                        # 我们遍历这个k
                        # 最终结果应该分三部分：
                        # 1. 最后这个气球k和left的left以及right的right的乘积
                        # 2. left到k-1之间气球全部打爆的分数
                        # 3. k+1到right之间全部气球打爆的分数
                        nums[left - 1] * nums[k] * nums[right + 1] + \
                            dp[left][k - 1] + dp[k + 1][right]
                    )

        return dp[1][n]
```

#### 314. Binary Tree Vertical Order Traversal
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import defaultdict
from collections import deque

class Solution:
    def verticalOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        
        # 即将root当成竖直中心轴节点，垂直来看左边-1， 右边+1
        queue = deque([(root, 0)])
        res = defaultdict(list)
        
        while queue:
            curr_node, level = queue.popleft()
            res[level].append(curr_node.val)
            if curr_node.left:
                queue.append((curr_node.left, level - 1))
            if curr_node.right:
                queue.append((curr_node.right, level + 1))
        
        # sorted返回的是一个list
        # 即对res这个dict的keys做sort
        return [res[i] for i in sorted(res)]
```

#### 315. Count of Smaller Numbers After Self, Hard, Facebook
```
class Solution:
    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []
        
        res = [0] * len(nums)
        self._merge_sort(list(enumerate(nums)), res)
        return res
    
    def _merge_sort(self, nums, res):
        if len(nums) <= 1:
            return nums
        
        mid = len(nums) // 2
        left_sorted = self._merge_sort(nums[:mid], res)
        right_sorted = self._merge_sort(nums[mid:], res)
        
        for i in range(len(nums) - 1, -1, -1):
            # 这里left_sorted[-1]是left_sorted中最大的
            # right_sorted[-1]是right_sorted里最大的
            if left_sorted and right_sorted:
                if left_sorted[-1][1] > right_sorted[-1][1]:
                    # 此时可以更新res数组了，因为出现了反转对儿,并且反转对儿的个数
                    # 正好是right_sorted数组的长度
                    res[left_sorted[-1][0]] += len(right_sorted)
                    nums[i] = left_sorted.pop()
                else:
                    # 这里的pop并赋值操作就相当于merge
                    # 因为for循环是倒着循环的
                    # 所以nums[i]应该是逐渐变小
                    # 每次取left_sorted和right_sorted中较大的数字放入nums[i]中
                    nums[i] = right_sorted.pop()
            else:
                if left_sorted:
                    nums[i] = left_sorted.pop()
                else:
                    nums[i] = right_sorted.pop()
        
        return nums
```

#### 316. Remove Duplicate Letters
```
class Solution:
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        m = [0] * 256
        visited = [False] * 256
        res = [0]
        
        for ch in s:
            m[ord(ch)] += 1
        
        for ch in s:
            m[ord(ch)] -= 1
            if visited[ord(ch)]:
                continue
            while ord(ch) < res[-1] and m[res[-1]] > 0:
                visited[res[-1]] = False
                res.pop()
            visited[ord(ch)] = True
            res.append(ord(ch))
        
        return ''.join(chr(i) for i in res[1:])
```

#### 317. Shortest Distance from All Buildings, Hard, Facebook
```
from collections import deque

class Solution:
    
    _DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def shortestDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or not grid[0]:
            return 0

        m, n = len(grid), len(grid[0])
        res = 2 ** 31 - 1
        building_cnt = 0
        distance = [[0] * n for _ in range(m)]
        reached_buildings = [[0] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                # 这道题很有意思
                # 起始入队的是一个grid[i][j]为1的点
                # 后面在while loop里入队的是grid[i][j]为0的点
                if grid[i][j] == 1:
                    building_cnt += 1
                    queue = deque()
                    queue.append((i, j))
                    visited = [[False] * n for _ in range(m)]
                    curr_dist = 1
                    while queue:
                        q_len = len(queue)
                        for _ in range(q_len):
                            ci, cj = queue.popleft()
                            for di, dj in self._DIRECTIONS:
                                newi, newj = ci + di, cj + dj
                                if 0 <= newi < m \
                                    and 0 <= newj < n \
                                    and grid[newi][newj] == 0 \
                                    and not visited[newi][newj]:
                                    distance[newi][newj] += curr_dist
                                    reached_buildings[newi][newj] += 1
                                    visited[newi][newj] = True
                                    queue.append((newi, newj))
                        curr_dist += 1
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and reached_buildings[i][j] == building_cnt:
                    res = min(res, distance[i][j])

        # 注意本题可能出现无解的情况（即不能遍历到所有的buildings）
        # 所以要额外check一下
        return res if res < 2 ** 31 - 1 else -1
```

#### 322. Coin Change
```
class Solution:
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        ## 维护一个一维动态数组dp，其中dp[i]表示钱数为i时的最小硬币数的找零，递推式为：
        ## dp[i] = min(dp[i], dp[i - coins[j]] + 1);
        
#         dp = [amount + 1 for i in range(amount + 1)]
#         dp[0] = 0
        
#         for i in range(1, amount + 1):
#             for j in range(len(coins)):
#                 if i - coins[j]>= 0:
#                     dp[i] = min(dp[i], dp[i - coins[j]] + 1)
        
#         return dp[-1] if dp[-1] <= amount else -1

        ## 书影博客
        ## 这家伙一向喜欢在遍历当前index时候去更新未来的状态
        dp = [0] + [-1] * amount
        for x in range(amount):
            ## 当遍历到当前dp[x]状态时候，应该已经考虑过了x左边所有的情况
            ## 所以此时如果仍然dp[x] == -1，则当前状态一定不能凑齐找零钱数
            if dp[x] == -1:
                continue
            for c in coins:
                if x + c > amount:
                    continue
                ## x + c一定在当前x的右边，
                ## 所以可以用现在的状态dp[x]去更新未来
                if dp[x + c] == -1 or dp[x + c] > dp[x] + 1:
                    dp[x + c] = dp[x] + 1
        return dp[amount]
```

#### 323. Number of Connected Components in an Undirected Graph
```
class Union:
    def __init__(self, n):
        self.father = [i for i in range(n)]
        self.count = n
    
    def find(self, a):
        if a == self.father[a]:
            return a
        # 重点！！！
        self.father[a] = self.find(self.father[a])
        return self.father[a]
    
    def connect(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = self.father[root_b]
            self.count -= 1
    
    def query(self):
        return self.count
        

class Solution:
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        
        union = Union(n)
        for a, b in edges:
            union.connect(a, b)
        
        return union.query()
```

#### 325. Maximum Size Subarray Sum Equals k
```
class Solution:
    def maxSubArrayLen(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        注意这道题求的是subarray
        """
        total = 0
        max_len = 0
        # mapping里放的是sum和对应的坐标
        mapping = {}
        
        for i, num in enumerate(nums):
            total += num
            if total == k:
                # max_len代表的是长度，所以要加1
                max_len = i + 1
            # 注意：这里一定是total - k
            # 因为相当于presum 当前的total要减去以前的total（total - k）
            # total - （total - k） == k！！！
            elif total - k in mapping:
                max_len = max(max_len, i - mapping[total - k])

            if total not in mapping:
                mapping[total] = i
        
        return max_len
```

#### 326. Power of Three
```
class Solution:
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        这道题是是要判断n是不是3的某次方
        """
        # 不停的除3即可
        # 因为如果n是3个某次方
        # 不停的除以3相当于消除了3
        # 最后的结果肯定是1
        # 5 % 3 = 2, 2 // 3 = 0
        while n and n % 3 == 0:
            n //= 3
        
        return n == 1
    
        # 如果不用循环，还有一种取巧的方式
        # 在int32中3最大的阶乘是3 ** 19=1162261467
        # 所以这道题直接看1162261467能不能被n整除即可
        # return n > 0 and 1162261467 % n == 0
```

#### 329. Longest Increasing Path in a Matrix, Hard, Facebook
```
from collections import deque
from collections import defaultdict

class Solution:
    
    _DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
#         if not matrix or not matrix[0]:
#             return 0
        
#         res = 1
#         m, n = len(matrix), len(matrix[0])
#         # dp[i][j]表示从matrix的i j开始最大递增路径的长度
#         dp = [[0] * n for _ in range(m)]
        
#         for i in range(m):
#             for j in range(n):
#                 if dp[i][j] > 0:
#                     continue
#                 queue = deque([(i, j)])
#                 length = 1
#                 while queue:
#                     length += 1
#                     q_len = len(queue)
#                     for _ in range(q_len):
#                         ci, cj = queue.popleft()
#                         for di, dj in self._DIRECTIONS:
#                             newi, newj = ci + di, cj + dj
#                             if 0 <= newi < m \
#                                 and 0 <= newj < n \
#                                 and matrix[newi][newj] > matrix[ci][cj] \
#                                 and dp[newi][newj] < length:
#                                 queue.append((newi, newj))
#                                 dp[newi][newj] = length
#                                 res = max(res, dp[newi][newj])
        
#         return res
                      
        if not matrix or not matrix[0]:
            return 0

        m, n = len(matrix), len(matrix[0])
        sequence = []
        for i in range(m):
            for j in range(n):
                sequence.append((matrix[i][j], i, j))
        sequence.sort()
        
        pos_length_dict = defaultdict(int)
        for val, ci, cj in sequence:
            pos_length_dict[(ci, cj)] = 1
            steps = 0
            # matrix[newi][newj] < val一定是小于号！！
            # 因为更新的是从哪个newi newj能走到ci cj来
            # 如果能走过来，就对ci cj的hash map上更新newi newj带来的步数
            # 所以要找小于val的方向
            for di, dj in self._DIRECTIONS:
                newi, newj = ci + di, cj + dj
                if 0 <= newi < m \
                    and 0 <= newj < n \
                    and matrix[newi][newj] < val:
                    steps = max(steps, pos_length_dict[(newi, newj)])
            pos_length_dict[(ci, cj)] += steps
        
        return max(pos_length_dict.values())
```

#### 332. Reconstruct Itinerary
```
from collections import defaultdict

class Solution:
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        graph = defaultdict(list)
        for from_, to_ in tickets:
            graph[from_].append(to_)
        
        for each in graph:
            graph[each].sort(reverse=True)
        
        res = []
        self._helper(graph, 'JFK', res)
        return res[::-1]
    
    def _helper(self, graph, from_, res):
        while graph[from_]:
            curr = graph[from_].pop()
            self._helper(graph, curr, res)
        # 本质上就是图的后序遍历 
        res.append(from_)
```

#### 336. Palindrome Pairs, Hard, Facebook
```
class Solution:
    def palindromePairs(self, words):
        """
        :type words: List[str]
        :rtype: List[List[int]]
        """
        mapping = {word: index for index, word in enumerate(words)}
        
        res = set()
        for inx, word in enumerate(words):
            
            # case 1: 检查空字符串是否是一个解
            if '' in mapping and word != '' and \
                self._valid(word):
                res.add((mapping[''], inx))
                res.add((inx, mapping['']))
            
            # case 2: 检查当前word的反词是否能组成一个解
            reversed_word = word[::-1]
            if reversed_word in mapping and mapping[reversed_word] != inx:
                res.add((mapping[reversed_word], inx))
                res.add((inx, mapping[reversed_word]))
            
            # case 3: 检查当前词拆成左右两部分
            # 并且能组成类似(reversed_right, left, right)
            # 或者(left, right, reversed_left)的形式
            # 注意两点：
            # 1是i索引要从1开始（因为空字符串已经考虑过了）
            # 2是最终加入到res的时候要注意顺序（两个词从从左到右能够拼接）
            for i in range(1, len(word)):
                left_word, right_word = word[:i], word[i:]
                reversed_left = left_word[::-1]
                reversed_right = right_word[::-1]
                # (reversed_right, left, right)形式
                if self._valid(left_word) and reversed_right in mapping:
                    res.add((mapping[reversed_right], inx))
                # (left, right, reversed_left)的形式
                if self._valid(right_word) and reversed_left in mapping:
                    res.add((inx, mapping[reversed_left]))
        
        return list(res)
    
    def _valid(self, word):
        left, right = 0, len(word) - 1
        while left < right:
            if word[left] != word[right]:
                return False
            left += 1
            right -= 1
        return True
```

#### 339. Nested List Weight Sum
```
class Solution:
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        # 这道题和364-Nested List Weight Sum不一样的地方在于
        # weight是从小到大的（那道题最外层weight最大）
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
                for t in curr.getList():
                    stack.append((t, level + 1))
        
        return res
```

#### 340. Longest Substring with At Most K Distinct Characters, Hard, Facebook
```
from collections import defaultdict

class Solution:
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        Given a string,
        find the length of the longest substring T 
        that contains at most k distinct characters.
        """
        counter = defaultdict(int)
        res = left = right = 0
        
        while right < len(s):
            counter[s[right]] += 1
            while len(counter) > k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    counter.pop(s[left])
                left += 1
            res = max(res, right - left + 1)
            right += 1

        return res
```

#### 341. Flatten Nested List Iterator
```
class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = nestedList[::-1][:]

    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop().getInteger()
        
    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack:
            curr = self.stack[-1]
            if curr.isInteger():
                return True
            else:
                temp_list = self.stack.pop().getList()
                for each in temp_list[::-1]:
                    self.stack.append(each)
        return False
```

#### 344. Reverse String
```
class Solution:
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ''
        for i in range(len(s) - 1, -1, -1):
            res += s[i]
        return res
```

#### 346. Moving Average from Data Stream
```
from collections import deque

class MovingAverage:

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self._queue = deque(maxlen=size)

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self._queue.append(val)
        return sum(self._queue) / len(self._queue)

# Your MovingAverage object will be instantiated and called as such:
# obj = MovingAverage(size)
# param_1 = obj.next(val)
```

#### 347. Top K Frequent Elements
```
import heapq
from collections import defaultdict

class Solution:
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        nums_count = defaultdict(int)
        for num in nums:
            nums_count[num] += 1
        
        ret = []
        for key, value in nums_count.items():
            heapq.heappush(ret, (value, key))
            if len(ret) > k:
                heapq.heappop(ret)
        
        return [i[1] for i in ret][::-1]
```

#### 348. Design Tic-Tac-Toe
```
class TicTacToe:

    def __init__(self, n):
        """
        Initialize your data structure here.
        :type n: int
        """
        self.row = [0] * n
        self.col = [0] * n
        self.diag = 0
        self.anti_diag = 0
        self.n = n

    def move(self, row, col, player):
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        :type row: int
        :type col: int
        :type player: int
        :rtype: int
        """
        offset = 1 if player == 1 else -1
        self.row[row] += offset
        self.col[col] += offset
        if row == col:
            self.diag += offset
        if row + col == self.n - 1:
            self.anti_diag += offset
        # 即当前这一步row col在或者行或者列或者对角或者反对角凑齐了n个
        if self.n in (self.row[row], self.col[col], self.diag, self.anti_diag):
            return 1
        if -self.n in (self.row[row], self.col[col], self.diag, self.anti_diag):
            return 2
        return 0

# Your TicTacToe object will be instantiated and called as such:
# obj = TicTacToe(n)
# param_1 = obj.move(row,col,player)
```

#### 349. Intersection of Two Arrays
```
class Solution:
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        
        s = set()
        for num in nums1:
            s.add(num)
        
        res = []
        for num in nums2:
            if num in s:
                res.append(num)
                # 这里的思路很重要
                s.remove(num)
        return res
```

#### 350. Intersection of Two Arrays II
```
class Solution:
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        len1 = len(nums1)
        len2 = len(nums2)
        
        nums1.sort()
        nums2.sort()
        
        i = j = 0
        res = []
        while i < len1 and j < len2:
            if nums1[i] > nums2[j]:
                j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                res.append(nums1[i])
                i += 1
                j += 1
        
        return res
```

#### 354. Russian Doll Envelopes
```
class Solution:
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        envelopes_copy = [(i[0], -i[1]) for i in envelopes]
        envelopes_copy.sort()
        n = len(envelopes_copy)
        # dp定义是在最终排好序的信封中，高度h从小到大的排序
        # 所以dp的长度就是最终的答案
        dp = []
        for i in range(n):
            l, r = 0, len(dp)
            t = -envelopes_copy[i][1]
            while l < r:
                mid = l + (r - l) // 2
                # 这里的高度h是从大到小排列的
                # 核心：要找到最后一个大于等于t的坐标right
                if dp[mid] >= t:
                    r = mid
                else:
                    l = mid + 1
            if r >= len(dp):
                dp.append(t)
            else:
                dp[r] = t
        return len(dp)

        # python TLE
#         if not envelopes:
#             return 0

#         envelopes.sort()
#         n = len(envelopes)
#         dp = [1] * n
#         for i in range(n):
#             for j in range(i):
#                 if envelopes[i][0] > envelopes[j][0] and envelopes[i][1] > envelopes[j][1]:
#                     dp[i] = max(dp[i], dp[j] + 1)
        
#         return max(dp)
```

#### 358. Rearrange String k Distance Apart
```
from collections import defaultdict
from heapq import heappush
from heapq import heappop

class Solution:
    def rearrangeString(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        # 说明此时s中所有的字符只需要相隔0个位置
        # 相当于不需要重置，直接返回即可
        if k == 0:
            return s
        
        mapping = defaultdict(int)
        for ch in s:
            mapping[ch] += 1
        
        hp = []
        for ch, count in mapping.items():
            # 这里需要一个最大堆，python只有最小堆
            heappush(hp, (-count, ch))
        
        n = len(s)
        res = []
        while hp:
            # 在每个while循环里
            # 都往res里添加k个字符
            # 添加的顺序就是优先取从当前剩余的最多字符
            temp = []
            cnt = min(k, n)
            for _ in range(cnt):
                if not hp:
                    return ''
                # 核心！！！
                # 注意此时在for循环里只是不停的将当前最多的字符pop出来
                # 这样就保证了在for循环中一定会添加k个不重复的字符
                # 用temp变量先存着这个pop出来并用过的(count, ch)
                # 在退出for循环后在push到堆里
                count, ch = heappop(hp)
                count *= -1
                res.append(ch)
                count -= 1
                # 如果用光了，就没有必要再push到堆中了
                if count > 0:
                    temp.append((-count, ch))
                n -= 1
            for each in temp:
                heappush(hp, each)
        
        return ''.join(res)
```

#### 360. Sort Transformed Array
```
class Solution:
    def sortTransformedArray(self, nums, a, b, c):
        """
        :type nums: List[int]
        :type a: int
        :type b: int
        :type c: int
        :rtype: List[int]
        """
        res = []

        if a == 0:
            res = [b * i + c for i in nums]
            return res if b > 0 else res[::-1]
        else:
            l, r = 0, len(nums) - 1
            axis = -float(b) / (2 * a)
            while l <= r:
                # 根据中轴线决定折半方向
                if abs(nums[l] - axis) >= abs(nums[r] - axis):
                    res.append(int(a * nums[l] ** 2 + b * nums[l] + c))
                    l += 1
                else:
                    res.append(int(a * nums[r] ** 2 + b * nums[r] + c))
                    r -= 1
            return res if a < 0 else res[::-1]
```

#### 364. Nested List Weight Sum II
```
class Solution:
    def depthSumInverse(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """

        ## http://www.cnblogs.com/grandyang/p/5615583.html
        ## Leetcode - StefanPochmann
        weighted = unweighted = 0
        
        while nestedList:
            next_level_list = []
            for each in nestedList:
                if each.isInteger():
                    unweighted += each.getInteger()
                else:
                    next_level_list += each.getList()
            weighted += unweighted
            nestedList = next_level_list
        
        return weighted
```

#### 366. Find Leaves of Binary Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def findLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        res = []
        self._dfs(root, res)
        return res
    
    # 返回的是以node为根的树的最大深度
    def _dfs(self, node, results):
        if not node:
            return 0
        
        current_level = 1 + max(
            self._dfs(node.left, results),
            self._dfs(node.right, results),
        )
        
        if current_level > len(results):
            # 当前的层数只可能比results的长度最多大1
            assert current_level == len(results) + 1
            results.append([])
        
        # 知道了当前深度值就可以将节点值加入到结果res中的正确位置了
        results[current_level - 1].append(node.val)
        return current_level

# class Solution:
#     def findLeaves(self, root):
#         """
#         :type root: TreeNode
#         :rtype: List[List[int]]
#         """
#         res = []
#         curr = root
#         while curr:
#             leaves = []
#             curr = self._dfs(curr, leaves)
#             res.append(leaves)
#         return res
    
#     def _dfs(self, node, leaves):
#         if not node:
#             return
#         if not node.left and not node.right:
#             leaves.append(node.val)
#             return
#         node.left = self._dfs(node.left, leaves)
#         node.right = self._dfs(node.right, leaves)
#         return node
```

#### 367. Valid Perfect Square
```
class Solution:
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        
        if num <= 1:
            return True
        
        start, end = 1, num // 2
        while start + 1 < end:
            mid = start + (end - start) // 2
            if mid * mid == num:
                return True
            elif mid * mid < num:
                start = mid + 1
            else:
                end = mid - 1
        
        if start * start == num:
            return True
        
        if end * end == num:
            return True
        
        return False
```

#### 371. Sum of Two Integers
```
class Solution:
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        # 这道题Python做没什么意义
        # Python没有无符号右移操作
        # 并且当左移操作的结果超过最大整数范围时
        # 会自动将int类型转换为long类型
        # 下面代码通不过
        # if b == 0:
        #     return a
        # total = a ^ b
        # carry = (a & b) << 1
        # return self.getSum(total, carry)
        
        return sum([a, b])
```

#### 373. Find K Pairs with Smallest Sums
```
from heapq import heappush
from heapq import heappop

class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """

        if not nums1 or not nums2:
            return []
        
        size1, size2 = len(nums1), len(nums2)
        res = []
        hp = []
        
        for i in range(size1):
            heappush(hp, (nums1[i] + nums2[0], i, 0))
        
        while len(res) < min(k, size1 * size2):
            total, i, j = heappop(hp)
            res.append([nums1[i], nums2[j]])
            if j + 1 < size2:
                heappush(hp, (nums1[i] + nums2[j + 1], i, j + 1))
        
        return res
```

#### 377. Combination Sum IV
```
# DP AC
class Solution:
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # dp[i]定义：凑成i这个数字可以有多少种凑法
        dp = [0] * (target + 1)
        dp[0] = 1
        for i in range(1, target + 1):
            for num in nums:
                if i >= num:
                    # 当前想要凑成i，则直接使用i - num的凑法就好了
                    # 当前的数字是num
                    dp[i] += dp[i - num]
        
        return dp[-1]
        
# DFS (TLE)
# class Solution:
#     def combinationSum4(self, nums, target):
#         """
#         :type nums: List[int]
#         :type target: int
#         :rtype: int
#         """
#         res = []
#         self._dfs(nums, target, 0, [], res)
#         return len(res)
    
#     def _dfs(self, nums, target, start, curr, res):
#         if target == 0:
#             res.append(curr[:])
#             return
        
#         if target < 0:
#             return
        
#         for i in range(len(nums)):
#             curr.append(nums[i])
#             self._dfs(nums, target - nums[i], i, curr, res)
#             curr.pop()
```

#### 378. Kth Smallest Element in a Sorted Matrix
```
from bisect import bisect_right

class Solution:
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        bisect_right: 比如有[1,2,3]，想要插入1
        返回坐标1
        而bisect_left会返回0
        如果要插入1.5， bisect_left和bisect_right都会返回1
        """
        low, high = matrix[0][0], matrix[-1][-1]
        
        while low <= high:
            mid = low + (high - low) // 2
            loc_sum = sum(bisect_right(m, mid) for m in matrix)
            if loc_sum >= k:
                high = mid - 1
            else:
                low = mid + 1

        return low
```

#### 380. Insert Delete GetRandom O(1)
```
from random import choice

class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = []
        self.hash = {}

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.hash:
            self.data.append(val)
            self.hash[val] = len(self.data) - 1
            return True
        else:
            return False
        
    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.hash:
            # if self.hash[val] == len(self.data) - 1:
            if val == self.data[-1]:
                self.data.pop()
            else:
                ## 为了保证O(1)
                ## 将data中最后一个pop出，保存
                ## 然后覆盖掉data里真正需要pop的value
                ## 别忘了修改hash
                temp = self.data.pop()
                self.data[self.hash[val]] = temp
                self.hash[temp] = self.hash[val]
            del self.hash[val]
            return True
        else:
            return False

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return choice(self.data)

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

#### 381. Insert Delete GetRandom O(1) - Duplicates allowed, Hard, Facebook, Linkedin
```
# 感觉是对的
# lintcode能通过
# leetcode通不过？
from random import choice

class RandomizedCollection:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.map = {}
        self.nums = []

    def insert(self, val):
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        self.nums.append(val)
        if val in self.map:
            self.map[val].append(len(self.nums) - 1)
            return False
        else:
            self.map[val] = [len(self.nums) - 1]
            return True

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.map:
            pos = self.map[val].pop()
            if not self.map[val]:
                del self.map[val]
            if pos != len(self.nums) - 1:
                self.map[self.nums[-1]][-1] = pos
                self.nums[pos], self.nums[-1] = self.nums[-1], self.nums[pos]
            self.nums.pop()
            return True
        else:
            return False

    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """
        import random
        return random.choice(self.nums)


# Your RandomizedCollection object will be instantiated and called as such:
# obj = RandomizedCollection()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()
```

#### 382. Linked List Random Node
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
from random import randint

class Solution:

    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        self.head = head

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        # 蓄水池典型题
        res = self.head
        node = self.head.next
        index = 1
        while node:
            # 这里的0就是hard coded一个点，fix住以后用来确定if的条件的
            if randint(0, index) == 0:
                res = node
            node = node.next
            index += 1
        return res.val

# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()
```

#### 384. Shuffle an Array
```
from random import randint

class Solution:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self._data = nums[:]
        self._base = nums[:]

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        self._data = self._base[:]
        return self._data

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        for x in range(len(self._data)):
            # 这里的思路很重要！！！
            index = randint(0, x)
            self._data[x], self._data[index] = self._data[index], self._data[x]
        return self._data

# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()
```

#### 386. Lexicographical Numbers
```
class Solution:
    def lexicalOrder(self, n):
        """
        :type n: int
        :rtype: List[int]
        给一个整数n，把区间[1,n]的所有数字按照字典顺序来排列
        """
        # 类似先序遍历，先将当前遍历的数字加入结果
        # 再进行处理
        # 对每一个数字，我们有两种处理的选择：
        # 1是优先将数字乘以10
        # 2是如果数字末尾<9，考虑将数字加1
        res = []
        self._dfs(n, 1, res)
        return res
    
    def _dfs(self, n, curr, res):
        res.append(curr)
        if curr * 10 <= n:
            self._dfs(n, curr * 10, res)
        # curr % 10就是当前curr的末尾的数字
        if curr < n and curr % 10 < 9:
            self._dfs(n, curr + 1, res)
```

#### 387. First Unique Character in a String
```
from collections import Counter

class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s:
            return -1
        
        mapping = Counter(s)
        for i in range(len(s)):
            if mapping[s[i]] == 1:
                return i
        
        return -1
```

#### 391. Perfect Rectangle
```
from collections import defaultdict

class Solution:
    def isRectangleCover(self, rectangles):
        """
        :type rectangles: List[List[int]]
        :rtype: bool
        """
        left = min(x[0] for x in rectangles)
        bottom = min(x[1] for x in rectangles)                
        right = max(x[2] for x in rectangles)
        top = max(x[3] for x in rectangles)

        points = defaultdict(int)
        for l, b, r, t in rectangles:
            a, b, c, d = (l, b), (r, b), (r, t), (l, t)
            for p, q in zip((a, b, c, d), (1, 2, 4, 8)):
                if points[p] == q:
                    return False
                points[p] |= q
        
        for px, py in points:
            # px和py不能是边界点
            # 便捷点可以为1， 2， 3， 4
            # 剩下的点一定在3, 6, 9, 12, 15中如果他们能组成完美矩形的话
            if left < px < right or bottom < py < top:
                if points[(px, py)] not in (3, 6, 9, 12, 15):
                    return False
        
        return True
```

#### 392. Is Subsequence
```
class Solution:
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if not s:
            return True
        
        # i是s的index j是t的index
        i = j = 0
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
            j += 1
        
        return i == len(s)
```

#### 393. UTF-8 Validation
```
class Solution:
    def validUtf8(self, nums):
        """
        :type data: List[int]
        :rtype: bool
        """
        # 这道题data list里每一个数字都是一个8位int（0~255）
        # UTF-8 是 Unicode 的实现方式之一。
        # UTF-8 最大的一个特点，就是它是一种变长的编码方式。
        # 它可以使用1~4个字节表示一个符号，根据不同的符号而变化字节长度。
        # UTF-8 的编码规则很简单，只有二条：
        # 1）对于单字节的符号，字节的第一位设为0，后面7位为这个符号的 Unicode 码
        # 因此对于英语字母，UTF-8 编码和 ASCII 码是相同的。
        # 2）对于n字节的符号（n > 1），第一个字节的前n位都设为1，第n + 1位设为0，
        # 后面字节的前两位一律设为10。剩下的没有提及的二进制位，全部为这个符号的 Unicode 码。
        start = 0
        while start < len(nums):
            first = nums[start]
            if first >> 7 == 0:
                start += 1
            # 下面3个elif都是指上面的第二条
            elif first >> 5 == 0b110 and self._check(nums, start, 2):
                # 表示当前UTF-8是用了2个字节
                start += 2
            elif first >> 4 == 0b1110 and self._check(nums, start, 3):
                # 表示当前UTF-8是用了3个字节
                start += 3
            elif first >> 3 == 0b11110 and self._check(nums, start, 4):
                # 表示当前UTF-8是用了4个字节
                start += 4
            else:
                return False
        
        return True
    
    def _check(self, nums, start, size):
        for i in range(start + 1, start + size):
            # 第一个条件指的是在UTF-8的声明里说了当前要用size个字节
            # 但是遍历不到（已经越界了）所以不符合条件，直接return False
            if i >= len(nums) or nums[i] >> 6 != 0b10:
                return False
        return True
```

#### 394. Decode String
```
class Solution:
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        # 把所有字符一个个放到 stack 里
        # 如果碰到了]，就从 stack 找到对应的字符串和重复次
        # decode 之后再放回 stack 里
        stack = []
        for ch in s:
            if ch != ']':
                stack.append(ch)
            else:
                # 先求中括号里的string
                strs = []
                while stack and stack[-1] != '[':
                    strs.append(stack.pop())
                
                # 去除 '[''
                stack.pop()

                # 再求出左中括号外面的数字（需要重复中括号里字符串的次数）
                repeat = 0
                base = 1
                while stack and '0' <= stack[-1] <= '9':
                    # 注意这里需要使用base的
                    # 因为stack是从后往前pop
                    # 所以当前pop出来的0-9数字需要乘以base
                    repeat += int(stack.pop()) * base
                    base *= 10
                
                curr_str = ''.join(strs[::-1])
                stack.append(curr_str * repeat)
        
        return ''.join(stack)
```

#### 395. Longest Substring with At Least K Repeating Characters
```
# 解法 1
# leetcode TLE，卡在最后一个大test cases上
# 但是解法思路是对的
class Solution:
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        # 这道题的题意就卡了很久
        # 实际上说的是找去找一个子串
        # 这个子串里的每一种出现的字符
        # 出现的次数都要大于k
        
        # 1. 常规解法， 类似two pointers, i是起点，j是终点
        # 由于字母只有26个，而整型mask有32位，足够用了
        # 每一位代表一个字母，如果为1，表示该字母不够k次，如果为0就表示已经出现了至少k次
        res = 0
        n = len(s)
        i = 0
        while i + k <= n:
            m = [0] * 26
            # mask = 0初始意味着32位上全是0
            mask = 0
            max_idx = i
            for j in range(i, n):
                t = ord(s[j]) - ord('a')
                m[t] += 1
                if m[t] < k:
                    # 将mask的二进制第t位设位1
                    mask |= 1 << t
                else:
                    # 将mask的二进制第t位设位0
                    mask &= ~(1 << t)
                # 0就说明符合了条件
                if mask == 0:
                    res = max(res, j - i + 1)
                    max_idx = j
            i = max_idx + 1
        
        return res

# 解法 2 （书影博客）
# from re import split
# from collections import Counter

# class Solution:
#     def longestSubstring(self, s, k):
#         """
#         :type s: str
#         :type k: int
#         :rtype: int
#         """
#         cnt = Counter(s)
#         filters = [x for x in cnt if cnt[x] < k]
#         if not filters:
#             return len(s)
#         tokens = split('|'.join(''.join(filters)), s)
#         return max(self.longestSubstring(token, k) for token in tokens)
```

#### 398. Random Pick Index
```
from collections import defaultdict
from random import choice

class Solution:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.mapping = defaultdict(list)
        for i, num in enumerate(nums):
            self.mapping[num].append(i)


    def pick(self, target):
        """
        :type target: int
        :rtype: int
        """
        if target in self.mapping:
            return choice(self.mapping[target])
        
# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.pick(target)
```

#### 399. Evaluate Division
```
from collections import defaultdict

class Solution:
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        Example:
        Given a / b = 2.0, b / c = 3.0. 
        queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? . 
        return [6.0, 0.5, -1.0, 1.0, -1.0 ].
        """
        # defaultdict的第一个参数必须是callable的
        graph = defaultdict(lambda: defaultdict(float))
        
        for (s, t), v in zip(equations, values):
            graph[s][t] = v
            graph[t][s] = 1.0 / v
    
        for k in graph:
            graph[k][k] = 1.0
            for i in graph:
                for j in graph:
                    if graph[i][k] and graph[k][j]:
                        graph[i][j] = graph[i][k] * graph[k][j]

        res = []
        for i, j in queries:
            if graph[i][j]:
                res.append(graph[i][j])
            else:
                res.append(-1.0)
        
        return res
```

#### 402. Remove K Digits
```
class Solution:
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        # 核心：维护一个单调递增栈
        stack = []
        for each in num:
            while k > 0 and stack and stack[-1] > each:
                stack.pop()
                k -= 1
            stack.append(each)
        
        # 根据while条件，到最后很可能k大于0（意味着我们并没有删除足够数目的字符）
        # 注意此时stack的长度肯定是大于的等于k的
        # 所以我们可以做一次切片，只取出长度为len(num) - k
        stack = stack[:len(stack) - k]
        
        # 这道题有点小坑：
        # 对于10200 k = 1的情况
        # 删除1以后，剩余0200等效于数字200
        # 所以我们要去除头部的0
        return str(int(''.join(stack))) if stack else '0'
```

#### 403. Frog Jump, Hard, Facebook
```
class Solution:
    def canCross(self, stones):
        """
        :type stones: List[int]
        :rtype: bool
        """
        # 感觉这道题确实是深度优先的思路
        # stone_hash里key表示当前stone的值
        # value是一个set，表示可以从几步远的地方跳过来！！！这个定义很重要
        stones_hash = {}
        for stone in stones:
            stones_hash[stone] = set()
            
        stones_hash[0].add(0)

        # 实际上在循环里就是在不停的往stones_hash的后面的石头里填充jumps
        for stone in stones:
            for jumps in stones_hash[stone]:
                # 表示当前是可以到达的
                # 这个jumps - 1的条件不可或缺
                # 否则当[0, 1...]头两块石头就会出错！！
                # 因为在1石头上会在循环中更新pos 1位置的set（额外加上一个0）
                if jumps - 1 > 0 and stone + jumps - 1 in stones_hash:
                    stones_hash[stone + jumps - 1].add(jumps - 1)
                if stone + jumps in stones_hash:
                    stones_hash[stone + jumps].add(jumps)
                if stone + jumps + 1 in stones_hash:
                    stones_hash[stone + jumps + 1].add(jumps + 1)
        
        return len(stones_hash[stones[-1]]) > 0
```

#### 407. Trapping Rain Water II
```
from heapq import heappush
from heapq import heappop

class Solution:
    def trapRainWater(self, heightMap):
        """
        :type heightMap: List[List[int]]
        :rtype: int
        """
        # 这道题是BFS，但是是使用优先队列而不是简单的队列
        if not heightMap or not heightMap[0]:
            return 0

        m, n = len(heightMap), len(heightMap[0])
        min_heap = []
        visited = [[False] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if i in {0, m - 1} or j in {0, n - 1}:
                    heappush(min_heap, (heightMap[i][j], i, j))
                    visited[i][j] = True

        DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        res = 0
        while min_heap:
            curr_sea_level, ci, cj = heappop(min_heap)
            for di, dj in DIRS:
                newi, newj = ci + di, cj + dj
                if not 0 <= newi < m or not 0 <= newj < n or \
                    visited[newi][newj]:
                    continue
                if curr_sea_level - heightMap[newi][newj] > 0:
                    res += curr_sea_level - heightMap[newi][newj]
                    # 核心：这里我们选择当前最小的高度作为当前newi和newj的海平面
                    # 而不是简单的直接把heightMap[newi][newj]作为当前的海平面
                    heappush(min_heap, (curr_sea_level, newi, newj))
                else:
                    heappush(min_heap, (heightMap[newi][newj], newi, newj))
                visited[newi][newj] = True
        
        return res
```

#### 408. Valid Word Abbreviation
```
class Solution:
    def validWordAbbreviation(self, word, abbr):
        """
        :type word: str
        :type abbr: str
        :rtype: bool
        """
        iw = ia = 0
        w, a = len(word), len(abbr)
        
        while iw < w and ia < a:
            if abbr[ia].isdigit():
                # 这里的abbr[ia]应该是数字子串的开头
                # 所以不能是0
                if abbr[ia] == '0':
                    return False
                num = 0
                while ia < a and abbr[ia].isdigit():
                    num = 10 * num + int(abbr[ia])
                    ia += 1
                iw += num
            else:
                if word[iw] != abbr[ia]:
                    return False
                iw += 1
                ia += 1
        
        # 到最后如果匹配，iw和ia应该相等
        # 表示两个字符串都全部匹配上了
        return iw == w and ia == a
```

#### 410. Split Array Largest Sum
```
class Solution:
    def splitArray(self, nums, m):
        """
        :type nums: List[int]
        :type m: int
        :rtype: int
        """
        # 很巧妙的二分思想
        l = r = 0
        for num in nums:
            l = max(l, num)
            r += num
        
        # l就是数组中的最大值
        # r是数组的和
        # 这道题二分的范围就是nums中子数组之和的范围
        while l < r:
            mid = l + (r - l) // 2
            # 核心：如果不能拆成了大于m个分组，说明我们的子数组的和mid太大了
            if self._can_split_no_more_than_m(nums, m, mid):
                r = mid
            else:
                l = mid + 1
        
        return l
    
    def _can_split_no_more_than_m(self, nums, m, sub_total):
        cnt = 1
        curr_sum = 0
        for num in nums:
            curr_sum += num
            if curr_sum > sub_total:
                # 说明此时已经找到一个最大（和不超过sub_total）的分组
                # 要重置了
                curr_sum = num
                cnt += 1
                # 如果curr_sum很快速的凑成了m个不大于sub_total的子数组
                # 说明我们的sub_total太小了
                if cnt > m:
                    return False
        return True
```

#### 412. Fizz Buzz
```
class Solution:
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                res.append('FizzBuzz')
            elif i % 3 == 0:
                res.append('Fizz')
            elif i % 5 == 0:
                res.append('Buzz')
            else:
                res.append(str(i))
        return res
```

#### 413. Arithmetic Slices
```

```

#### 415. Add Strings
```
class Solution:
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        res = ''
        m, n = len(num1), len(num2)
        i, j = m - 1, n - 1
        carry = 0
        
        while i >= 0 or j >= 0:
            a = ord(num1[i]) - ord('0') if i >= 0 else 0
            b = ord(num2[j]) - ord('0') if j >= 0 else 0
            total = a + b + carry
            res = str(total % 10) + res
            carry = total // 10
            i -= 1
            j -= 1
        
        if carry == 1:
            return '1' + res
        return res
```

#### 416. Partition Equal Subset Sum
```
class Solution:
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        # 最优解应该是DP
        total = sum(nums)
        if total % 2 != 0:
            return False
        target = total // 2
        # dp[i]定义：数字i能否用任意一个nums中的子集之和来表示
        dp = [False] * (target + 1)
        dp[0] = True
        for num in nums:
            for i in range(target, num - 1, -1):
                # 对于当前要找的数字i
                # 如果i-num可以用nums中的子集之和来表示
                # 则i-num+num=i也可以用nums中的子集之和来表示
                # 因为外层循环用的是num
                # 直接加上当前这个num就好了
                dp[i] |= dp[i - num]
        return dp[-1]

        # 可能比较好理解
        # total = sum(nums)
        # if total % 2 != 0:
        #     return False
        # nset = set([0])
        # for n in nums:
        #     for m in nset.copy():
        #         nset.add(m + n)
        # return total // 2 in nset
```

#### 419. Battleships in a Board
```
class Solution:
    def countBattleships(self, board):
        """
        :type board: List[List[str]]
        :rtype: int
        这道题是里的battleship都是矩形的
        """
        if not board or not board[0]:
            return 0
        
        m, n = len(board), len(board[0])
        res = 0
        # i j表示舰船的开头
        # 则如果是开头，其左边和上边一定没有x
        for i in range(m):
            for j in range(n):
                if board[i][j] == '.':
                    continue
                else:
                    if j > 0 and board[i][j - 1] == 'X':
                        continue
                    if i > 0 and board[i - 1][j] == 'X':
                        continue
                    res += 1
        
        return res
```

#### 426. Convert Binary Search Tree to Sorted Doubly Linked List
```
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return
        
        first = last = None
        stack = []
        
        curr = root
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            if not first:
                first = curr
            if last:
                last.right = curr
                curr.left = last
            last = curr
            curr = curr.right
        
        first.left = last
        last.right=  first
        return first
```

#### 428. Serialize and Deserialize N-ary Tree, Hard, Facebook, Linkedin
```
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
"""
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: Node
        :rtype: str
        """
        res = []
        self._pre_order(root, res)
        return ' '.join(res)
    
    def _pre_order(self, node, res):
        if not node:
            return
        
        res.append(str(node.val))
        for child in node.children:
            self._pre_order(child, res)
        
        # indicates no more children
        # continue serialization from parent
        res.append('#')

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: Node
        """
        if not data:
            return
        
        tokens = data.split(' ')
        root = Node(int(tokens.pop(0)), [])
        
        self._bfs(root, tokens)
        return root
    
    def _bfs(self, node, tokens):
        if not tokens:
            return
        
        # add child nodes with subtrees
        while tokens[0] != '#':
            value = tokens.pop(0)
            child = Node(int(value), [])
            node.children.append(child)
            self._bfs(child, tokens)
        
        # discard the "#"
        tokens.pop(0)

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```

#### 430. Flatten a Multilevel Doubly Linked List
```
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""
class Solution(object):
    def flatten(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        curr = head
        while curr:
            if curr.child:
                temp = curr.next
                last = curr.child
                while last.next:
                    last = last.next
                curr.next = curr.child
                curr.next.prev = curr
                curr.child = None
                # 不好理解的地方
                # last是本次循环到达的最后node
                # temp是预先存起来的curr的next node
                # 所以last的下一个应该是这个temp
                # 同时temp的前一个就是last（因为是双向链表）
                # 这题重点：需要拿到这个last指针
                last.next = temp
                if temp:
                    temp.prev = last
            curr = curr.next
        
        return head
```

#### 432. All O`one Data Structure, Hard, Facebook, Linkedin
```
from collections import defaultdict

class Node:
    def __init__(self):
        self.key_set = set()
        self.prev = self.next = None
    
    def add_key(self, key):
        self.key_set.add(key)
    
    def remove_key(self, key):
        if key in self.key_set:
            self.key_set.remove(key)
    
    def get_any_key(self):
        if self.key_set:
            result = self.key_set.pop()
            self.add_key(result)
            return result
    
    def count(self):
        return len(self.key_set)
    
    def is_empty(self):
        return len(self.key_set) == 0

class DoubleLinkedList:
    def __init__(self):
        # head_node和tail_node都是dummy_node
        # 因为需要双向插入删除节点
        self.head_node = Node()
        self.tail_node = Node()
        self.head_node.next = self.tail_node
        self.tail_node.prev = self.head_node
    
    def insert_after(self, x):
        new_node = Node()
        temp = x.next
        x.next = new_node
        new_node.next = temp
        new_node.prev = x
        temp.prev = new_node
        return new_node
    
    def insert_before(self, x):
        new_node = Node()
        temp = x.prev
        x.prev = new_node
        new_node.prev = temp
        new_node.next = x
        temp.next = new_node
        return new_node
    
    def remove(self, x):
        prev_node = x.prev
        prev_node.next = x.next
        x.next.prev = prev_node
    
    def get_head(self):
        return self.head_node.next

    def get_tail(self):
        return self.tail_node.prev
    
    def get_dummy_head(self):
        return self.head_node
    
    def get_dummy_tail(self):
        return self.tail_node

class AllOne:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # 这里的双向链表
        # head是当前frequency最小的node
        # tail是当前frequency最大的node
        self.dll = DoubleLinkedList()
        # key到frequency的映射
        self.key_counter = defaultdict(int)
        # frequency到某个节点的映射
        # 而这个节点用来承装所有的key（使用key_set）
        self.node_freq = {0: self.dll.get_dummy_head()}

    # pf就是前一个frequency的意思
    # 能执行到下面的函数
    # 说明pf一定已经在node_freq里面了
    # 所以不需要再检查
    def _remove_key_pf_node(self, pf, key):
        node = self.node_freq[pf]
        node.remove_key(key)
        if node.is_empty():
            self.dll.remove(node)
            self.node_freq.pop(pf)
        
    def inc(self, key):
        """
        Inserts a new key <Key> with value 1. Or increments an existing key by 1.
        :type key: str
        :rtype: void
        """
        self.key_counter[key] += 1
        cf = self.key_counter[key]
        pf = self.key_counter[key] - 1
        if cf not in self.node_freq:
            self.node_freq[cf] = self.dll.insert_after(self.node_freq[pf])
        self.node_freq[cf].add_key(key)
        # 如果pf为0，对应的是dummy_head
        # 所以不需要任何操作
        if pf > 0:
            self._remove_key_pf_node(pf, key)
        
    def dec(self, key):
        """
        Decrements an existing key by 1. If Key's value is 1, remove it from the data structure.
        :type key: str
        :rtype: void
        """
        if key in self.key_counter:
            self.key_counter[key] -= 1
            cf = self.key_counter[key]
            pf = self.key_counter[key] + 1
            if self.key_counter[key] == 0:
                self.key_counter.pop(key)
            # cf不可能小于0
            if cf > 0:
                if cf not in self.node_freq:
                    self.node_freq[cf] = self.dll.insert_before(self.node_freq[pf])
                self.node_freq[cf].add_key(key)
            self._remove_key_pf_node(pf, key)
            
    def getMaxKey(self):
        """
        Returns one of the keys with maximal value.
        :rtype: str
        """
        if not self.dll.get_tail().is_empty():
            return self.dll.get_tail().get_any_key()
        return ''

    def getMinKey(self):
        """
        Returns one of the keys with Minimal value.
        :rtype: str
        """
        if not self.dll.get_head().is_empty():
            return self.dll.get_head().get_any_key()
        return ''

# Your AllOne object will be instantiated and called as such:
# obj = AllOne()
# obj.inc(key)
# obj.dec(key)
# param_3 = obj.getMaxKey()
# param_4 = obj.getMinKey()
```

#### 435. Non-overlapping Intervals
```
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        问最少移除多少个区间，能使的剩下的区间non-overlapping
        """
        n = len(intervals)
        ins = [[interval.start, interval.end] for interval in intervals]
        ins.sort()
        res = 0
        last = 0
        
        for i in range(1, n):
            start, end = ins[i]
            last_start, last_end = ins[last]
            if start < last_end:
                # 此时要移除一个了
                # 实际上并没有移除，但是将last的位置变成当前end最大的那个i
                # 为了保证我们总体去掉的区间数最小，我们去掉那个end值较大的区间
                # end值较大的区间意味着可能会overlapping后面的区间
                res += 1
                if end < last_end:
                    last = i
            else:
                last = i
        
        return res
```

#### 437. Path Sum III
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        # 这里的pathSum定义是可以包括也可以不包括当前root的sum种类数目
        if not root:
            return 0
        return self.pathSum(root.left, sum) + \
            self.pathSum(root.right, sum) + \
            self._contain_root(root, sum)
    
    # _contain_root的定义就是必须包括当前root的sum种类数目
    def _contain_root(self, root, sum):
        if not root:
            return 0
        res = 0
        if root.val == sum:
            res += 1
        res += self._contain_root(root.left, sum - root.val) + \
            self._contain_root(root.right, sum - root.val)
        return res
```

#### 438. Find All Anagrams in a String
```
from collections import defaultdict

class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        if not s or not p:
            return []
        
        mapping = defaultdict(int)
        for ch in p:
            mapping[ch] += 1
            
        l = r = 0
        required = len(mapping)
        res = []
        while r < len(s):
            # 把这个mapping理解成“需求”
            # 所以当左边指针扩大窗口范围的时候
            # 相当于potentially“需求”小了一个
            mapping[s[r]] -= 1
            if mapping[s[r]] == 0:
                required -= 1

            while required == 0:
                mapping[s[l]] += 1
                if mapping[s[l]] > 0:
                    required += 1
                # 重要：这里是要通过长度判断是否找到了一个p的anagram
                # 两个坐标值差再加1才是两个坐标（左必右闭包括这两个坐标）之间所有的元素个数
                if r - l + 1 == len(p):
                    res.append(l)
                l += 1
            
            r += 1
        
        return res
```

#### 445. Add Two Numbers II
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        s1 = []
        s2 = []
        
        while l1:
            s1.append(l1.val)
            l1 = l1.next
        
        while l2:
            s2.append(l2.val)
            l2 = l2.next
        
        res_val = 0
        res = ListNode(0)
        
        while s1 or s2:
            if s1:
                res_val += s1.pop()
            if s2:
                res_val += s2.pop()
            res.val = res_val % 10
            curr_head = ListNode(res_val // 10)
            curr_head.next = res
            res = curr_head
            # 这里是必须的,因为当前的个位数字已经被带入到之前的res node里了
            # 只剩下要处理的十位上的数字（如果存在的话）
            res_val //= 10
        
        return res if res.val != 0 else res.next
```

#### 446. Arithmetic Slices II - Subsequence, Hard, Facebook
```
from collections import defaultdict

class Solution:
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        if not A:
            return 0

        n = len(A)
        # 这里的dp[0]永远是一个空的dict
        # 因为以索引0结尾的只是一个数，不会有等差数列
        # 这里的dp[i]是一个hashmap
        # 表示以i结尾的数组存在以diff为等差的等差数列的个数
        dp = [defaultdict(int) for _ in range(n)]
        res = 0
        for i in range(1, n):
            for j in range(i):
                diff = A[i] - A[j]
                # 首先i和j这两个数字就能组成一个等差数列
                dp[i][diff] += 1
                # 如果diff已经在之前的循环里出现过
                # 说明dp[j][diff]已经能组成一些等差数列
                # 里面每一个都可以和i再组成一个
                # 所以这里面要再加上dp[j][diff]
                if diff in dp[j]:
                    # 注意只有此时res才会被更新
                    # 因为发现了之前存在一个diff
                    # 之前如果存在一个diff，则一定是至少两个数才能计算diff
                    # 现在又发现一个新的
                    # 则凑成的数列长度至少为三个了
                    # 所以可以来更新res
                    dp[i][diff] += dp[j][diff]
                    res += dp[j][diff]
        
        return res
```

#### 449. Serialize and Deserialize BST
```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        res = []
        self._serialize(root, res)
        return ', '.join(res)
    
    def _serialize(self, node, res):
        if not node:
            res.append('#')
            return
        res.append(str(node.val))
        self._serialize(node.left, res)
        self._serialize(node.right, res)
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        node_list = data.split(', ')
        return self._deserialize(node_list)
        
    def _deserialize(self, node_list):
        if not node_list:
            return
        
        # 核心：这里要pop(0)，注意将node_list当成deque来用
        curr = node_list.pop(0)
        if curr == '#':
            return
        
        root = TreeNode(int(curr))
        root.left = self._deserialize(node_list)
        root.right = self._deserialize(node_list)
        return root

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
```

#### 450. Delete Node in a BST
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        
        if not root:
            return None
        
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left or not root.right:
                root = root.left if root.left else root.right
            else: # 此时left和right都存在
                curr = root.right
                # 在root的右子树中寻找最小值
                while curr.left:
                    curr = curr.left
                root.val = curr.val
                root.right = self.deleteNode(root.right, curr.val)
        
        return root
```

#### 451. Sort Characters By Frequency
```
from heapq import heappush
from heapq import heappop
from collections import defaultdict

class Solution:
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        
        mapping = defaultdict(int)
        res_hp = []
        
        for ch in s:
            mapping[ch] += 1
        
        for ch, cnt in mapping.items():
            # python默认是最小堆
            heappush(res_hp, (cnt, -ord(ch)))
        
        res = ''
        while res_hp:
            val, ch = heappop(res_hp)
            res = chr(-ch) * val + res
        
        return res
```

#### 465. Optimal Account Balancing
```
from collections import defaultdict

class Solution:
    def minTransfers(self, transactions):
        """
        :type transactions: List[List[int]]
        :rtype: int
        要求的是最少的transaction次数
        """
        # 实际上这个mapping并没大用
        # 就是为了求有多少个不平衡的账户acconts而已
        mapping = defaultdict(int)
        for acc1, acc2, money in transactions:
            mapping[acc1] -= money
            mapping[acc2] += money
            
        accounts = []
        # cnt统计的是accounts现在有多少个不平衡的账户
        for acc, money in mapping.items():
            if money != 0:
                accounts.append(money)
        
        return self._dfs(accounts, 0, 0)
    
    def _dfs(self, accounts, start, old_steps):
        new_steps = 2 ** 31 - 1
        
        # 先找第一个不平衡的账户的位置
        # 注意第一次进入这个递归函数的时候，start肯定是0
        # 但是后面由于在for循环中更新了account的元素就不一定了
        # 所以我们这里需要再求一下
        while start < len(accounts) and accounts[start] == 0:
            start += 1
        
        # 核心递归：遍历所有account里start以后的不平衡情况
        # 查看用某个账户来平衡start能否最终能使的steps变小
        for i in range(start + 1, len(accounts)):
            # 此时异号，说明双方可以来平衡
            # 很好理解：两个账户都是正的或者都是负的没有意义
            # 从这个角度来讲这是贪心的思路
            if accounts[start] * accounts[i] < 0:
                accounts[i] += accounts[start]
                new_steps = min(new_steps, self._dfs(accounts, start + 1, old_steps + 1))
                # 一定记得要回溯！！！
                # 我们在这里是在尝试不同的情况
                accounts[i] -= accounts[start]
        
        return new_steps if new_steps != 2 ** 31 - 1 else old_steps
```

#### 516. Longest Palindromic Subsequence
```
class Solution:
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        
        for i in range(n - 1, -1, -1):
            dp[i][i] = 1
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        
        return dp[0][n - 1]
```

#### 452. Minimum Number of Arrows to Burst Balloons
```
class Solution:
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        # 每个气球都有一个范围start到end
        # 能打爆某个气球的箭在start和end之间
        if not points:
            return 0

        points.sort()
        res = 1
        end = points[0][1]
        for curr_start, curr_end in points[1:]:
            if end >= curr_start:
                # 不能是max
                # 比如例子[[1, 6], [2, 8], [7, 12], [10, 16]]
                # 本来第一只箭在1和6之间
                # 如果max的话就会把end变成8
                # 导致7和12这个范围也被认为可以被第一只箭包括
                # 从而使得res值少了
                end = min(end, curr_end)
            else:
                end = curr_end
                res += 1
        
        return res
```

#### 460. LFU Cache
# 这道题不是太懂，抄的leetcode discussion解答
```
from collections import defaultdict

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = self.nxt = None 

class DoubleLinkedList:
    def __init__(self):
        self.head_sentinel = Node(None, None)
        self.tail_sentinel = Node(None, None)
        self.count = 0
        self.head_sentinel.nxt = self.tail_sentinel
        self.tail_sentinel.prev = self.head_sentinel
        self.count = 0
    
    def insert(self, x, node):
        temp = x.nxt
        x.nxt, node.prev = node, x
        node.nxt, temp.prev = temp, node
        self.count += 1

    def appendleft(self, node):
        self.insert(self.head_sentinel, node)

    def append(self, node):
        self.insert(self.get_tail(), node)

    def remove(self, node):
        prev_node = node.prev
        prev_node.nxt, node.nxt.prev = node.nxt, prev_node
        self.count -= 1

    def pop():
        self.remove(self.get_tail())

    def popleft():
        self.remove(self.get_head())

    def size(self):
        return self.count

    def get_head(self):
        return self.head_sentinel.nxt if self.count > 0 else None
    
    def get_tail(self):
        return self.tail_sentinel.prev if self.count > 0 else None        

class LinkedHashSet:
    def __init__(self):
        self.node_map, self.dll = {}, DoubleLinkedList()

    def size(self):
        return len(self.node_map)
        
    def contains(self, key):
        return key in self.node_map
    
    def search(self, key):
        return self.node_map[key].value

    def appendleft(self, key, value):
        if self.contains(key) == False:
            node = Node(key, value)
            self.dll.appendleft(node)
            self.node_map[key] = node
        else:
            self.node_map[key].value = value
            self.moveleft(key)

    def append(self, key, value):
        if self.contains(key) == False:
            node = Node(key, value)
            self.dll.append(node)
            self.node_map[key] = node
        else:
            self.node_map[key].value = value
            self.moveright(key)
    
    def moveleft(self, key):
        node = self.node_map[key]
        self.dll.remove(node)
        self.dll.appendleft(node)

    def moveright(self, key):
        node = self.node_map[key]
        self.dll.remove(node)
        self.dll.append(node)

    def remove(self, key):
        node = self.node_map[key]
        self.dll.remove(node)
        self.node_map.pop(key)
    
    def popleft(self):
        key = self.dll.get_head().key
        self.remove(key)
        return key

    def pop(self):
        key = self.dll.get_tail().key
        self.remove(key)
        return key

class LFUCache:
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cap = capacity
        self.min_f = -1
        # key:(value, frequency)
        self.cache = {}
        # frequency:LinkedHashSet
        self.freq_map = defaultdict(LinkedHashSet)

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.cache:
            # Update new frequency
            # Update freq_map
            # Move key to front in its linkedHashSet
            # Update new minimum frequency
            v, f = self.cache[key][0], self.cache[key][1]
            self.cache[key][1] += 1
            
            f_count_zero = False
            self.freq_map[f].remove(key)
            if self.freq_map[f].size() == 0:
                f_count_zero = True
                self.freq_map.pop(f)
            self.freq_map[f+1].appendleft(key, v)
                
            if f == self.min_f and f_count_zero == True:
                self.min_f += 1  
            return v
        return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if self.cap == 0:
            return
        if key in self.cache:
            self.cache[key][0] = value
            self.get(key)
        else:
            curr_size = len(self.cache)
            if curr_size == self.cap:
                min_list = self.freq_map[self.min_f]
                x = min_list.pop()
                self.cache.pop(x)
            
            self.cache[key] = [value, 1]
            self.freq_map[1].appendleft(key, value)
            self.min_f = 1
```

#### 463. Island Perimeter
```
class Solution:
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        求二维grid里的周长
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    continue
                # 上下左右4个方向遍历
                # 或者是墙边或者临近的那个点是0
                # res就可以加1
                # 两个点只要相邻，相邻边就不是周长一部分
                # 注意这里是4个if，相当于四个方向分别去判断是否需要加1
                if j == 0 or grid[i][j - 1] == 0:
                    res += 1
                if i == 0 or grid[i - 1][j] == 0:
                    res += 1
                if j == n - 1 or grid[i][j + 1] == 0:
                    res += 1
                if i == m - 1 or grid[i + 1][j] == 0:
                    res += 1
        
        return res
```

#### 468. Validate IP Address
```
class Solution:
    def validIPAddress(self, IP):
        """
        :type IP: str
        :rtype: str
        """
        if '.' in IP:
            parts = IP.split('.')
            cnt = len(parts)
            if cnt != 4:
                return 'Neither'
            for part in parts:
                if not part:
                    return 'Neither'
                if len(part) > 3:
                    return 'Neither'
                if len(part) > 1 and part[0] == '0':
                    return 'Neither'
                for ch in part:
                    if not '0' <= ch <= '9':
                        return 'Neither'
                if not 0 <= int(part) <= 255:
                    return 'Neither'
            return 'IPv4'
        else:
            parts = IP.split(':')
            cnt = len(parts)
            if cnt != 8:
                return 'Neither'
            for i in range(len(parts)):
                if i > 0 and not parts[i] and (not parts[i - 1] or parts[i - 1] == '0'):
                    return 'Neither'
                if len(parts[i]) > 4:
                    return 'Neither'
                for ch in parts[i]:
                    if not '0' <= ch <= '9' \
                        and not 'a' <= ch <= 'f' \
                        and not 'A' <= ch <= 'F':
                        return 'Neither'
            return 'IPv6'
```

#### 471. Encode String with Shortest Length
```
class Solution:
    def encode(self, s):
        """
        :type s: str
        :rtype: str
        aaaaa -> 5[a]
        """
        n = len(s)
        # dp[i][j]表示s在[i, j]区间内的最短encode表示
        dp = [[''] * n for _ in range(n)]
        for step in range(1, n + 1):
            for i in range(n - step + 1):
                j = i + step - 1
                dp[i][j] = s[i:i + step]
                for k in range(i, j):
                    left = dp[i][k]
                    right = dp[k + 1][j]
                    if len(left) + len(right) < len(dp[i][j]):
                        dp[i][j] = left + right
                temp = s[i:j + 1]
                replace = ''
                # find函数： S.find(sub[, start[, end]]) -> int
                # 应该从1开始找，避免找到0
                pos = (temp * 2).find(temp, 1)
                if pos >= len(temp):
                    replace = temp
                else:
                    # 核心之一： str(len(temp) // pos) 就是temp中哟多少个重复的pattern！
                    replace = str(len(temp) // pos) + '[' + dp[i][i + pos - 1] + ']'
                if len(replace) < len(dp[i][j]):
                    dp[i][j] = replace
        
        return dp[0][n - 1]
```

#### 477. Total Hamming Distance
```
class Solution:
    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        汉明距离就是两个整数有多少个bits是不一样的
        """
        # 其实就是0的个数乘以1的个数
        # 只要统计出32位上每一位的1的个数即可
        n = len(nums)
        res = 0
        for i in range(32):
            num_ones = 0
            for num in nums:
                if num & (1 << i) != 0:
                    num_ones += 1
            res += num_ones * (n - num_ones)
        return res
```

#### 482. License Key Formatting
```
class Solution:
    def licenseKeyFormatting(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        new_S = ''
        for ch in S:
            if ch == '-':
                continue
            new_S += ch.upper()
        
        res = []
        i = len(new_S) - 1
        while i >= K:
            res.append(new_S[i - K + 1:i + 1])
            i -= K
        
        if i >= 0:
            res.append(new_S[:i + 1])
        
        return '-'.join(res[::-1])
```

#### 489. Robot Room Cleaner, Hard, Facebook
```
# """
# This is the robot's control interface.
# You should not implement it, or speculate about its implementation
# """
#class Robot:
#    def move(self):
#        """
#        Returns true if the cell in front is open and robot moves into the cell.
#        Returns false if the cell in front is blocked and robot stays in the current cell.
#        :rtype bool
#        """
#
#    def turnLeft(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def turnRight(self):
#        """
#        Robot will stay in the same cell after calling turnLeft/turnRight.
#        Each turn will be 90 degrees.
#        :rtype void
#        """
#
#    def clean(self):
#        """
#        Clean the current cell.
#        :rtype void
#        """

class Solution:
    
    def cleanRoom(self, robot):
        """
        :type robot: Robot
        :rtype: None
        """
        # 这道题因为没有给出房间地图
        # 所以用相对于远点(0, 0)的位置作为visited
        visited = set()
        # 初始：robot以(0, 1)的方向到达(0, 0)点
        self._dfs(robot, 0, 0, 0, 1, visited)
    
    def _dfs(self, robot, x, y, dx, dy, visited):
        robot.clean()
        visited.add((x, y))

        for _ in range(4):
            # 向四个方向递归下去
            newx, newy = x + dx, y + dy
            if (newx, newy) not in visited \
                and robot.move():
                # 注意：在if中已经调用了robot.move()
                # 这里可以理解成如果robot能过去
                # 就已经到了
                # 在dfs里面已经保证了能回到原点
                # 所以直接转向即可
                self._dfs(robot, newx, newy, dx, dy, visited)
            # 这里不管是turnLeft还是turnRight都是一样的
            robot.turnLeft()
            # 虽然方向是逆时针（turnLeft）的
            # 很好的技巧：转向，变号
            dx, dy = -dy, dx
        
        robot.turnLeft()
        robot.turnLeft()
        robot.move()
        robot.turnLeft()
        robot.turnLeft()
```

#### 490. The Maze
```
from collections import deque

class Solution:
    
    _DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def hasPath(self, maze, start, destination):
        """
        :type maze: List[List[int]]
        :type start: List[int]
        :type destination: List[int]
        :rtype: bool
        """
        if not maze or not maze[0]:
            return False
        
        if maze[start[0]][start[1]] == 1:
            return False
        
        m, n = len(maze), len(maze[0])
        visited = [[False] * n for _ in range(m)]
        # queue里放的是小球能够停留的点
        queue = deque()
        queue.append(start)
        visited[start[0]][start[1]] = True
        
        while queue:
            cx, cy = queue.popleft()
            if cx == destination[0] and cy == destination[1]:
                return True
            for dx, dy in self._DIRECTIONS:
                newx, newy = cx + dx, cy + dy
                while 0 <= newx < m and 0 <= newy < n and maze[newx][newy] == 0:
                    newx += dx
                    newy += dy
                newx -= dx
                newy -= dy

                if not visited[newx][newy]:
                    queue.append([newx, newy])
                    visited[newx][newy] = True
        
        return False
```

#### 491. Increasing Subsequences
```
class Solution:
    def findSubsequences(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = set()
        self._dfs(nums, 0, [], res)
        return list(res)
    
    def _dfs(self, nums, start, curr, res):
        if len(curr) >= 2:
            res.add(tuple(curr))
            # 此时不能return
            # 这道题是继续去往下找

        for i in range(start, len(nums)):
            if curr and curr[-1] > nums[i]:
                continue
            curr.append(nums[i])
            self._dfs(nums, i + 1, curr, res)
            curr.pop()
```

#### 493. Reverse Pairs, Hard, Facebook
```
class Solution:
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 这道题也可以用树状数组来解
        # 假设nums里的数字是有上限的
        # 则用一个数组（或者哈希表）c，数字的下标就是nums里的值
        # 而c[nums[i]]的含义就是有多少个小于nums[i]的个数
        # 由于不需要考虑num[i]的值具体是多少
        # 可以将num[i]归一化，这样就肯定能用一个固定大小的数组来解了
        return self._merge_sort(nums, 0, len(nums) - 1)
    
    def _merge_sort(self, nums, left, right):
        if left >= right:
            return 0
        
        mid = left + (right - left) // 2
        res = self._merge_sort(nums, left, mid) + self._merge_sort(nums, mid + 1, right)
        
        j = mid + 1
        for i in range(left, mid + 1):
            while j <= right and nums[i] > 2 * nums[j]:
                j += 1
            res += j - (mid + 1)
        
        nums[left:right + 1] = sorted(nums[left:right + 1])
        
        return res
```

#### 494. Target Sum
```
# DP
class Solution:
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        dp = [{} for _ in range(len(nums))]
        return self._dfs(nums, S, 0, dp)
    
    # nums从start开始凑成total有多少种答案
    def _dfs(self, nums, total, start, dp):
        if start == len(nums):
            return 1 if total == 0 else 0
        
        if total in dp[start]:
            return dp[start][total]

        cnt1 = self._dfs(nums, total - nums[start], start + 1, dp)
        cnt2 = self._dfs(nums, total + nums[start], start + 1, dp)
        dp[start][total] = cnt1 + cnt2
        return dp[start][total]
        

# DFS TLE
# class Solution:
#     def findTargetSumWays(self, nums, S):
#         """
#         :type nums: List[int]
#         :type S: int
#         :rtype: int
#         """
#         self.res = 0
#         self._dfs(nums, S, 0)
#         return self.res
    
#     def _dfs(self, nums, S, start):
#         if start >= len(nums):
#             if S == 0:
#                 self.res += 1
#             return
        
#         self._dfs(nums, S - nums[start], start + 1)
#         self._dfs(nums, S + nums[start], start + 1)
```

#### 498. Diagonal Traverse
```
class Solution:
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        row = col = 0
        d = 0
        # 第一次是假设往右上方走的
        # 所以dir[0] = (-1, 1)
        dirs = [(-1, 1), (1, -1)]
        
        res = []
        for _ in range(m * n):
            res.append(matrix[row][col])
            row += dirs[d][0]
            col += dirs[d][1]
            
            # 注意重点：
            # 这道题一定要先处理前两个if case
            # 再处理后面的小于0的情况
            
            # 要不然：比如当中反对角线穿出去的时候，此时row < 0 并且 col >= n
            # 就会被两个if都catch住处理了
            # 小trick：d初始为0，不停的d = 1 - d就能让d在0 1之间不停的变化
            # 很棒的技巧！！！
            if row >= m:
                row = m - 1
                col += 2
                d = 1 - d
            if col >= n:
                col = n - 1
                row += 2
                d = 1 - d
            if row < 0:
                row = 0
                d = 1 - d
            if col < 0:
                col = 0
                d = 1 - d
        
        return res
```

#### 505. The Maze II
```
from collections import deque

class Solution:
    def shortestDistance(self, maze, start, destination):
        """
        :type maze: List[List[int]]
        :type start: List[int]
        :type destination: List[int]
        :rtype: int
        """
        m, n = len(maze), len(maze[0])
        dists = [[2 ** 31 - 1] * n for _ in range(m)]
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        queue = deque()
        queue.append([start[0], start[1]])
        dists[start[0]][start[1]] = 0
        
        while queue:
            curr = queue.popleft()
            for di, dj in dirs:
                # 四个方向
                # 每个方向都是一种选择
                # 所以要把i j重置为curr！！！！
                # 如果把下面两行放到for的外层去
                # 在每次循环里面都会沿用上次的i j以及dist
                i, j = curr
                dist = dists[curr[0]][curr[1]]
                while 0 <= i < m and 0 <= j < n and maze[i][j] == 0: 
                    i += di
                    j += dj
                    dist += 1
                # 根据while里的条件
                # 当滚出while时候是滚过头了一步
                # 要减掉
                i -= di
                j -= dj
                dist -= 1
                # 实际上每个点有可能多次入队的
                # 点i j可以通过不同距离的路径到达
                # 如果当前到达i j是更短的距离
                # 就可以重新更新一下
                # 感觉可以优化？？
                if dists[i][j] > dist:
                    dists[i][j] = dist
                    if i != destination[0] or j != destination[1]:
                        queue.append([i, j])

        res = dists[destination[0]][destination[1]]
        return res if res != 2 ** 31 -1 else -1
```

#### 518. Coin Change 2
```
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        # dp[i][j]定义：
        # 前i个硬币（可以重复取）凑成j币值的种类数
        dp = [[0] * (amount + 1) for _ in range(len(coins) + 1)]
        dp[0][0] = 1
        for i in range(1, len(coins) + 1):
            dp[i][0] = 1
            for j in range(1, amount + 1):
                dp[i][j] = dp[i - 1][j]
                if j >= coins[i - 1]:
                    # 凑成当前j面值
                    # 而且当前硬币面值是coins[i - 1]
                    # 则我们一定还另外有dp[i][j - coins[i - 1]]种做法
                    # 直观理解是只要凑成了j - coins[i - 1]面值就够了，剩下的没有别的选择(加上当前的硬币coins[i - 1]就可以了)
                    dp[i][j] += dp[i][j - coins[i - 1]]
        
        return dp[-1][-1]
```

#### 523. Continuous Subarray Sum
```
class Solution:
    def checkSubarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        # 这道题是说找到连续子数组的和是k的倍数
        # 如果能找到，return True
        # 否则return False
        """
        n = len(nums)

        for i in range(n):
            total = nums[i]
            for j in range(i + 1, n):
                total += nums[j]
                if total == k:
                    return True
                if k != 0 and total % k == 0:
                    return True
        
        return False
```

#### 527. Word Abbreviation
```
class Solution:
    def wordsAbbreviation(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        n = len(words)
        res = list(map(self._abbrev, words))
        prefix = [0] * n
        
        for i in range(n):
            while True:
                dup = set()
                for j in range(i + 1, n):
                    if res[i] == res[j]:
                        dup.add(j)
                
                if not dup:
                    break
                
                # 注意一定要将i也添加，此时不光是j不能沿用之前的prefix了
                # i也不能用
                dup.add(i)
                for k in dup:
                    prefix[k] += 1
                    res[k] = self._abbrev(words[k], prefix[k])
        
        return res
    
    # abbrev函数定义：
    # 保留前i个字符，从第i+1到最后执行缩写操作
    def _abbrev(self, word, i=0):
        if len(word) - i <= 3:
            return word
        # 下面的索引实际上应该是len(word) - (i + 1) - 1
        # 因为i是从0开始的，所以这里应该加1
        # 整合一下就是如下的写法
        return word[:i + 1] + str(len(word) - i - 2) + word[-1]
```

#### 528. Random Pick with Weight
```
from random import randint

class Solution:

    def __init__(self, w):
        """
        :type w: List[int]
        带weighting的随机数
        """
        # 这里不是严格的presum数组
        self._pre_sum = [0] * len(w)
        self._pre_sum[0] = w[0]
        for i in range(1, len(w)):
            self._pre_sum[i] = self._pre_sum[i - 1] + w[i]

    def pickIndex(self):
        """
        :rtype: int
        """
        # 从0到总和total sum之间取一个数字
        # 相当于看这个数字落在哪个范围内
        # 则pre_sum数组中**第一个**大于这个随机数字的就是解
        rand_num = randint(0, self._pre_sum[-1] - 1)
        start, end = 0, len(self._pre_sum) - 1
        # 二分去找presum数组里第一个严格大于随机数rand_num的坐标
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._pre_sum[mid] > rand_num:
                # 右边一定不存在解，直接丢掉
                end = mid
            else:
                # 左边一定不存在解，直接丢掉
                start = mid + 1
        
        if self._pre_sum[start] > rand_num:
            return start
        
        return end

# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()
```

#### 543. Diameter of Binary Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        从左边最深的点经过root到右边最深的点
        叫做直径
        """
        self.res = 0
        self._dfs(root)
        return self.res
    
    # 定义为寻找以node为根的树的最大深度
    def _dfs(self, node):
        if not node:
            return 0
        left_dia = self._dfs(node.left)
        right_dia = self._dfs(node.right)
        self.res = max(self.res, left_dia + right_dia)
        return max(left_dia, right_dia) + 1
```

#### 547. Friend Circles
```
class Solution:
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        
        if not M or not M[0]:
            return 0
        
        res = 0
        visited = [False] * len(M)
        
        for i in range(len(M)):
            if not visited[i]:
                res += 1
                self._dfs(M, i, visited)
        
        return res
    
    # DFS定义：从i开始找未访问过的
    # 如果有连接，就递归调用自己
    # 并且（很重要）要在调用递归之前
    # 将当前设置为visited
    def _dfs(self, M, start_inx, visited):
        for i in range(len(M)):
            if i != start_inx and \
                not visited[i] and \
                M[start_inx][i] == 1:
                visited[i] = True
                self._dfs(M, i, visited)
```

#### 548. Split Array with Equal Sum
```
# python2能通过，3通不过-_-
class Solution:
    def splitArray(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        # 这道题是判断能否用i j k三个不相邻的索引将数组分成4段sum和相等的子数组
        """
        # . i . j . k .  -> 最少需要7个数字
        if not nums or len(nums) < 7:
            return False
        
        n = len(nums)
        pre_sum = [0]
        for num in nums:
            pre_sum.append(num + pre_sum[-1])
            
        for j in range(3, n - 3):
            hash_set = set()
            for i in range(1, j - 1):
                if pre_sum[i] == pre_sum[j] - pre_sum[i + 1]:
                    hash_set.add(pre_sum[i])
            for k in range(j + 1, n - 1):
                if pre_sum[k] - pre_sum[j + 1] == pre_sum[n] - pre_sum[k + 1]:
                    if pre_sum[k] - pre_sum[j + 1] in hash_set:
                        return True

        return False
```

#### 552. Student Attendance Record II
```
class Solution:
    def checkRecord(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 1
        if n == 1:
            return 3
        
        # 我们先考虑不含'A'的情况
        # num[i + 1]表示前n中前i个字符在不包含'A'的时候能凑成多少种答案
        # 则此时只有如下3中结尾的方案：
        # 1. 以'P'结尾，这个结尾长度为1，所以我们要回溯到nums[i]
        # 2. 以'PL'结尾，这个结尾长度为2，所以我们要回溯到nums[i - 1]
        # 3. 以'PLL'结尾，这个结尾长度为3，所以我们要回溯到nums[i - 2]
        nums = [1, 1, 2]
        for i in range(2, n):
            nums.append((nums[i] + nums[i - 1] + nums[i - 2]) % 1000000007)
        
        # 先求一遍此时的答案
        res = (nums[n] + nums[n - 1] + nums[n - 2]) % 1000000007
        
        # 然后insert 'A'到答案中，假设insert的位置就是i
        # 此时左边表示n中i个字符有多少种答案
        # 右边表示n中n - i - 1个字符有多少种答案
        # 注意总长度还是i + (n - i - 1) + 1 = n的
        for i in range(n):
            res += nums[i + 1] * nums[n - i] % 1000000007
            res %= 1000000007
        
        return res
```

#### 554. Brick Wall
```
from collections import defaultdict

class Solution:
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        if not wall:
            return 0
        mapping = defaultdict(int)
        res = 0
        for each_row in wall:
            total = 0
            # 注意坑：each_row不能遍历到最后一个！！！
            # 因为到所有的row总长度一样
            # 所以在每行最后的断点必定是最多的！！！
            for each_brick in each_row[:-1]:
                total += each_brick
                mapping[total] += 1
                res = max(res, mapping[total])
        return len(wall) - res
```

#### 556. Next Greater Element III
```
class Solution:
    def nextGreaterElement(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 这道题需要4个步骤：
        # 1. 从右往左，找到第一个小于右边数字的位置i
        # 2. 从右往左，找到第一个大于i的位置j
        # 3. 将i和j交换值
        # 4. 将i + 1到最后逆序
        nums = list(str(n))
        n = len(nums)
        i = n - 2
        
        # 1. 从右往左，找到第一个小于右边数字的位置i
        while i >= 0:
            if nums[i] < nums[i + 1]:
                break
            i -= 1
        # 找到头也没有出现i，直接return -1
        if i < 0:
            return -1

        # 2. 从右往左，找到第一个大于i的位置j
        j = n - 1
        while j >= 0:
            if nums[j] > nums[i]:
                break
            j -= 1

        # 3. 将i和j交换值
        nums[i], nums[j] = nums[j], nums[i]
        # 4. 将i + 1到最后逆序
        res_list = nums[:i + 1] + nums[i + 1:][::-1]

        res = int(''.join(i for i in res_list))
        # 这道题有个坑，因为需要32位int，所以当越界就return -1
        return res if res <= 2 ** 31 - 1 else -1
```

#### 560. Subarray Sum Equals K
```
from collections import defaultdict


class Solution:
    def subarraySum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        res = 0
        total = 0
        # 某个sum值（这个值是从最右边0开始算的）
        # 基本思路就是如果这个值出现过两次
        # 则中间的坐标之间的和就是0
        # 推广一下
        # 如果total出现过，而且total-target也出现过
        # 则中间的差就是target！！！
        mapping = defaultdict(int)
        
        for num in nums:
            mapping[total] += 1
            total += num
            res += mapping[total - target]
    
        return res
```

#### 564. Find the Closest Palindrome
```
class Solution:
    def nearestPalindromic(self, n):
        """
        :type n: str
        :rtype: str
        """
        # 核心：备选答案里有5中情况
        # 将左半部分的翻转放入备选中
        n_len = len(n)
        n_num = int(n)
        
        candidates = set()
        # 比如n是一个3位数字string
        # 先将1001和99加入备选
        candidates.add(10 ** n_len + 1)
        candidates.add(10 ** (n_len - 1) - 1)
        
        # 比如数字是123，prefix为12
        # 数字是1234，prefix为12
        pre_fix = int(n[:(n_len + 1) // 2])
        
        for i in (-1, 0, 1):
            pre = str(pre_fix + i)
            if n_len % 2 == 0:
                new_str = pre + pre[::-1]
            else:
                new_str = pre + pre[-2::-1]
            candidates.add(int(new_str))

        if n_num in candidates:
            candidates.remove(n_num)

        res = min_diff = 2 ** 31 - 1
        for each in candidates:
            diff = abs(each - n_num)
            if diff < min_diff:
                res = each
                min_diff = diff
            # 这里是找较小的数字
            # 比如88，99和77都是一样的diff
            # 但是答案要返回77因为77较小
            elif diff == min_diff:
                res = min(res, each)
        
        return str(res)
```

#### 567. Permutation in String
```
class Solution:
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        n1, n2 = len(s1), len(s2)
        mapping = [0] * 256
        for ch in s1:
            mapping[ord(ch)] += 1
        
        l = r = 0
        required = n1
        while r < n2:
            # 注意这3行顺序
            # 说明当前处理的s2[r]是在s1中的
            # 因为我们本次循环会加入这个字符
            # 所以required就可以少一个了
            # 这个mapping是用来计算required的
            # 所以第三行要无论如何减去1
            if mapping[ord(s2[r])] > 0:
                required -= 1
            mapping[ord(s2[r])] -= 1

            while required == 0:
                if r - l + 1 == n1:
                    return True
                # 注意这3行顺序
                mapping[ord(s2[l])] += 1
                # 如果这里是大于0的
                # 说明之前一定是大于-1的
                # 就说明不是-1
                # 因为l坐标肯定在r坐标之前遍历过的（修改了mapping）
                # 所以如果之前的l对应的字符不是属于s1的
                # 之前遍历的时候肯定已经是-1或者小于-1
                # 这里大于-1(等价于mapping[ord(s2[l])] > 0)就说明l肯定是在s1里的！！！！
                if mapping[ord(s2[l])] > 0:
                    required += 1
                l += 1
            r += 1
        
        return False
```

#### 568. Maximum Vacation Days, Hard, Facebook
```
# TLE 但是思路是对的，C++版本AC
class Solution:
    def maxVacationDays(self, flights, days):
        """
        :type flights: List[List[int]]
        :type days: List[List[int]]
        :rtype: int
        N are cities
        K are weeks
        flights: N*N
        days: N*K
        """
        n = len(flights)
        k = len(days[0])
        # dp[n][k]的定义是在第k周当前在n城市，已经获得的总的休假天数
        dp = [[0] * k for _ in range(n)]
        
        # 这道题的时间是从后往前退的
        # 当前dp[i][j]的
        for j in range(k - 1, -1, -1):
            for i in range(n):
                dp[i][j] = days[i][j]
                for p in range(n):
                    # 第一个条件表示能否从i城市到达p城市
                    # 后面仅是边界条件，表示不是最后一周
                    if (i == p or flights[i][p] == 1) and j < k - 1:
                        # 这道题后面weeks的假期是被消耗掉的
                        # 会原来越少
                        # 所以当前的最多假期应该是加上未来的最多假期
                        dp[i][j] = max(dp[i][j], dp[p][j + 1] + days[i][j])
        
        # 第一周不一定从那个城市开始
        # 我们是从周一就可以上飞机的
        # 可以从任何可以去的城市开始
        return max(dp[i][0] for i in range(n) if i == 0 or flights[0][i] == 1)
```

#### 591. Tag Validator
```
import re

class Solution:
    def isValid(self, code):
        """
        :type code: str
        :rtype: bool
        """
        tag_stack = []
        while code:
            if code.startswith('<![CDATA['):
                if not tag_stack:
                    return False
                next_inx = code.find(']]>')
                if next_inx == -1:
                    return False
                code = code[next_inx + 3:]
            elif code.startswith('</'):
                next_inx = code.find('>')
                if next_inx == -1:
                    return False
                tag_name = code[2:next_inx]
                if not tag_stack or tag_stack.pop() != tag_name:
                    return False
                code = code[next_inx + 1:]
                # 这里实际上是为了一个极端case
                # "<A></A><B></B>"要求返回False
                # 感觉应该是返回True的
                if not tag_stack:
                    return not code
            elif code.startswith('<'):
                next_inx = code.find('>')
                if next_inx == -1:
                    return False
                tag_name = code[1:next_inx]
                # 正则里[]里表示单个匹配
                # {}表示前面的长度是1到9
                if not re.match('^[A-Z]{1,9}$', tag_name):
                    return False
                tag_stack.append(tag_name)
                code = code[next_inx + 1:]
            elif not tag_stack:
                return False
            else:
                code = code[1:]
        
        return not tag_stack
```

#### 605. Can Place Flowers
```
class Solution:
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        
        res = 0
        length = len(flowerbed)
        for i in range(length):
            if flowerbed[i] == 1:
                continue
            if i > 0 and flowerbed[i - 1] == 1:
                continue
            if i < length - 1 and flowerbed[i + 1] == 1:
                continue
            # 核心逻辑
            res += 1
            flowerbed[i] = 1
        return res >= n
```

#### 611. Valid Triangle Number
```
class Solution:
    def triangleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        nums.sort()
        n = len(nums)
        res = 0

        for i in range(n - 1, 1, -1):
            left, right = 0, i - 1
            while left < right:
                if nums[left] + nums[right] > nums[i]:
                    res += right - left
                    right -= 1
                else:
                    left += 1
        
        return res
```

#### 621. Task Scheduler
```
class Solution:
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        每个相同的task需要相隔n个间隔
        """
        cnt = [0] * 26
        for task in tasks:
            cnt[ord(task) - ord('A')] += 1
        
        cnt.sort()
        mx = cnt[25]
        length = len(tasks)
        # first_less_mx_inx是sort过后的cnt数组中从左往右最后一个值小于mx的
        first_less_mx_inx = cnt.index(mx) - 1
        
        # n + 1就是每个子集的长度
        # mx是出现的最多次数
        # 因为第一项是不考虑最后一大坨出现次数相等并且出现次数都是最多的task
        # 所以mx - 1
        # 25 - first_less_mx_inx是指在最后要排多少个出现多大次数的不同任务
        return max(length, (n + 1) * (mx - 1) + 25 - first_less_mx_inx)
```

#### 622. Design Circular Queue
```
class MyCircularQueue:

    def __init__(self, k):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        :type k: int
        """
        self._data = [None] * k
        self._size = k
        # 头部出队，尾部入队
        self._head = k - 1
        self._tail = 0
        self._cnt = 0
        
    def enQueue(self, value):
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        :type value: int
        :rtype: bool
        """
        # 实际上并不返回什么
        # 只是把能取值的head index变换一下
        if self.isFull():
            return False
        self._data[self._tail] = value
        self._tail = (self._tail + 1) % self._size
        self._cnt += 1
        return True
        
    def deQueue(self):
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        :rtype: bool
        """
        if self.isEmpty():
            return False
        self._head = (self._head + 1) % self._size
        self._cnt -= 1
        return True

    def Front(self):
        """
        Get the front item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        # 这题重点
        # head最大取到size - 1
        # 这种情况下就取到tail
        return self._data[(self._head + 1) % self._size]

    def Rear(self):
        """
        Get the last item from the queue.
        :rtype: int
        """
        if self.isEmpty():
            return -1
        # 这题重点
        # tail最小取到0
        # 这种情况下就取到head
        return self._data[(self._tail - 1 + self._size) % self._size]

    def isEmpty(self):
        """
        Checks whether the circular queue is empty or not.
        :rtype: bool
        """
        return self._cnt == 0

    def isFull(self):
        """
        Checks whether the circular queue is full or not.
        :rtype: bool
        """
        return self._cnt == self._size


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
```

#### 632. Smallest Range, Hard, Facebook
```
from collections import defaultdict

class Solution:
    def smallestRange(self, nums):
        """
        :type nums: List[List[int]]
        :rtype: List[int]
        """
        res = []
        sorted_num = []
        
        for i in range(len(nums)):
            for num in nums[i]:
                sorted_num.append((num, i))
        sorted_num.sort()
        
        left = 0
        inx_mapping = defaultdict(int)
        required = len(nums)
        found = 0
        diff = 2 ** 31 - 1
        for right in range(len(sorted_num)):
            _, inx = sorted_num[right]
            if inx not in inx_mapping:
                found += 1
            inx_mapping[inx] += 1
            while found == required:
                if diff > sorted_num[right][0] - sorted_num[left][0]:
                    diff = sorted_num[right][0] - sorted_num[left][0]
                    res = [sorted_num[left][0], sorted_num[right][0]]
                inx_mapping[sorted_num[left][1]] -= 1
                if inx_mapping[sorted_num[left][1]] == 0:
                    # 重要：如果此时为0，则说明要把left inx从mapping中移出去了！！！
                    found -= 1
                    del inx_mapping[sorted_num[left][1]]
                left += 1
        
        return res
```

#### 636. Exclusive Time of Functions
```
class Solution:
    def exclusiveTime(self, n, logs):
        """
        :type n: int
        :type logs: List[str]
        :rtype: List[int]
        """
        
        # 栈中保存元素格式为函数ID，时间戳 [fid, timestamp]
        # 当日志为'start'时：
        # 若栈非空，记栈顶元素为top_fid, top_timestamp；timestamp - top_timestamp；
        # 将[fid, timestamp]压栈
        # 否则：
        # 将栈顶元素的时间累加timestamp - top_timestamp + 1
        # 弹出栈顶元素
        # 若栈非空，将栈顶元素top_timestamp更新为timestamp + 1
        # 重要：栈顶元素就是上一次执行的函数
        res = [0] * n
        stack = []
        for log in logs:
            fid, flag, timestamp = log.split(':')
            fid = int(fid)
            timestamp = int(timestamp)
            if flag == 'start':
                if stack:
                    top_fid, top_timestamp = stack[-1]
                    res[top_fid] += timestamp - top_timestamp
                stack.append([fid, timestamp])
            else:
                # 此时上一次的函数要被终止了
                # 所以要去更新上一次函数的执行时间
                res[stack[-1][0]] += timestamp - stack[-1][1] + 1
                stack.pop()
                if stack:
                    stack[-1][1] = timestamp + 1
        
        return res
```

#### 642. Design Search Autocomplete System
```
class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.sentence = ''
        self.hotness = 0
        
class AutocompleteSystem():
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.prefix = ''
        for sentence, hotness in zip(sentences, times):
            self.add(sentence, hotness)

    def add(self, sentence, hotness):
        curr = self.root
        for ch in sentence:
            if ch not in curr.children:
                curr.children[ch] = TrieNode()
            curr = curr.children[ch]
        curr.is_end = True
        curr.sentence = sentence
        # 为何这里要用减法？
        # 这道题是要求hostness从大到小排序
        # 而sentence是按照ord从小到大排序！！！
        # 比如空格的ord是32， 字母r的ord是114
        # 两者hotness一样的情况下，
        # 虽然ord的值是r大，但是这道题认为空格是排在字母r前面的
        # 所以要优先返回这个空格！！！
        # 所以单纯的加hotness然后妄想sort(reverse=True)这道题是行不通的
        curr.hotness -= hotness

    def input(self, c):
        if c == '#':
            self.add(self.prefix, 1)
            self.prefix = ''
            return []
        else:
            self.prefix += c
            curr = self.root
            for ch in self.prefix:
                if ch not in curr.children:
                    return []
                curr = curr.children[ch]
            res = self._dfs(curr)
            res.sort()
            return [i[1] for i in res[:3]]

    def _dfs(self, root):
        res = []
        if root:
            if root.is_end:
                res.append((root.hotness, root.sentence))
            for _, node in root.children.items():
                res.extend(self._dfs(node))
        return res
```

#### 647. Palindromic Substrings
```
class Solution:
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        ## http://www.cnblogs.com/grandyang/p/7404777.html
        ## 在s[i]和s[j]相等这个条件下，去判断[i...j]是否是回文的时候，i和j的位置关系很重要，
        ## 如果i和j相等了，那么dp[i][j]肯定是true；
        ## 如果i和j是相邻的，那么dp[i][j]也是true；
        ## 如果i和j中间只有一个字符，那么dp[i][j]还是true；
        ## 如果中间有多余一个字符存在，那么我们需要看dp[i+1][j-1]是否为true，若为true，那么dp[i][j]就是true。
        ## 赋值dp[i][j]后，如果其为true，结果res自增1
        if not s:
            return 0
        
        n = len(s)
        # dp[i][j]定义为s[i]到s[j]是否是回文串
        dp = [[False] * n for _ in range(n)]
        res = 0
        
        # 从下往上，从左往右遍历（倒三角）
        # why? 因为根据递推公式先使用j - i <= 2
        # 再使用dp[i + 1][j - 1]时候一定已经有值了
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    res += 1
        
        return res
```

#### 654. Maximum Binary Tree
```
class Solution:
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return self._helper(nums)
    
    def _helper(self, nums):
        if not nums:
            return
        inx = nums.index(max(nums))
        node = TreeNode(nums[inx])
        node.left = self._helper(nums[:inx])
        node.right = self._helper(nums[inx + 1:])
        return node
```

#### 658. Find K Closest Elements
```
class Solution:
    def findClosestElements(self, arr, k, x):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """
        # 应该想到最终的返回的数组也是有序的
        res = arr[:]
        while len(res) > k:
            if x - res[0] <= res[-1] - x:
                # 说明最左边的数字和目标差距太大
                # 移除最后一位
                res.pop()
            else:
                # 反之移除第一位
                res.pop(0)
        
        return res
```

#### 663. Equal Tree Partition
```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def checkEqualTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        这道题是说树只砍一刀，分成两个子树
        这两个子树的和是相等的
        """
        cuts = set()
        self._root = root
        total = self._sum(root, cuts)
        return total / 2 in cuts
    
    # 这里的递归定义就是在求node为根的树的和
    # 但是在递归里额外将每次递归的和放入set中
    def _sum(self, node, cuts):
        if not node:
            return 0
        res = node.val + self._sum(node.left, cuts) \
            + self._sum(node.right, cuts)
        # 核心之一:
        # 这里实际上是为了避开[0, -1, 1]这种情况
        # 在[0, -1, 1]这种情况下不应该将res放入cuts中
        # 所以用下面的if给去除掉
        if node is not self._root:
            cuts.add(res)   
        return res
```

#### 668. Kth Smallest Number in Multiplication Table
```
class Solution:
    def findKthNumber(self, m, n, k):
        """
        :type m: int
        :type n: int
        :type k: int
        :rtype: int
        """
        if k > m * n or k <= 0:
            return -1
        
        l, r = 1, m * n
        while l < r:
            mid = l + (r - l) // 2
            cnt = 0
            # 基本思路：求出矩阵中小于mid的元素的个数
            for i in range(1, m + 1):
                if mid > n * i:
                    cnt += n
                else:
                    # 核心之一：由于每一行都是当前行号i乘以从1开始的自然序列：
                    # i * [1, 2, 3, 4, 5...]
                    # 所以用mid // i就相当于对mid归一化
                    # 就是这个序列中有多少个小于mid的数字
                    # 直接加到cnt中即可
                    cnt += mid // i
            # 核心之二：说明此时第k小的数字肯定在另一个半区
            # 可以直接抛弃所有当前左半区的内容
            if cnt < k:
                l = mid + 1
            # 核心之三：此时cnt >= k，由于这个等号的存在
            # 我们并不确定当前的mid是不是答案
            # 所以更新的r要包括当前的mid
            else:
                r = mid
        # 此时return l和return r都是一样的
        return l
```

#### 670. Maximum Swap
```
class Solution:
    def maximumSwap(self, num):
        """
        :type num: int
        :rtype: int
        """
        num_list = list(int(i) for i in str(num))
        n = len(num_list)

        last = [-1] * 10
        for i in range(n):
            last[num_list[i]] = i

        for i in range(n):
            for d in range(9, num_list[i], -1):
                # 当前遍历的数字是num_list[i]
                # d是从9到num_list[i]倒序遍历的
                # 则d一定是比num_list[i]大的
                # 如果d在原来num string中出现的位置是在i之后的
                # 说明我们找到了一个swap的位置
                # 就可以交换并且return了
                if last[d] > i:
                    num_list[i], num_list[last[d]] = num_list[last[d]], num_list[i]
                    return int(''.join(str(i) for i in num_list))
        
        # 如果没有在循环中return，说明没有找到swap的点（说明当前数字倒序）
        # 直接return即可
        return num
```

#### 671. Second Minimum Node In a Binary Tree
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # 层序遍历即可
        # global最优和local最优问题
        if not root:
            return -1
        
        queue = deque()
        queue.append(root)
        global_min = second_min = 2 ** 31 - 1
        while queue:
            curr = queue.popleft()
            global_min = min(global_min, curr.val)
            
            ## 在这种情况下我们应该去更新second_min
            if global_min < curr.val < second_min:
                second_min = curr.val
            
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        
        return second_min if second_min != 2 ** 31 - 1 else -1
```

#### 673. Number of Longest Increasing Subsequence
```
class Solution:
    def findNumberOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 动态规划（Dynamic Programming）
        # 数组dp_length[x]表示以x结尾的子序列中最长子序列的长度
        # 数组dp_counts[x]表示以x结尾的子序列中最长子序列(或者说长度为上面dp_length[x]的子序列)的个数
        if not nums:
            return 0

        n = len(nums)
        dp_length = [1] * n
        dp_counts = [1] * n
        
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    if dp_length[j] + 1 > dp_length[i]:
                        dp_length[i] = dp_length[j] + 1
                        dp_counts[i] = dp_counts[j]
                    # 就是说以j结尾的子序列可以替换以i为结尾的子序列中的一部分
                    # 而不增加以i结尾的子序列的长度
                    # 所以说就意味着此时以i结尾的子序列的个数(dp_counts[i])可以增加了dp_counts[j]
                    elif dp_length[j] + 1 == dp_length[i]:
                        dp_counts[i] += dp_counts[j]
        
        max_lis_length = max(dp_length)
        res = 0
        for p, z in zip(dp_length, dp_counts):
            if p == max_lis_length:
                res += z
        return res
```

#### 674. Longest Continuous Increasing Subsequence
```
class Solution:
    def findLengthOfLCIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        global_max = 1
        local_max = 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                local_max += 1
            else:
                global_max = max(global_max, local_max)
                local_max = 1

        # 这里是因为遍历到最后一个退出的时候
        # 可能没有经历过循环中的else
        return max(global_max, local_max)
```

#### 678. Valid Parenthesis String
```
class Solution:
    def checkValidString(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # 既然星号可以当左括号和右括号，那么我们就正反各遍历一次
        # 正向遍历的时候，我们把星号都当成左括号
        # 此时用经典的验证括号的方法，即遇左括号计数器加1，遇右括号则自减1
        # 如果中间某个时刻计数器小于0了，直接返回false。如果最终计数器等于0了
        # 我们直接返回true，因为此时我们把星号都当作了左括号，可以跟所有的右括号抵消。
        # 而此时就算计数器大于0了，我们暂时不能返回false，
        # 因为有可能多余的左括号是星号变的，星号也可以表示空，所以有可能多余这部分并没有多，
        # 可以被当成右边括号用
        # 我们还需要再次反向遍历一下，这是我们将所有的星号当作右括号，
        # 遇右括号计数器加1，遇左括号则自减1，如果中间某个时刻计数器小于0了，直接返回false。
        # 遍历结束后直接返回true，这是为啥呢？
        # 此时计数器有两种情况，要么为0，要么大于0。为0不用说，肯定是true，
        # 为啥大于0也是true呢？因为之前正向遍历的时候，我们的左括号多了，
        # 我们之前说过了，多余的左括号可能是星号变的，也可能是本身就多的左括号。
        # 本身就多的左括号这种情况会在反向遍历时被检测出来(第二次循环里的return False)，
        # 如果没有检测出来，说明多余的左括号一定是星号变的。
        # 而这些星号在反向遍历时又变做了右括号，最终导致了右括号有剩余，
        # 所以当这些星号都当作空的时候，左右括号都是对应的，即是合法的。
        # 可能会有疑问，右括号本身不会多么，其实不会的，
        # 如果多的话，会在正向遍历中被检测出来
        # ((((((((((((********中点右边的星号可以看作为空字符串*)))))))))))))))))
        # 结论：只要左右两次遍历left和right的值始终都是大于等于零就好了
        if not s:
            return True
        
        n = len(s)
        # left和right的含义分别是
        # 正向和反向遍历左右括号的count数目
        left = right = 0
        for i in range(n):
            if s[i] in '(*':
                left += 1
            else:
                left -= 1
            if left < 0:
                return False
        if left == 0:
            return True
        for i in range(n - 1, -1, -1):
            if s[i] in ')*':
                right += 1
            else:
                right -= 1
            if right < 0:
                return False
        return True
```

#### 679. 24 Game
```
class Solution:
    def judgePoint24(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        # 注意：这道题是可以包括小数的
        """
        return self._dfs(nums)
    
    def _dfs(self, nums):
        if not nums:
            return False
        if len(nums) == 1:
            return abs(nums[0] - 24) <= 1e-9
        
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i == j:
                    continue
                left_inx, right_inx = min(i, j), max(i, j)
                sub_nums = nums[:right_inx] + nums[right_inx + 1:]
                # 这里其实是在合并num[i]和nums[j]
                # 生成的sub_nums实际上就是不包含right_inx的新数组
                # 注意此时这个新数组实际上市包括原来left_inx的
                # 所以要在下面的循环里将i和j各种操作的结果（注意i和j实际上就是left_inx和right_inx）
                # 覆盖掉left_inx的值
                # 看到最后merge以后能不能凑成24
                for each_res in self._calculate(nums[i], nums[j]):
                    sub_nums[left_inx] = each_res
                    if self._dfs(sub_nums):
                        return True
        
        return False
     
    def _calculate(self, a, b):
        res = [a + b, a - b, a * b]
        if b > 1e-9:
            res.append(1.0 * a /b)
        return res
```

#### 680. Valid Palindrome II
```
class Solution:
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return self._is_valid(s, left + 1, right) or \
                    self._is_valid(s, left, right - 1)
            left += 1
            right -= 1
        
        # 此时说明s整体就是palindrome
        return True
    
    def _is_valid(self, s, left, right):
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
```

#### 681. Next Closest Time
```
class Solution:
    def nextClosestTime(self, time):
        """
        :type time: str
        :rtype: str
        """
        # 模拟时钟
        # 一分一分钟增加，直到所有的时间数字都在allowed这个set里
        curr = 60 * int(time[:2]) + int(time[3:])
        allowed = {int(x) for x in time if x != ':'}
        
        while True:
            # 当前的时间可能加过头了
            # 所以要mod一下每天的分钟总数
            # divmod返回两个值：
            # divmod(8, 3) == 2, 2
            curr = (curr + 1) % (24 * 60)
            if all(digit in allowed
                   # 重点：
                   # 注意python这里one-liner多重循环
                   # 顺序是反着的，要注意是从左到右判断是否变量存在
                   # 正常逻辑是：
                   # for digit in divmod(block, 10))
                   # for block in divmod(curr, 60):
                   # 但是由于这里的'block'第一次使用时候还没有定义
                   # 所以会报错
                   # 这里需要将这两句反过来
                   # 跟one-liner if-else不太一样
                   # res[0] if res else -1
                   # 会先去检查if的条件满不满足再去取res[0]
                   # 所以不会出现越界问题
                   for block in divmod(curr, 60)
                   for digit in divmod(block, 10)):
                return '{:02d}:{:02d}'.format(*divmod(curr, 60))
```

#### 685. Redundant Connection II
```
class Solution:
    def findRedundantDirectedConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        n = len(edges)
        parents = [0] * (n + 1)
        candidate1 = []
        candidate2 = []
        
        # step 1: 检查是否存在某个点有两个parents
        for i in range(n):
            start, end = edges[i]
            # 说明end这个点不在parents中
            if parents[end] == 0:
                parents[end] = start
            else:
                candidate1 = [parents[end], end]
                candidate2 = [start, end]
                # 仅仅是为了标记当前edge的end点
                edges[i][1] = 0
        
        # step 2: union find
        for i in range(1, n + 1):
            parents[i] = i
        
        for start, end in edges:
            # 这个点是标记过的，在union find中我们是去找是否存在环
            # 不考虑这个点
            if end == 0:
                continue
            start_parent = self._find_root(parents, start)
            # 此时说明找到了环，则答案必然跟环有关
            if start_parent == end:
                # 初始candidate1为空list[]
                # 如果此时仍旧为空list
                # 说明在step 1并没有更新过candidate1
                # 即说明step 1中没有找到具有两个parents的node
                # 此时直接返回造成环的当前edge [start, end]即可
                if not candidate1:
                    return [start, end]
                # 此时说明既有环，又有两个parents的node
                # 直接返回candidate2即可
                return candidate1
            # 执行union操作
            parents[end] = start_parent
        
        # 说明在第二步里没有找到环
        # 直接返回candidagte2
        return candidate2
    
    def _find_root(self, parents, point):
        if parents[point] != point:
            parents[point] = self._find_root(parents, parents[point])
        return parents[point]
```

#### 688. Knight Probability in Chessboard
```
class Solution:
    def knightProbability(self, N, K, r, c):
        """
        :type N: int
        :type K: int
        :type r: int
        :type c: int
        :rtype: float
        start at r c, K moves, still stay on the chessboard
        """
        if K == 0:
            return 1
        
        # dp[i][j]表示在棋盘(i, j)位置上走完当前步骤还留在棋盘上的走法总和
        # 初始化为1 (一步都不走就是一种答案)
        # 我们其实将步骤这个维度当成了时间维度在不停更新
        dp = [[1] * N for _ in range(N)]
            directions = [
                (1, 2),
                (-1, -2),
                (1, -2),
                (-1, 2),
                (2, 1),
                (-2, -1),
                (2, -1),
                (-2, 1),
            ]
        
        # 一共走K步
        # 每一步都重新计算dp矩阵
        for _ in range(K):
            temp = [[0] * N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    for di, dj in directions:
                        newi, newj = i + di, j + dj
                        if not 0 <= newi < N or not 0 <= newj < N:
                            continue
                        # 这里的dp实际上是上一个时间维度上在newi newj点上
                        # 能走的步数
                        # 将direction的8个方向上的步数都加到temp[i][j]上
                        temp[i][j] += dp[newi][newj]
            dp = temp
        
        # 每一步可以选择8个方向
        # 所以K步可以有8 ** K种选择
        return dp[r][c] / (8 ** K)
```

#### 689. Maximum Sum of 3 Non-Overlapping Subarrays, Hard, Facebook
```
class Solution:
    def maxSumOfThreeSubarrays(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        
        n = len(nums)
        pre_sum = [0]
        for num in nums:
            pre_sum.append(num + pre_sum[-1])
        
        # left[i]表示在区间[0, i]范围内长度为k且和最大的子数组的起始位置
        left = [0] * n
        total = pre_sum[k] - pre_sum[0]
        for i in range(k, n):
            if pre_sum[i + 1] - pre_sum[i + 1 - k] > total:
                left[i] = i + 1 - k
                total = pre_sum[i + 1] - pre_sum[i + 1- k]
            else:
                left[i] = left[i - 1]
        
        # right[i]表示在区间[i, n - 1]范围内长度为k且和最大的子数组的起始位置
        right = [n - k] * n
        total = pre_sum[n] - pre_sum[n - k]
        for i in range(n - 1 - k, -1, -1):
            if pre_sum[i + k] - pre_sum[i] > total:
                right[i] = i
                total = pre_sum[i + k] - pre_sum[i]
            else:
                right[i] = right[i + 1]
        
        mx = -2 ** 31
        res = None
        for i in range(k, n - 2 * k + 1):
            left_max_index = left[i - 1]
            right_max_index = right[i + k]
            total = (pre_sum[i + k] - pre_sum[i]) + \
                (pre_sum[left_max_index + k] - pre_sum[left_max_index]) + \
                (pre_sum[right_max_index + k] - pre_sum[right_max_index])
            if total > mx:
                mx = total
                res = [left_max_index, i, right_max_index]
            
        return res
```

#### 692. Top K Frequent Words
```
from collections import defaultdict
from functools import cmp_to_key
import heapq

class Solution:
    """
    @param: words: an array of string
    @param: k: An integer
    @return: an array of string
    """
    def topKFrequent(self, words, k):
        # write your code here
        # hash = defaultdict(int)
        # hq = []
        
        # for word in words:
        #     hash[word] += 1
        
        # for word, freq in hash.items():
        #     if len(hq) < k:
        #         heapq.heappush(hq, (freq, word))
        #     else:
        #         if (freq, word) > hq[0]:
        #             heapq.heappop(hq)
        #             heapq.heappush(hq, (freq, word))

        # hq.sort(cmp = self.cmp)
        # return [i[1] for i in hq]

        def cmp(a, b):
            if a[0] > b[0] or (a[0] == b[0] and a[1] < b[1]):
                return -1
            elif a[0] == b[0] and a[1] == b[1]:
                return 0
            else:
                return 1

        mapping = defaultdict(int)
        for word in words:
            mapping[word] += 1
        
        words_counts = []
        for word, count in mapping.items():
            words_counts.append([count, word])
        
        words_counts.sort(key=cmp_to_key(cmp))
        res = []
        for i in range(k):
            res.append(words_counts[i][1])
        
        return res
```

#### 694. Number of Distinct Islands
```
from collections import deque

class Solution:
    def numDistinctIslands(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        res = set()
        visited = [[False] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and not visited[i][j]:
                    queue = deque()
                    queue.append((0, 0))
                    visited[i][j] = True
                    # 小技巧！！！
                    # 不好想使用那种hashable的数据结构
                    # 用字符串拼接即可
                    path = ''
                    while queue:
                        ci, cj = queue.popleft()
                        path += '{}, {}'.format(ci, cj)
                        for di, dj in directions:
                            newi, newj = ci + di + i, cj + dj + j
                            if 0 <= newi < m and \
                                0 <= newj < n and \
                                not visited[newi][newj] and \
                                grid[newi][newj] == 1:
                                queue.append([(ci + di), (cj + dj)])
                                visited[newi][newj] = True
                    res.add(path)
        
        return len(res)
```

#### 695. Max Area of Island
```
# BFS
class Solution:
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        visited = [[False] * n for _ in range(m)]
        res = 0
        
        for i in range(m):
            for j in range(n):
                if not visited[i][j] and grid[i][j] == 1:
                    res = max(res, self._bfs(grid, i, j, visited))
        
        return res
    
    def _bfs(self, grid, i, j, visited):
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        m, n = len(grid), len(grid[0])
        queue = [(i, j)]
        visited[i][j] = True
        res = 1
        while queue:
            curr_i, curr_j = queue.pop(0)
            for di, dj in directions:
                new_i, new_j = curr_i + di, curr_j + dj
                if 0 <= new_i < m and \
                    0 <= new_j < n and \
                    not visited[new_i][new_j] and \
                    grid[new_i][new_j] == 1:
                    visited[new_i][new_j] = True
                    queue.append((new_i, new_j))
                    res += 1
        
        return res

## DFS
# class Solution:
#     def maxAreaOfIsland(self, grid):
#         """
#         :type grid: List[List[int]]
#         :rtype: int
#         """
        
#         if not grid or not grid[0]:
#             return 0
        
#         m, n = len(grid), len(grid[0])
#         res = 0
        
#         for i in range(m):
#             for j in range(n):
#                 res = max(res, self._dfs(grid, i, j))
        
#         return res
    
#     # 这里DFS应该定义为从i，j出发
#     # 最多找到多少个能连在一起的点
#     def _dfs(self, grid, i, j):
        
#         m, n = len(grid), len(grid[0])
#         if not 0 <= i < m or not 0 <= j < n:
#             return 0
#         if grid[i][j] == 0 or grid[i][j] == -1:
#             return 0
#         grid[i][j] = -1
        
#         # 注意：这里之前用1 + sum(...)是错的！
#         # 这样[[1, 1], [1, 0]]会返回2
#         # 而正确答案是3
#         return 1 + sum([
#             self._dfs(grid, i + 1, j),
#             self._dfs(grid, i, j + 1),
#             self._dfs(grid, i - 1, j),
#             self._dfs(grid, i, j - 1),
#         ])
```

#### 698. Partition to K Equal Sum Subsets
```
class Solution:
    def canPartitionKSubsets(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        
        if not nums:
            return False
        
        total = sum(nums)
        if total % k != 0:
            return False
        
        target = total // k
        visited = [False] * len(nums)
        return self._dfs(nums, k, target, 0, 0, visited)
    
    def _dfs(self, nums, k, target, start, curr_sum, visited):
        if k == 1:
            return True
        
        if curr_sum == target:
            return self._dfs(nums, k - 1, target, 0, 0, visited)
        
        for i in range(start, len(nums)):
            if visited[i]:
                continue
            visited[i] = True
            if self._dfs(nums, k, target, i + 1, curr_sum + nums[i], visited):
                return True
            visited[i] = False
        
        return False
```

#### 703. Kth Largest Element in a Stream
```
from heapq import heappush
from heapq import heappop

class KthLargest:

    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        # 核心：维护一个大小为k的最小堆
        # 每次入堆得的时候都将堆顶当前堆中最小的元素pop掉
        # 就能保持一个保证存在当前前K大值的堆了
        self._k = k
        self._data = []
        for num in nums:
            heappush(self._data, num)
            if len(self._data) > self._k:
                heappop(self._data)

    def add(self, val):
        """
        :type val: int
        :rtype: int
        """
        heappush(self._data, val)
        if len(self._data) > self._k:
            heappop(self._data)
        return self._data[0]


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)
```

#### 706. Design HashMap
```
class MyHashMap:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._num_buckets = 1000
        self._items_per_bucket = 1000
        self._hash_map = [[] for _ in range(self._num_buckets)]
        

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: void
        """
        hash_key = self._hash_bucket_pos(key)
        if not self._hash_map[hash_key]:
            self._hash_map[hash_key] = [None] * self._items_per_bucket
        inside_bucket_pos = self._hash_inside_bucket_pos(key)
        self._hash_map[hash_key][inside_bucket_pos] = value
        

    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        hash_key = self._hash_bucket_pos(key)
        if not self._hash_map[hash_key]:
            return -1
        inside_bucket_pos = self._hash_inside_bucket_pos(key)
        if self._hash_map[hash_key][inside_bucket_pos] is None:
            return -1
        return self._hash_map[hash_key][inside_bucket_pos]
        

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: void
        """
        hash_key = self._hash_bucket_pos(key)
        if self._hash_map[hash_key]:
            inside_bucket_pos = self._hash_inside_bucket_pos(key)
            self._hash_map[hash_key][inside_bucket_pos] = None
        

    def _hash_bucket_pos(self, key):
        # 这题的重点之1 ！！！！！！！
        return key % self._num_buckets
    
    
    def _hash_inside_bucket_pos(self, key):
        # 这题的重点之2 ！！！！！！！
        return key // self._num_buckets


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
```

#### 708. Insert into a Cyclic Sorted List
```
"""
# Definition for a Node.
class Node:
    def __init__(self, val, next):
        self.val = val
        self.next = next
"""
class Solution:
    def insert(self, head, insertVal):
        """
        :type head: Node
        :type insertVal: int
        :rtype: Node
        """
        if not head:
            new_head = Node(insertVal, None)
            new_head.next = new_head
            return new_head
        
        pre, curr = head, head.next
        while curr is not head:
            if pre.val <= insertVal <= curr.val:
                break
            # 这种情况是因为起始head不一定是这个list中的最小值！！！
            # 比如例子中的3 -> 4 -> 1 -> 2 -> 3
            # 3是头, 要插入5
            # 当pre为4， curr为1时break
            if pre.val > curr.val and (pre.val <= insertVal or curr.val >= insertVal):
                break
            pre = curr
            curr = curr.next
        
        pre.next = Node(insertVal, curr)
        return head
```

#### 713. Subarray Product Less Than K
```
class Solution:
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # 题目要求小于k的乘积
        # 则如果k小于等于1肯定是没有乘积为0的
        if k <= 1:
            return 0
        
        res = 0
        prod = 1
        l = r = 0
        while r < len(nums):
            prod *= nums[r]
            while prod >= k:
                prod //= nums[l]
                l += 1
            # 核心：为什么此时结果可以直接加上r - l + 1?
            # 考虑此时实际上市加上的所有l和r之间以r为结尾的子数组
            # 则此时只能移动l
            res += r - l + 1
            r += 1

        return res
        
        # Naive暴力解法 TLE
#         res = 0
#         for i in range(len(nums)):
#             temp = 1
#             for j in range(i, len(nums)):
#                 temp *= nums[j]
#                 if temp < k:
#                     res += 1
#                 else:
#                     break
        
#         return res
```

#### 716. Max Stack
```
class MaxStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self._data = []
        self._max_stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self._data.append(x)
        if not self._max_stack or self._max_stack[-1] <= x:
            self._max_stack.append(x)
        
    def pop(self):
        """
        :rtype: int
        """
        curr = self._data.pop()
        if self._max_stack and curr == self._max_stack[-1]:
            self._max_stack.pop()
        return curr
        
    def top(self):
        """
        :rtype: int
        """
        return self._data[-1]
        
    def peekMax(self):
        """
        :rtype: int
        """
        return self._max_stack[-1]

    def popMax(self):
        """
        :rtype: int
        """
        curr = self._max_stack.pop()
        
        temp = []
        while self._data[-1] != curr:
            temp.append(self._data.pop())
        
        self._data.pop()
        
        while temp:
            # 注意，这里不能写成self._data.append(temp.pop())
            # 因为如果max_stack在开始pop被清空的情况下
            # 需要在这里重新push进当前的最大值
            self.push(temp.pop())
        
        return curr
        


# Your MaxStack object will be instantiated and called as such:
# obj = MaxStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.peekMax()
# param_5 = obj.popMax()
```

#### 721. Accounts Merge
```
class Solution:
    """
    @param accounts: List[List[str]]
    @return: return a List[List[str]]
    """
    def accountsMerge(self, accounts):
        # 这道题用accounts数组的index作为accounts的id
        self.initialize(len(accounts))
        email_to_ids = self.get_email_to_ids(accounts)
        
        for email, ids in email_to_ids.items():
            root_id = ids[0]
            for _id in ids[1:]:
                self.union(root_id, _id)
        
        id_to_email_set = self.get_id_to_email_set(accounts)
        merged_accounts = []
        for user_id, email_set in id_to_email_set.items():
            merged_accounts.append([
                accounts[user_id][0],
                *sorted(email_set),
            ])
        return merged_accounts
    
    def initialize(self, number_of_total_accounts):
        self.father = {}
        for i in range(number_of_total_accounts):
            self.father[i] = i
    
    def get_email_to_ids(self, accounts):
        email_to_ids = defaultdict(list)
        for i, account in enumerate(accounts):
            for email in account[1:]:
                # email_to_ids[email] = email_to_ids.get(email, [])
                email_to_ids[email].append(i)
        return email_to_ids
    
    def union(self, id1, id2):
        self.father[self.find(id1)] = self.find(id2)
    
    def find(self, user_id):
        path = []
        while user_id != self.father[user_id]:
            path.append(user_id)
            user_id = self.father[user_id]
        
        for u in path:
            self.father[u] = user_id
        
        return user_id
    
    def get_id_to_email_set(self, accounts):
        id_to_email_set = {}
        for user_id, account in enumerate(accounts):
            root_user_id = self.find(user_id)
            email_set = id_to_email_set.get(root_user_id, set())
            for email in account[1:]:
                email_set.add(email)
            id_to_email_set[root_user_id] = email_set
        return id_to_email_set
    

# BFS
# from collections import deque
# from collections import defaultdict

# class Solution:
#     def accountsMerge(self, accounts):
#         """
#         :type accounts: List[List[str]]
#         :rtype: List[List[str]]
#         """
#         n = len(accounts)
#         mapping = defaultdict(list)
        
#         for account_id in range(n):
#             for email in accounts[account_id][1:]:
#                 mapping[email].append(account_id)
        
#         visited = [False] * n
#         res = []
#         for account_id in range(n):
#             if visited[account_id]:
#                 continue
#             queue = deque()
#             # 在放入queue的时候设置visited的值是比较好的时机
#             queue.append(account_id)
#             visited[account_id] = True
#             # 在本次循环中寻找account_id对应的所有emails
#             s = set()
#             while queue:
#                 curr_account_id = queue.popleft()
#                 emails = accounts[curr_account_id][1:]
#                 for email in emails:
#                     s.add(email)
#                     for email_to_account_id in mapping[email]:
#                         if not visited[email_to_account_id]:
#                             queue.append(email_to_account_id)
#                             visited[email_to_account_id] = True
#             res.append([accounts[account_id][0]] + sorted(list(s)))
        
#         return res
```

#### 722. Remove Comments
```
class Solution:
    def removeComments(self, source):
        """
        :type source: List[str]
        :rtype: List[str]
        """
        in_block = False
        res = []
        for line in source:
            i = 0
            if not in_block:
                # 只有当前不在block中
                # 才新建一个newline
                # 否则沿用老的
                newline = []
            while i < len(line):
                # 表示一个块级block开始
                if line[i:i + 2] == '/*' and not in_block:
                    in_block = True
                    i += 2
                # 表示一个块级block结束
                # 此时为何不需要break？
                # 因为/**/形式的注释可能出现在行中
                # 比如`int a; /*this is a demo*/ int b;`
                elif line[i:i + 2] == '*/' and in_block:
                    in_block = False
                    i += 2
                # 表示一个行级注释开始
                # 后面的肯定不用考虑
                elif not in_block and line[i:i + 2] == '//':
                    break
                else:
                    if not in_block:
                        newline.append(line[i])
                    i += 1
            if newline and not in_block:
                res.append(''.join(newline))
        return res
```

#### 730. Count Different Palindromic Subsequences, Hard, Linkedin
```
class Solution:
    def countPalindromicSubsequences(self, S):
        """
        :type S: str
        :rtype: int
        """
        ## http://www.cnblogs.com/grandyang/p/7942040.html
        
        n = len(S)
        M = 10 ** 9 + 7
        
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
        
        for length in range(1, n):
            for i in range(n - length):
                j = i + length
                if S[i] == S[j]:
                    left = i + 1
                    right = j - 1
                    while left <= right and S[left] != S[i]:
                        left += 1
                    while left <= right and S[right] != S[i]:
                        right -= 1
                    if left > right:
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 2
                    elif left == right:
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 1
                    else:
                        dp[i][j] = dp[i + 1][j - 1] * 2 - dp[left + 1][right - 1]
                else:
                    dp[i][j] = dp[i][j - 1] + dp[i + 1][j] - dp[i + 1][j - 1]
                dp[i][j] = dp[i][j] % M
        
        return dp[0][-1]
```

#### 727. Minimum Window Subsequence
```
class Solution:
    def minWindow(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        m, n = len(S), len(T)
        start = -1
        min_len = 2 ** 31 - 1
        # dp[i][j]表示S中前i个字符包含T中前j个字符的话的起始位置(在S中的位置)
        # 注意返回的子串的起始字母和T的起始字母一定相同，这样才能保证最短
        dp = [[-1] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        
        for i in range(1, m + 1):
            # j是遍历T串的
            # 所以j的长度一定是小于当前的S串的长度i
            for j in range(1, min(i, n) + 1):
                if S[i - 1] == T[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = dp[i - 1][j]
            # 说明此时找到一个可能的答案
            # 需要和全局的min_len比较一下
            if dp[i][n] != -1:
                # 我们在dp里存的是起始的下标
                # 所以length就应该是i减去这个下标
                temp_len = i - dp[i][n]
                if temp_len < min_len:
                    min_len = temp_len
                    start = dp[i][n]
        
        return S[start:start + min_len] if start != -1 else ''
```

#### 731. My Calendar II
```
class MyCalendarTwo:

    def __init__(self):
        # 用两个set
        # s1存的是完整区间的集合
        # s2存的是重叠区间（即已经有两个event在这个区间里了）
        # 所以如果新的start, end在这个区域里
        # 下面的book函数之间返回False
        self._s1 = set()
        self._s2 = set()

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        # 两个时间段的重叠区域是两个区间的起始时间中的较大值，到结束时间中的较小值
        for saved_start, saved_end in self._s2:
            # 因为这道题给定的start和end一定是valid的
            # 即start < end
            if start >= saved_end or end <= saved_start:
                continue
            else:
                # 说明有重叠区域
                return False
        for curr_start, curr_end in self._s1:
            if start >= curr_end or end <= curr_start:
                continue
            else:
                # 说明有重叠区域
                # 需要将该重叠区域加入到s2中
                self._s2.add((max(curr_start, start), min(curr_end, end)))
        self._s1.add((start, end))
        return True


# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)
```

#### 732. My Calendar III
```
from collections import defaultdict

class MyCalendarThree:

    def __init__(self):
        # 这道题跟II不一样的地方在于
        # 要求的是在同一时刻在interval start和end之间
        # 有多少个事件
        # 换句话说，在当前的间隔内有多少个事件
        self._freq = defaultdict(int)

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: int
        """
        # 好题！仔细体会
        self._freq[start] += 1
        self._freq[end] -= 1
        
        curr = res = 0
        # 核心之一：注意freq要根据key排序
        for time_stamp in sorted(self._freq):
            curr += self._freq[time_stamp]
            res = max(res, curr)
        
        return res

# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)
```

#### 733. Flood Fill
```
from collections import deque

class Solution:
    def floodFill(self, image, sr, sc, new_color):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        if not image or not image[0]:
            return image
        
        m, n = len(image), len(image[0])
        if not 0 <= sr < m or not 0 <= sc < n:
            return image
        
        old_color = image[sr][sc]
        queue = deque()
        queue.append((sr, sc))
        
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            ci, cj = queue.popleft()
            image[ci][cj] = new_color
            for di, dj in dirs:
                newi, newj = ci + di, cj + dj
                # 这里别忘了如果新点不是就颜色（old_color）
                # 或者已经是新颜色（new_color）时候，都不需要入队
                # 尤其是后者，如果已经是新颜色还入队的话
                # 会使得死循环(反复将这个点入队只因为它不是旧颜色)
                if not 0 <= newi < m or not 0 <= newj < n or \
                    image[newi][newj] != old_color or \
                    image[newi][newj] == new_color:
                    continue
                queue.append((newi, newj))
                image[newi][newj] = new_color
        
        return image
```

#### 745. Prefix and Suffix Search
```
class WordFilter:

    def __init__(self, words):
        """
        :type words: List[str]
        返回具有前缀 prefix 和后缀suffix 的词的最大权重。
        如果没有这样的词，返回 -1。
        """
        self._map = {}
        for inx, word in enumerate(words):
            for i in range(len(word) + 1):
                prefix = word[:i]
                for j in range(len(word) + 1):
                    suffix = word[j:]
                    self._map[prefix + '#' + suffix] = inx

    def f(self, prefix, suffix):
        """
        :type prefix: str
        :type suffix: str
        :rtype: int
        """
        return self._map.get(prefix + '#' + suffix, -1)


# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(prefix,suffix)
```

#### 751. IP to CIDR
```
class Solution:
    def ipToCIDR(self, ip, n):
        """
        :type ip: str
        :type n: int
        :rtype: List[str]
        """
        # 这道题理解题意就比较困难
        # 是说把这个ip每个int8按照二进制连接起来
        # 比如 255.0.0.1 -> 11111111 00000000 00000000 0000001
        # 后面的slash跟着的数字表示去上面的多少个前缀
        # 比如/29就表示取从左到右29个相同前缀，后面3位自由发挥
        # 所以255.0.0.1/29就相当于包括了8个不同的ip，而且这就是一个CIDR块
        # 这道题是问想要通过原始ip表示n个ip，能凑成的最少的CIDR块都是什么
        # 而且要注意，这个CIDR块里的地址是以原始的ip为起始地址往上增加的
        # 也就是说我们只能把最后的0变成1，不能把1变成0
        ips = ip.split('.')
        x = 0
        for i in range(len(ips)):
            x = x * 256 + int(ips[i])
        
        res = []
        while n > 0:
            # last_1肯定是一个2的n次方的数
            # 表示x中最后的1的位置
            last_1 = x & -x
            while last_1 > n:
                last_1 >>= 1
            res.append(self._num_to_ip(x, last_1))
            x += last_1
            n -= last_1
        
        return res
    
    def _num_to_ip(self, x, last_1):
        res = [0] * 4
        
        # res从低到高位先还原出来每个int8原来是多少
        res[0] = str(x & 255)
        x >>= 8
        res[1] = str(x & 255)
        x >>= 8
        res[2] = str(x & 255)
        x >>= 8
        res[3] = str(x)
        
        length = 33
        while last_1 > 0:
            length -= 1
            last_1 >>= 1
        
        return '.'.join(res[::-1]) + '/{}'.format(length)
```

#### 753. Cracking the Safe
```
class Solution:
    def crackSafe(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        if k < 0:
            return -1
        
        # 密码每位数字就是0-K之间（K是0到9）
        # 最短的密码也应该有n位
        res = '0' * n
        # 注意：这里要写set([res])
        # 如果写成set(res)就相当于把res这个字符串里的每个字母都加到set里了
        # 因为res本身就是一个iteralble
        visited = set([res])
        
        # 遍历k的n次方
        for i in range(pow(k, n)):
            pre = res[len(res) - n + 1:]
            for j in range(k - 1, -1, -1):
                curr = pre + str(j)
                if curr not in visited:
                    visited.add(curr)
                    res += str(j)
                    # 因为此时res变长了
                    # 说明我们增加了一种备选方案
                    # 下次一定是继续要在当前基础上变长的
                    # 所以要break掉才能在res基础上重新生成新的pre
                    break
        
        return res
```

#### 759. Employee Free Time
```
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def employeeFreeTime(self, schedule):
        """
        :type schedule: List[List[Interval]]
        :rtype: List[Interval]
        """
        schedule_list = []
        for each_employee in schedule:
            for each_interval in each_employee:
                schedule_list.append((each_interval.start, each_interval.end))
        schedule_list.sort()
        
        res = []
        # 跟merge interval很类似
        # 遍历sort过后的interval
        # 预先存下最小的里的prev_end
        # 然后从第1个开始遍历（0-index）
        # 如果当前遍历的start是大于之前的prev_end
        # 说明出现了一个间隔，放入结果中，并更新prev_end为当前的curr_end
        # 如果当前遍历的start不大于死之前的prev_end
        # 说明有重叠了
        # 此时我们只需要更新prev_end为最大的（最右边的）end值即可
        _, prev_end = schedule_list[0]
        for curr_start, curr_end in schedule_list[1:]:
            if prev_end < curr_start:
                res.append(Interval(prev_end, curr_start))
                prev_end = curr_end
            else:
                prev_end = max(prev_end, curr_end)
        
        return res
```

#### 765. Couples Holding Hands
```
class Solution:
    def minSwapsCouples(self, row):
        """
        :type row: List[int]
        :rtype: int
        """
        res = 0
        n = len(row)
        for i in range(0, n, 2):
            # 每次的i和i+1 suppose是一对儿
            # 小trick：比如2和3应该是一对儿
            # 则2^1正好是3，而3^1正好是2
            if row[i + 1] == row[i] ^ 1:
                continue
            for j in range(i + 1, n):
                if row[j] == row[i] ^ 1:
                    row[j], row[i + 1] = row[i + 1], row[j]
                    res += 1
                    break
        return res
```

#### 767. Reorganize String
```
from collections import Counter

class Solution:
    def reorganizeString(self, s):
        """
        :type S: str
        :rtype: str
        """
        n = len(s)
        ch_counts = [(counts, ch) for ch, counts in Counter(s).items()]

        sorted_s = []
        for counts, ch in sorted(ch_counts):
            if counts > (n + 1) // 2:
                return ''
            sorted_s += ch * counts
        
        # python string是immutable
        # 先转成字符串数组再进行interleaving的操作
        res_list = [None] * n
        sorted_s_list = list(sorted_s)
        # 核心！！！
        # res_list从零开始每隔两位，从1开始每隔两位
        # 分别用sorted_s_list的前后两半来填充
        # 支持slicing的语言都可以这么做
        res_list[::2], res_list[1::2] = sorted_s_list[n // 2:], sorted_s_list[:n // 2]
        return ''.join(res_list)
```

#### 772. Basic Calculator III, Hard, Facebook
```
class Solution:
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        这道题是既有加减乘除又有括号的
        """
        n = len(s)
        # num指的是在循环中遇到的数字（在左右两个op之间或者左右括号之间的数字）
        # curr_res是在出现一个op之后，出现下一个op之前的结果
        # num的优先级要比curr_res高
        num = curr_res = res = 0
        op = '+'
        i = 0
        while i < n:
            ch = s[i]
            if '0' <= ch <= '9':
                num = 10 * num + int(ch)
            elif ch == '(':
                # 所以当前i的位置是左括号
                # 需要找到和它对应闭合的右括号位置j
                cnt = 0
                j = i
                while j < n:
                    if s[j] == '(':
                        cnt += 1
                    if s[j] == ')':
                        cnt -= 1
                    if cnt == 0:
                        break
                    j += 1
                num = self.calculate(s[i + 1:j])
                # 这里是坑：注意要更新i的位置
                # 所以这里用while循环而不是for循环
                i = j
            if ch in '+-*/' or i == n - 1:
                if op == '+':
                    curr_res += num
                elif op == '-':
                    curr_res -= num
                elif op == '*':
                    curr_res *= num
                elif op == '/':
                    curr_res //= num
                if ch in '+-' or i == n - 1:
                    res += curr_res
                    curr_res = 0
                op = ch
                num = 0
            i += 1
        return res
```

#### 773. Sliding Puzzle
```
from collections import deque

class Solution:
    def slidingPuzzle(self, board):
        """
        :type board: List[List[int]]
        :rtype: int
        """
        # 标准BFS题目
        m, n = len(board), len(board[0])
        target = '123450'
        # 核心: dirs数组的第i个元素表示它可以跟周围哪些位置交换（一维数组）
        # 比如0号下标，只能和右边的1下标和下面的3下标交换
        dirs = [
            [1, 3],
            [0, 2, 4],
            [1, 5],
            [0, 4],
            [1, 3, 5],
            [2, 4],
        ]
        
        # 初始化开始的状态
        start = ''
        for i in range(m):
            for j in range(n):
                start += str(board[i][j])

        queue = deque()
        queue.append(start)
        visited = set()

        res = 0
        while queue:
            # 注意这里重点要分层遍历才能更新res！！！！
            # 所以要for循环而不是单纯的每次popleft
            q_len = len(queue)
            for _ in range(q_len - 1, -1, -1):
                curr = queue.popleft()
                if curr == target:
                    return res
                zero_inx = curr.find('0')
                for next_pos in dirs[zero_inx]:
                    # python里面string是immutable
                    # 这里只好先转成list处理
                    curr_list = list(curr)
                    curr_list[next_pos], curr_list[zero_inx] = curr_list[zero_inx], curr_list[next_pos]
                    new_str = ''.join(curr_list)
                    if new_str in visited:
                        continue
                    queue.append(new_str)
                    visited.add(new_str)
            res += 1
        
        # 遍历完了queue都没有找到解答
        # 说明做不到，只能返回-1
        return -1
```

#### 774. Minimize Max Distance to Gas Station
```
from math import floor

class Solution:
    def minmaxGasDist(self, stations, K):
        """
        :type stations: List[int]
        :type K: int
        :rtype: float
        """
        # 这道题正解是二分法
        # 比较难想
        # 二分的是**最小的任意两个加油站之间的最大距离**(
        # 最大距离的最小值，其实就是题目要求解的值)
        # 好处是这道题给定了上下界
        # 因为加油站的位置是在0到10 ** 8之间
        # 所以任意两个加油站之间的距离就在0到10 ** 8之间
        # 我们需要在这个范围内找到一个最小值，
        # 以这个最小间隔，在现有的两两加油站之间插入一共K个新油站
        l, r = 0, 1e8
        while r - l > 1e-6:
            mid = l + (r - l) / 2
            # 说明以现有的mid为interval，在stations
            # 两两之间插入油站的话做不到插入K个新油站
            # 此时说明间隔太大了，需要缩小上届
            if self._helper(stations, K, mid):
                r = mid
            else:
                l = mid
        
        return l
    
    # 核心重点：helper求的是在两两相邻的加油站之间，
    # 能够存放下多少个以当前mid为间隔的加油站！！！
    # 则如果以当前的间隔interval能插入的新加油站数量小于K
    # 说明间隔太大了；反之则说明间隔太小了
    def _helper(self, stations, K, interval):
        cnt = 0
        for i in range(len(stations) - 1):
            cnt += floor((stations[i + 1] - stations[i]) / interval)
        return cnt <= K
```

#### 785. Is Graph Bipartite?
```
# 1. DFS
# class Solution:
#     def isBipartite(self, graph):
#         """
#         :type graph: List[List[int]]
#         :rtype: bool
#         """ 
#         colors = [0] * len(graph)
#         for i in range(len(graph)):
#             if colors[i] == 0 and not self._valid(graph, 1, i, colors):
#                 return False
#         return True
    
#     def _valid(self, graph, curr_color, curr, colors):
#         if colors[curr] != 0:
#             return colors[curr] == curr_color
#         colors[curr] = curr_color
#         for i in graph[curr]:
#             if not self._valid(graph, -1 * curr_color, i, colors):
#                 return False
#         return True

# 2. BFS
from collections import deque

class Solution:
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        输入的每一个子list都是当前index下连接的节点
        """ 
        colors = [0] * len(graph)
        for i in range(len(graph)):
            if colors[i] != 0:
                continue
            queue = deque()
            queue.append(i)
            colors[i] = 1
            while queue:
                curr = queue.popleft()
                for each in graph[curr]:
                    if colors[each] == 0:
                        queue.append(each)
                        colors[each] = -1 * colors[curr]
                    else:
                        if colors[each] == colors[curr]:
                            return False
        
        return True
```

#### 791. Custom Sort String
```
from collections import Counter
class Solution:
    def customSortString(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        count = Counter(T)
        res = []
        # 按顺序遍历S就是能拿到S里字符的顺序
        # 因为这道题说了S和T中都没有重复的字符
        for ch in S:
            # 这里是将T中同时也在S中出现的字符给先给添加到res中
            # 并将T自己的count置为0
            res.append(ch * count[ch])
            count[ch] = 0
        
        for ch in count:
            res.append(ch * count[ch])
        
        # 这里不用filter掉’‘，因为空字符串join还是空字符串
        return ''.join(res)
```

#### 792. Number of Matching Subsequences
```
class Solution:
    def numMatchingSubseq(self, S, words):
        """
        :type S: str
        :type words: List[str]
        :rtype: int
        """
        # 核心思路：将words里的词的iterator加入一个map
        # 遍历一次S,每次遍历到的ch对应的iterator往后移动
        # 如果某个iterator全部跑完了，res加1
        res = 0
        heads = [[] for _ in range(26)]
        for word in words:
            it = iter(word)
            # 先用掉一个it里的开头words!!!
            # 并且指向次一个ch的iterator加入list
            heads[ord(next(it)) - ord('a')].append(it)
        
        for ch in S:
            inx = ord(ch) - ord('a')
            prev = heads[inx]
            heads[inx] = []
            while prev:
                it = prev.pop()
                curr_ch = next(it, None)
                if curr_ch:
                    heads[ord(curr_ch) - ord('a')].append(it)
                else:
                    # 此时说明iterator里已经全部都找到了
                    # curr_ch为None了
                    res += 1
        
        return res
```

#### 803. Bricks Falling When Hit
```
class Solution:

    _DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def hitBricks(self, grid, hits):
        """
        :type grid: List[List[int]]
        :type hits: List[List[int]]
        :rtype: List[int]
        注意条件：最顶层的1（第0行里的1）是不会掉下来的
        """
        # 核心：先hit打掉，再标记剩余不会掉的砖块为2
        res = [0] * len(hits)
        if not grid or not grid[0]:
            return []
        m, n = len(grid), len(grid[0])

        # 注意grid里只有0和1两种情况
        # 所以进行过下面的操作
        # 是不会新的砖块的（为1）
        # 只是将原有的砖块（为1的点）打掉（变成0）
        # 或者标记原来的empty点（值是0的点）为-1
        for i, j in hits:
            grid[i][j] -= 1
        
        # 先将第一行里所有现存的砖块以及和它们相连的砖块给标记成2
        for i in range(n):
            self._dfs(0, i, grid)
        
        for k in range(len(hits) - 1, -1, -1):
            i, j = hits[k]
            # 因为上面有对hits里的每一步-1的操作
            # 这里相当于还原
            grid[i][j] += 1
            # 如果还原之后等于1
            # 说明这个地方是通过hits打下来的砖块
            # 换句话说是每次hit的起始点
            # 如果原来就是0的点
            # 经过初始-1的操作，变成-1
            # 即使经过这步的+1，只会变成0
            # 不会对下面的判断造成影响

            # 核心：
            # 这步里是说此时这个砖块是衔接四周为1和顶部的纽带
            # 所以在它的周围所有为1的砖块都会掉下来
            if grid[i][j] == 1 and self._is_connected(i, j, grid):
                res[k] = self._dfs(i, j, grid) - 1
        
        return res
    
    # dfs里是将和i j相连的砖块用2连接了起来
    # 并返回一共连接了多少个砖块
    def _dfs(self, i, j, grid):
        m, n = len(grid), len(grid[0])
        if not 0 <= i < m or not 0 <= j < n or grid[i][j] != 1:
            return 0
        
        # 这里标记2，既表示这个点被访问过了，又表示这个点被连接过了
        grid[i][j] = 2
        res = 1
        for di, dj in self._DIRECTIONS:
            newi, newj = i + di, j + dj
            res += self._dfs(newi, newj, grid)
        
        return res
    
    # 判断i j点是否和其他值为2的点连接过
    def _is_connected(self, i, j, grid):
        if i == 0:
            return True

        m, n = len(grid), len(grid[0])
        for di, dj in self._DIRECTIONS:
            newi, newj = i + di, j + dj
            if 0 <= newi < m and 0 <= newj < n and grid[newi][newj] == 2:
                return True
        
        return False
```

#### 805. Split Array With Same Average
```
class Solution:
    def splitArraySameAverage(self, A):
        """
        :type A: List[int]
        :rtype: bool
        这道题限制了A中元素非负
        这个条件很重要
        """
        # dp定义：
        # 从A中拿出一些元素
        # 键值表示这些元素的sum和
        # dp的value表示这些元素的个数
        # 初始状态0的sum和不需要任何元素，因此value也为0
        dp = {0: 0}
        len_A, total_A = len(A), sum(A)
        
        for a in A:
            # sub_sum一定要从大到小遍历
            # 因为这道题有个限定A一定是非负元素
            # 而且当前要更新的的状态sub_sum + a
            # 是依赖于之前遍历过的状态sub_sum的
            for sub_sum in sorted(dp.keys(), reverse=True):
                # 当前遍历a时候
                # 凑成sub_sum + a的值肯定是凑成dp[sub_sum]的元素个数再加1
                dp[sub_sum + a] = dp[sub_sum] + 1
                counts = dp[sub_sum + a]
                if counts > 0 and len_A - counts > 0 and \
                    (sub_sum + a) * (len_A - counts) == (total_A - sub_sum - a) * counts:
                    return True
        
        return False
```

#### 815. Bus Routes
```
from collections import deque
from collections import defaultdict

class Solution:
    def numBusesToDestination(self, routes, S, T):
        """
        :type routes: List[List[int]]
        :type S: int
        :type T: int
        :rtype: int
        """
        # 这道题并不是问最短的routes是什么
        # 问的是最少需要换乘几次车
        # 核心思路：广度优先
        # 确定车站到该车站里的buses的映射
        # 具体思路跟200 numbers of island是一样的
        buses_in_stops = defaultdict(set)
        for bus, stops in enumerate(routes):
            for stop in stops:
                buses_in_stops[stop].add(bus)
                
        queue = deque()
        visited = set()
        queue.append((S, 0))
        visited.add(S)
        
        while queue:
            curr_stop, curr_steps = queue.popleft()
            if curr_stop == T:
                return curr_steps
            # 核心：这里用了两个循环
            # 第一层循环遍历从当前车站能联通的车
            # 第二层循环遍历用该车找到该车能到达的新站点
            # 如果新站点没有被访问过，加到queue里即可
            for next_bus in buses_in_stops[curr_stop]:
                for each_stop in routes[next_bus]:
                    if each_stop not in visited:
                        queue.append((each_stop, curr_steps + 1))
                        visited.add(each_stop)
        
        return -1
```

#### 818. Race Car
```
class Solution:
    def __init__(self):
        self._dp = {0:0}

    def racecar(self, target):
        """
        :type target: int
        :rtype: int
        题意：
        遇到A，position += speed并且speed *= 2
        遇到R，speed = -1如果当前speed是1；speed = 1如果
        当前speed已经是-1了
        给定最终的位置target，问最短的sequence（由A和R组成）能够到达target
        """
        dp = [0, 1, 4] + [2 ** 31 - 1] * target
        for t in range(3, target + 1):
            k = t.bit_length()
            if t == 2 ** k - 1:
                dp[t] = k
                continue
            
            # 情况1：走2 ** (k - 1)的距离
            # 然后用一个R和m个A去往回走
            # 这时候已经走了(k - 1) + 1 + (m - 1) + 1 + 2步（即A**(k - 1)RA**(m - 1)R
            # 然后去dp里找剩余的步数t - 2 ** (k - 1) + 2 ** m
            for m in range(k - 1):
                # 注意：后面的k - 1 + j + 2是当前走了这么多步
                dp[t] = min(dp[t], dp[t - 2 ** (k - 1) + 2 ** m] + k - 1 + m + 2)
            
            # 情况2：直接2 ** k步（已经走了最大的k + 1步, 就是在最后min函数括号里加的那一部分，加1是指一个R），然后再往回走
            # 走了k + 1步距离就是2 ** k - 1，这个距离已经超越了t
            # 所以还要继续走2 ** k - 1 - t步才能走到t
            if 2 ** k - 1 - t > 0:
                dp[t] = min(dp[t], dp[2 ** k - 1 - t] + k + 1)
        return dp[target]
```

#### 824. Goat Latin
```
class Solution:
    def toGoatLatin(self, S):
        """
        :type S: str
        :rtype: str
        """
        words = S.split(' ')
        res = []
        for i in range(len(words)):
            curr_word = words[i]
            if curr_word[0] in 'aeiouAEIOU':
                curr_word += 'ma'
            else:
                curr_word = curr_word[1:] + curr_word[0] + 'ma'
            curr_word += 'a' * (i + 1)
            res.append(curr_word)
        
        return ' '.join(res)
```

#### 825. Friends Of Appropriate Ages
```
class Solution:
    def numFriendRequests(self, ages):
        """
        :type ages: List[int]
        :rtype: int
        """
        # 这道题可以做一个假设
        # 就是人的年龄在0到120岁
        # 这样就大大减少了loop次数
        # 非常明智的假设
        age_counter = [0] * 121
        for age in ages:
            age_counter[age] += 1
        
        res = 0
        for age_a, age_a_counts in enumerate(age_counter):
            for age_b, age_b_counts in enumerate(age_counter):
                if age_a * 0.5 + 7 >= age_b:
                    continue
                if age_a < age_b:
                    continue
                if age_a < 100 < age_b:
                    continue
                res += age_a_counts * age_b_counts
                if age_a == age_b:
                    # 实际上减去的是相同的人
                    # 某个人（假设年龄age_a，会被重复计算age_a_counts次）
                    # 所以需要从age_a_counts * age_b_counts中减掉一个age_a_counts
                    # 实际上减去age_a_counts或者age_b_counts都可以
                    # 因为age_a_counts == age_b_counts
                    assert age_a_counts == age_b_counts
                    res -= age_a_counts
        
        return res
```

#### 843. Guess the Word
```
# """
# This is Master's API interface.
# You should not implement it, or speculate about its implementation
# """
#class Master:
#    def guess(self, word):
#        """
#        :type word: str
#        :rtype int
#        """

class Solution:
    def findSecretWord(self, wordlist, master):
        """
        :type wordlist: List[Str]
        :type master: Master
        :rtype: None
        """
        # itertools.permutations(iterable, n)
        # 是从iterable从取出来n个元素
        n = 0
        while n < 6:
            # count的含义是指将wordlist中的词两两比较
            # 看有多少个词跟其他的词完全不一样
            # Counter实例是一个字典，但是支持defaultdict的特性（访问不存在的元素会返回默认值）
            count = collections.Counter(
                w1 for w1, w2 in itertools.permutations(wordlist, 2) \
                    if self._match(w1, w2) == 0
            )
            # 基本思路就是从count中取出与wordlist中的某个词与其他词两两不match这种情况出现最多的词
            # 核心之一：注意这里实际上隐含了一个条件
            # 我们用min函数，如果当前count为空，count[w]就是0
            # 所以我们在这种情况下去wordlist里第一个词（其实就相当于随便取）
            
            # 核心之二：这道题还有一个很重要的intuition，假设秘密词是S，现在有A词和S词match了n个字符
            # n < 6 (说明A并不是答案)，现在有一个B词，如果B词是答案，则B和S应该是一样的（B == S）
            # 所以A也应该与B有n个match
            # 这就是wordlist的更新式子有由来
            guess = min(wordlist, key=lambda w: count[w])
            n = master.guess(guess)
            wordlist = [w for w in wordlist if self._match(w, guess) == n]
            
    def _match(self, w1, w2):
        res = 0
        for i in range(len(w1)):
            if w1[i] == w2[i]:
                res += 1
        return res
```

#### 844. Backspace String Compare
```
class Solution:
    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """
        parsed_S = self._helper(S)
        parsed_T = self._helper(T)
        return parsed_S == parsed_T
    
    # helper定义是将string的#都用了之后的string是什么
    def _helper(self, string):
        if not string:
            return string
        
        res = []
        for ch in string:
            # 如果不是#，res就直接append
            # 如果是#，res就pop
            if ch != '#':
                res.append(ch)
            else:
                # 如果#数目多于当前res的长度
                # 相当于多打了delete键而已
                # 并没有什么影响（当前的res stack已经被清空了）
                if res:
                    res.pop()
        
        return ''.join(res)
```

#### 850. Rectangle Area II
```
class Solution:
    def rectangleArea(self, rectangles):
        """
        :type rectangles: List[List[int]]
        :rtype: int
        """
        # 扫描线经典题
        OPEN, CLOSE = 0, 1
        events = []
        for x1, y1, x2, y2 in rectangles:
            events.append([y1, OPEN, x1, x2])
            events.append([y2, CLOSE, x1, x2])
        events.sort()
        
        # ative里存的是当前的活动的x区间x2 - x1
        active = []
        curr_y = events[0][0]
        res = 0
        for y, flag, x1, x2 in events:
            res += self._query(active) * (y - curr_y)
            if flag is OPEN:
                active.append((x1, x2))
                # 因为python里没有自己带排序的treeset
                # 所以这里需要手动排序一遍
                active.sort()
            else:
                active.remove((x1, x2))
            curr_y = y
        
        return res % (10 ** 9 + 7)
    
    # query函数返回的是当前的active数组里
    # 在水平方向上的活动的x范围（类似活动的x区间）
    def _query(self, active):
        res = 0
        curr = -2 ** 31
        for x1, x2 in active:
            curr = max(curr, x1)
            res += max(0, x2 - curr)
            curr = max(curr, x2)
        return res
```

#### 852. Peak Index in a Mountain Array
```
class Solution:
    def peakIndexInMountainArray(self, A):
        """
        :type A: List[int]
        :rtype: int
        是说A中间肯定存在一个山峰点，找出来对应的坐标
        """
        # 这道题是确保了输入的array一定是符合条件的moutain array
        # 所以省去了很多判断
        start, end = 0, len(A) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if A[mid - 1] < A[mid] < A[mid + 1]:
                start = mid + 1
            elif A[mid - 1] > A[mid] > A[mid + 1]:
                end = mid - 1
            else:
                return mid
        
        if A[start] > A[end]:
            return start
        
        return end
```

#### 857. Minimum Cost to Hire K Workers
```
from heapq import heappush
from heapq import heappop

class Solution:
    def mincostToHireWorkers(self, quality, wage, K):
        """
        :type quality: List[int]
        :type wage: List[int]
        :type K: int
        :rtype: float
        """
        q_w_ratio = sorted((w / q, q, w) for q, w in zip(quality, wage))
        
        max_heap = []
        sum_q = 0
        res = float('inf')
        for ratio, q, w in q_w_ratio:
            heappush(max_heap, -q)
            sum_q += q
            
            if len(max_heap) > K:
                # 相当于把最差性价比的给pop掉
                # 这里实际上相当于减去quality
                # 但是由于我们push的是-q
                # 所以是加号
                sum_q += heappop(max_heap)
            
            if len(max_heap) == K:
                # 核心: 此时的ratio相当于所有K worker的最低ratio
                # 就是基准
                # 根据ratio的公式：ratio = w / q
                # 得出来的就是总的wage
                res = min(res, ratio * sum_q)
        
        return res
```

#### 862. Shortest Subarray with Sum at Least K, Hard, Facebook
```
from collections import deque

class Solution:
    def shortestSubarray(self, A, k):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        n = len(A)
        pre_sum = [0]
        for a in A:
            pre_sum.append(pre_sum[-1] + a)
        
        queue = deque()
        res = n + 1
        for i in range(n + 1):
            while queue and pre_sum[queue[-1]] >= pre_sum[i]:
                queue.pop()
            while queue and pre_sum[i] - pre_sum[queue[0]] >= k:
                front_inx = queue.popleft()
                res = min(res, i - front_inx)
            queue.append(i)
        
        return res if res < n + 1 else -1
```

#### 865. Smallest Subtree with all the Deepest Nodes
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def subtreeWithAllDeepest(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        # 这道题是说在树中找到一个节点
        # 这个节点是书中所有deepest node的根节点
        # 初始用一个None节点表示root的父亲
        depth = {None: -1}
        self._dfs(root, None, depth)
        self._max_depth = max(depth.values())
        return self._helper(root, depth)
    
    # dfs做的事情是根据每个node的父亲节点的深度，更新自己的深度
    # 并将更新过后的值付给depth这个字典
    def _dfs(self, node, parent, depth):
        if node:
            depth[node] = depth[parent] + 1
            self._dfs(node.left, node, depth)
            self._dfs(node.right, node, depth)
    
    # helper定义
    # 以node为根的树中，找到所有深度为全局max_depth的子树，并返回该子树的根
    # 实际上返回的就是题目要求的
    def _helper(self, node, depth):
        if not node:
            return
        if node in depth and depth[node] == self._max_depth:
            return node
        left = self._helper(node.left, depth)
        right = self._helper(node.right, depth)
        if left and right:
            return node
        if left:
            return left
        if right:
            return right
```

#### 879. Profitable Schemes, Hard, Linkedin
```
class Solution:
    def profitableSchemes(self, g, p, groups, profits):
        """
        :type G: int
        :type P: int
        :type group: List[int]
        :type profit: List[int]
        :rtype: int
        """
        
        MOD = 10 ** 9 + 7
        # curr[i][j]表示要挣到j的钱以及最多能用到i个人时候的profit
        curr = [[0] * (g + 1) for _ in range(p + 1)]
        curr[0][0] = 1
        
        # 先按照所有的group，profit对儿进行遍历
        for incoming_g, incoming_p in zip(groups, profits):
            curr2 = [row[:] for row in curr]
            # p1是当前能挣到的profit
            # 而g1是当前最多能用到的人数
            for p1 in range(p + 1):
                # p2是当group，profit对儿里的incoming_p加进来时候的profit
                p2 = min(p1 + incoming_p, p)
                for g1 in range(g - incoming_g + 1):
                    # g2是当group，profit对儿里的incoming_g用到以后的总人数
                    g2 = g1 + incoming_g
                    curr2[p2][g2] += curr[p1][g1]
                    curr2[p2][g2] %= MOD
            curr = curr2
        
        return sum(curr[-1]) % MOD
```

#### 886. Possible Bipartition
```
from collections import defaultdict

class Solution:
    def possibleBipartition(self, N, dislikes):
        """
        :type N: int
        :type dislikes: List[List[int]]
        :rtype: bool
        """
        # 染色法
        # True和False分别代表两种颜色
        graph = defaultdict(list)
        for a, b in dislikes:
            graph[a].append(b)
            graph[b].append(a)
        
        color_map = {}
        for ppl in range(1, N + 1):
            # 对于人群中的每一个人
            # 都先尝试染成一种颜色
            # curr_color = False
            # 如果ppl在color_map里
            # 则说明这个人被之前染过色了
            # 如果不在
            # 说明当前这个人和之前遍历的所有人之间都没有利害关系
            # 而且这个人后面能连接到的群体也一定和之前所有人没有利害关系
            # 所以实际上染成什么颜色都可以
            if ppl in color_map:
                curr_color = color_map[ppl]
            if not self._dfs(ppl, curr_color, color_map, graph):
                return False
        
        return True
    
    # 递归定义：对于刚刚染成颜色curr_color的ppl，递归其孩子，看是否能够染成不一样的颜色
    def _dfs(self, ppl, curr_color, color_map, graph):
        if ppl in color_map:
            return color_map[ppl] == curr_color

        color_map[ppl] = curr_color
        for neighbor in graph[ppl]:
            if not self._dfs(neighbor, not curr_color, color_map, graph):
                return False

        return True
```

#### 896. Monotonic Array
```
class Solution:
    def isMonotonic(self, A):
        """
        :type A: List[int]
        :rtype: bool
        判断是否是单调递增或者单调递减Array
        """
        if not A or len(A) <= 2:
            return True
        
        asc = True
        desc = True
        for i in range(len(A) - 1):
            if A[i] > A[i + 1]:
                asc = False
            if A[i] < A[i + 1]:
                desc = False
        
        return asc or desc
```

#### 904. Fruit Into Baskets
```
from collections import defaultdict

class Solution:
    def totalFruit(self, tree):
        """
        :type tree: List[int]
        :rtype: int
        """
        # 就是说从tree中找到一个子数组
        # 这个子数组中只包含两种水果
        # 这道题实际上应该转化成双指针的题
        # 在tree中找到一个窗口
        # 这个窗口中只包含两种水果
        window = defaultdict(int)
        l = r = 0
        res = 0
        while r < len(tree):
            window[tree[r]] += 1
            while len(window) > 2:
                window[tree[l]] -= 1
                if window[tree[l]] == 0:
                    del window[tree[l]]
                l += 1
            res = max(res, r - l + 1)
            r += 1
        return res
```

#### 913. Cat and Mouse
```
from collections import defaultdict
from collections import deque

class Solution:
    def catMouseGame(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: int
        """        
        DRAW_GAME, MOUSE_WIN, CAT_WIN = 0, 1, 2
        MOUSE_TURN, CAT_TURN = 1, 2
        HOLE_POS, MOUSE_START_POS, CAT_START_POS = 0, 1, 2
        
        def parents(m_curr_pos, c_curr_pos, curr_turn):
            # 如果这一次的turn是猫
            # 则上一次一定是老鼠的turn
            if curr_turn == CAT_TURN:
                for m_prev_pos in graph[m_curr_pos]:
                    yield m_prev_pos, c_curr_pos, MOUSE_TURN
            else:
                for c_prev_pos in graph[c_curr_pos]:
                    # 猫不能到达hole(0点)
                    # 换句话说猫也不能从hold点到curr pos
                    if c_prev_pos != HOLE_POS:
                        yield m_curr_pos, c_prev_pos, CAT_TURN
        
        n = len(graph)
        # outgoings表示能够到达这个key的状态的种类数量
        outgoings = {}
        for m in range(n):
            for c in range(n):
                outgoings[m, c, MOUSE_TURN] = len(graph[m])
                outgoings[m, c, CAT_TURN] = len(graph[c])
                if HOLE_POS in graph[c]:
                    outgoings[m, c, CAT_TURN] -= 1

        # 核心：定义状态为(鼠的位置，猫的位置，谁动)
        # 并将这个状态的结果染色为
        # DRAW_GAME | MOUSE_WIN | CAT_WIN
        color = defaultdict(int)
        queue = deque()
        # 初始化queue
        # 将肯定已知胜负的点先入队
        # 并将这个确定的状态
        for curr_pos in range(n):
            for whose_turn in (MOUSE_TURN, CAT_TURN):
                color[HOLE_POS, curr_pos, MOUSE_TURN] = MOUSE_WIN
                # 队列为四元组
                # (鼠的位置，猫的位置，谁动，谁动谁能赢)
                queue.append((HOLE_POS, curr_pos, whose_turn, MOUSE_WIN))
                if curr_pos != HOLE_POS:
                    color[curr_pos, curr_pos, whose_turn] = CAT_WIN
                    queue.append((curr_pos, curr_pos, whose_turn, CAT_WIN))
        
        # 倒着反推的BFS
        while queue:
            m_curr_pos, c_curr_pos, curr_turn, res = queue.popleft()
            for m_prev_pos, c_prev_pos, prev_turn in parents(
                m_curr_pos,
                c_curr_pos,
                curr_turn,
            ):
                if (m_prev_pos, c_prev_pos, prev_turn) not in color:
                    # 注意这里是因为在开头设置了
                    # MOUSE_WIN, CAT_WIN = 1, 2
                    # MOUSE_TURN, CAT_TURN = 1, 2
                    # 让猫鼠的move标识和他们输赢标志一致了
                    # 还是有疑惑？？
                    if prev_turn == res:
                        color[m_prev_pos, c_prev_pos, prev_turn] = res
                        queue.append((
                            m_prev_pos,
                            c_prev_pos,
                            prev_turn,
                            res,
                        ))
                    else:
                        outgoings[m_prev_pos, c_prev_pos, prev_turn] -= 1
                        if outgoings[m_prev_pos, c_prev_pos, prev_turn] == 0:
                            color[m_prev_pos, c_prev_pos, prev_turn] = \
                                MOUSE_TURN if prev_turn == CAT_TURN else CAT_TURN
                            queue.append((
                                m_prev_pos,
                                c_prev_pos,
                                prev_turn,
                                MOUSE_TURN if prev_turn == CAT_TURN else CAT_TURN,
                            ))
        
        return color[MOUSE_START_POS, CAT_START_POS, MOUSE_TURN]
```

#### 920. Number of Music Playlists
```
from math import factorial

class Solution:
    def numMusicPlaylists(self, N, L, K):
        """
        :type N: int
        :type L: int
        :type K: int
        :rtype: int
        有N首不同的曲子，要凑成L长度的playlist(L >= N)
        要保证N中每首曲子都至少播放一遍
        而且重复的曲子要相隔K
        求有多少种L的组成方式
        """
        # dp[i][j]表示组成j长度的playlist从i长度的unique songs
        # 可以有多少种答案
        dp = [[0] * (L + 1) for _ in range(N + 1)]
        # 题目说明了0 <= K < N <= L <= 100
        # 所以行坐标大于K
        # 至少从K + 1开始(即至少要有K + 1首输入的unique songs)
        for i in range(K + 1, N + 1):
            # 播放列表长度长度大于等于N（即j的长度要大于等于i）
            for j in range(i, L + 1):
                if i == j:
                    dp[i][j] = factorial(i)
                else:
                    # 第一种情况：
                    # 相当于在当前的i首unique songs中剔除去1首曲子（变成i - 1）
                    # 看凑成j - 1长度的list
                    # 所以一共有dp[i - 1][j - 1] * i
                    # 乘以i是因为i中任何一首都可以剔除
                    # 第二种情况：
                    # 用当前的i首unique songs凑成j - 1长度的play list
                    # 则j长度的play list中最后一首曲子只能从前i - k首曲子中挑选
                    dp[i][j] = dp[i - 1][j - 1] * i + dp[i][j - 1] * (i - K)
        
        return dp[N][L] % (10 ** 9 + 7)
```

#### 921. Minimum Add to Make Parentheses Valid
```
class Solution:
    def minAddToMakeValid(self, S):
        """
        :type S: str
        :rtype: int
        """
        if not S:
            return 0
        
        stack = []
        res = 0
        for ch in S:
            if ch == '(':
                stack.append(ch)
            else:
                if not stack:
                    res += 1
                else:
                    temp = stack.pop()
                    if temp != '(':
                        res += 1
        
        return res + len(stack)
```

#### 929. Unique Email Addresses
```
class Solution:
    def numUniqueEmails(self, emails):
        """
        :type emails: List[str]
        :rtype: int
        """
        res = set()
        for email in emails:
            local_name, domain_name = email.split('@')
            local_name = local_name.split('+')[0]
            local_name = local_name.replace('.', '')
            res.add(local_name + domain_name)
        return len(res)
```

#### 936. Stamping The Sequence
```
class Solution:
    def movesToStamp(self, stamp, target):
        """
        :type stamp: str
        :type target: str
        :rtype: List[int]
        """
        # 反向stamp
        # 假设我们把最终的答案aabccbc作为input
        # 看怎么打stamp能让最后的结果变成*******
        # https://leetcode.com/problems/stamping-the-sequence/discuss/189576/C%2B%2B-simple-greedy
        # 'aabccbc' ? 'abc' = [1]
        # 'a***cbc' ? '*bc' = []
        # 'a***cbc' ? 'ab*' = []
        # 'a***cbc' ? 'a**' = [0]
        # '****cbc' ? '*b*' = []
        # '****cbc' ? '**c' = [2]
        # '*****bc' ? '*bc' = [4]
        # 最终结果就是[4, 2, 0, 1]
        res = []
        total_stamp = 0
        turn_stamp = -1
        
        while turn_stamp != 0:
            turn_stamp = 0
            for sz in range(len(stamp), 0, -1):
                for i in range(len(stamp) - sz + 1):
                    new_stamp = '*' * i + stamp[i:i + sz] + '*' * (len(stamp) - sz - i)
                    pos = target.find(new_stamp)
                    while pos != -1:
                        res.append(pos)
                        turn_stamp += sz
                        target = target[:pos] + '*' * len(stamp) + target[pos + len(stamp):]
                        pos = target.find(new_stamp)
            total_stamp += turn_stamp
        res = res[::-1]
        return res if total_stamp == len(target) else []
```

#### 940. Distinct Subsequences II
```
class Solution:
    def distinctSubseqII(self, S):
        """
        :type S: str
        :rtype: int
        """
        endswith = [0] * 26
        # 思路是寻找以某一个字符为结尾的字符
        # 按照顺序遍历S，这样保证此时遍历的字符ch的意义很明确
        # 就是以当前字符ch结尾的子序列
        # 然后用endswith的和（正好此时和就是前一个字符结尾的所有子序列的和）
        for ch in S:
            endswith[ord(ch) - ord('a')] = sum(endswith) + 1
        return sum(endswith) % (10 ** 9 + 7)
```

#### 953. Verifying an Alien Dictionary
```
class Solution:
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        if not words or len(words) <= 1:
            return True

        mapping = {}
        for inx, ch in enumerate(order):
            mapping[ch] = inx
        
        for i in range(1, len(words)):
            last_word = words[i - 1]
            curr_word = words[i]
            for j in range(min(len(last_word), len(curr_word))):
                if last_word[j] != curr_word[j]:
                    if mapping[last_word[j]] > mapping[curr_word[j]]:
                        return False
                    break
            else:
                # 如果没有找到不一样的字符
                # 则理论上短的word应该小于长的word
                # 就是说last_word的长度在这个else block里应该是小于curr_word的长度
                # 比如 "app"是应该小于"apple"的
                # 所以如果出现"app" > "apple"
                # 可以直接返回False
                if len(last_word) > len(curr_word):
                    return False

        return True
```