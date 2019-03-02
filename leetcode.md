## Leetcode

#### 1. Two Sum
```
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # nums中找两个数，和为target
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
        # 两个用链表表示的数相加
        # 最终返回的是从最低位到最高位
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

#### 4. Median of Two Sorted Arrays
```
class Solution:
    def findMedianSortedArrays(self, A, B):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        # A, B两个排序过的数组，如果A的中位数比B的中位数小
        # 则总的A + B这个长的数组的中位数一定不在A的左半部分
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
        # dp[i][j]定义：s从i到j是不是回文
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
        
        # 很好的思路：因为负数不好处理，同一先变成正数
        # 再存下符号位
        sign = 1
        if x < 0:
            sign = -1
            x = -x
        
        reverse = 0
        while x > 0:
            reverse = 10 * reverse + x % 10
            x //= 10
        reverse *= sign
        
        # 越界就直接返回0
        if not -2 ** 31 <= reverse <= 2 ** 31 - 1:
            return 0
        return reverse
```

#### 8. String to Integer (atoi)
```
class Solution:
    def myAtoi(self, s):
        """
        :type str: str
        :rtype: int
        """
        # 这道题是指输入为100， 99， -3等等有效的数字字符串
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
        # 负数肯定不是答案
        if x < 0:
            return False
        
        temp = x
        y = 0
        
        # 实际上解题思路就是reverse numerb那道题
        while temp:
            y = y * 10 + temp % 10
            temp = temp // 10

        return y == x
```

#### 10. Regular Expression Matching
```
# 递归
class Solution(object):
    def isMatch(self, text, pattern):
        # 这道题是要判断pattern是否和text正则匹配
        # .是表示匹配任意一个字符
        # *是表示匹配任意数量的字符
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
            # 这样显然不管p[i - 2]个字符是什么都可以（可以为空，所以可以不考虑因为*号就表示任意数量，当然也包括0个）
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
                            (p[j - 2] == s[i - 1] or p[j - 2] == '.')
                        )
                    )
                elif p[j - 1] == '.':
                    # 为.就很简单
                    # .表示任意一个字符（可以是任何只要非空字符的意思）
                    # 所以不用管当前的s[i - 1]了
                    # 之间看s的前i - 1和p的前j - 1（即dp[i - 1][j - 1]是否匹配即可）
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # 其他情况就是单n纯的当前字符匹配，并且前i - 1和前j - 1匹配
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
        # 两根指针经典思路(反向双指针)
        res = 0
        i, j = 0, len(heights) - 1
        
        while i < j:
            res = max(res, min(heights[i], heights[j]) * (j - i))
            # 增大较小的指针
            # 短板效应 最长的并不能决定最多的水
            # 最短的才能决定
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
        # 记忆： 10 9 5 4 最后还有一个1
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
            # 说明是IV=4的情况
            if roman[s[i]] < roman[s[i + 1]]:
                res -= roman[s[i]]
            # 说明是VI=6的情况
            else:
                res += roman[s[i]]
        
        # 别忘了加上最后一个
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

        # 只遍历最短的字符串就好了
        min_len = min(len(s) for s in strs)
        res = ''
        for i in range(min_len):
            pivot_ch = strs[0][i]
            if any(word[i] != pivot_ch for word in strs[1:]):
                # 核心：出现了不相等
                # 直接break就好了
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
        这道题问的是nums中3个数字和为0的下标的所有的情况
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
                    # 核心之一：注意可能有重复
                    # 所以这里左右指针都要跳过重复的情况
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
    
    _LETTER_MAP = {
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
        这道题说的是给定一个digits，返回所有可能的字符串组合
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
        for c in self._LETTER_MAP[digits[index]]:
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
        这道题要求的是4个数字组成target的情况
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
        
        # 因为p2正好是要删除的node的前一位
        # 所以一句就能删除
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

#### 23. Merge k Sorted Lists
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
            # 这里pre相当于跳了两个位置
            pre = temp.next
        
        return dummy_head.next
```

#### 25. Reverse Nodes in k-Group
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
        # left -> (last -> .....) 括号里的就是需要翻转的部分
        #          curr
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
        # 这个last_node最终会放在链表最后一位
        # 我们需要用这个last_node确定pre的下一个位置
        # 所以这道题要返回的是last_node
        last_node = left.next
        curr = last_node.next
        
        while curr is not right:
            last_node.next = curr.next
            curr.next = left.next
            left.next = curr
            curr = last_node.next
        
        return last_node
```

#### 26. Remove Duplicates from Sorted Array
```
class Solution:
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        这道题是说原地"remove"
        即将所有不重复的数字放到数组前面
        最终返回有多少个重复的数字
        """
        if not nums:
            return 0
        
        if len(nums) == 1:
            return 1
        
        last_pos = 0
        for i in range(1, len(nums)):
            # 因为这道题是sorted array
            # 所以可以这么做
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
            # 这道题OJ的not haystack and not needle情况不太合理
            # 也应该返回-1 但是OJ要求0
            return 0 if not needle else -1
        if not needle:
            return 0
        
        # i指针表示haystack里的位置
        # j指针表示needle里的位置
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
```
class Solution:

    _INT_MAX = 2 ** 31 - 1
    _INT_MIN = -2 ** 31

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
            # 这里的shift就是dividend最终要减去的那个数的左移量（乘了多少个2）
            shift = 1
            while dividend >= (divisor << shift):
                shift += 1
            shift -= 1

            dividend -= divisor << shift
            res += 1 << shift
        
        # 最终判断下是否越界
        if sign * res > self._INT_MAX:
            return self._INT_MAX
        
        if sign * res < self._INT_MIN:
            return self._INT_MIN:
        
        return sign * res
```

#### 30. Substring with Concatenation of All Words
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
        输入一个string和words，返回所有string中的下标
        这个下标开始能够有所有words中词的连接
        """
        if not s or not words:
            return []
        
        # 这道题是滑动窗口问题
        # 先用一个mapping存住words中的词（因为words中的词可能有重复）
        mapping = defaultdict(int)
        for word in words:
            mapping[word] += 1
        
        res = []
        len_s = len(s)
        n, m = len(words), len(words[0])
        for i in range(len_s - n * m + 1):
            temp_mapping = defaultdict(int)
            for j in range(n):
                # 这里的new_str就是s中从i + j * m这个位置开始长度为m的子串（一段一段的变化）
                # 先检查这个子串是不是words中的一部分，如果不是直接break
                # 然后将这个词放入temp_mapping中
                # 最后再查一下temp_mapping中存的这个new_str是否多于了words中这个new_str本身出现的次数
                # 如果是，直接break
                new_str = s[i + j * m:i + j * m + m]
                if new_str not in mapping:
                    break
                temp_mapping[new_str] += 1
                if temp_mapping[new_str] > mapping[new_str]:
                    break
            # 最终如果没有break过，说明当前外层遍历的这个i是一个可行解，将这个i放入到res中
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
        # 1. pos1: 从后往前遍历寻找第一个值变小的index（num[i] < num[i + 1]）。
        # 2. pos2: 从后往前遍历寻找第一个比num[pos1]大的index。(这个位置肯定在pos1的右边)
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
            # 如果没有break过
            # 说明当前的这个排列就是最大的（即54321）
            # 所以直接reverse返回即可
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

#### 32. Longest Valid Parentheses
```
class Solution:
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 当前的stack只有长度大于等于2的时候才有意义
        # stack里存的是当前最长有效字符串的长度
        stack = [0]
        res = 0
        
        for ch in s:
            # 这道题只考虑()，不考虑其他字符
            if ch == '(':
                stack.append(0)
            elif ch == ')':
                if len(stack) > 1:
                    top = stack.pop()
                    # +2是因为我们在append左括号的时候append长度为0
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
        
        # 这道题核心：需要洗判断mid落在了两段中的哪一段
        # 使用当前的start或者end判断
        # 只有在确定好了mid括在那一段以后
        # 再去判断target落在mid的哪一边
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

        # 基本思路就是先去二分查找target的起始位置
        # 再二分查找target的结束位置
        start, end = 0, n - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            # 因为是找起始位置，所以end+1（end右边的）肯定不是解，直接舍弃
            if nums[mid] >= target:
                end = mid
            else:
                start = mid + 1
        
        # 小陷阱：注意下面两个if的顺序
        # 要先判断startPos 再判断endPos
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

#### 36. Valid Sudoku
```
class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # create 9 empty set, python2-> set([]), python3-> set()
        # 基本思路就是检查3种group
        # 行，列，和grid
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
                # 比如有一个100 * 100的棋盘 变成10 * 10的小棋盘（一共有100个，所以用一个100长度的array）
                # 对于坐标i j来说，对应到小棋盘array里的index就是 i // 10 * 10 + j // 10
                # 因为//是向下取整操作
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
            # 因为有多重选择了，开始递归
            for k in range(1, 10):
                board[i][j] = str(k)
                # 注意第二个条件，增加列数来推动下一层递归
                if self._is_valid(board, i, j) and self._dfs(board, i, j + 1):
                    return True
                board[i][j] = '.'
        else:
            # 同上，增加列数来推动下一层递归
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
        # 这个边界值不太make sense
        # 虽然是OJ要求的
        if n <= 0:
            return '1'
        # 对于前一个数
        # 找出相同元素的个数(存在cnt中)
        # 把个数和该元素存到curr里
        # 最后在每次循环里更新res
        # 核心：数相同的数字！(这叫count)
        res = '1'
        i = 0
        for _ in range(n - 1):
            curr = ''
            i = 0
            # 相当于每次都重新遍历一遍上次的res
            while i < len(res):
                cnt = 1
                while i + 1 < len(res) and res[i] == res[i + 1]:
                    cnt += 1
                    # 注意这里和下面都更新了i
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
        # 这道题是求所有和为target的集合
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
            # 如果当前的candidates[i]已经大于target了
            # 将这个数字加入到结果中是不可能凑成target的
            # 而且因为candidates是排过序的
            # 所以后面的也不用看了，直接break掉
            if candidates[i] > target:
                break
            # 这道题要求不能有重复的解，即使[2, 2]和[2, 2]是同一个解
            if i != 0 and candidates[i] == candidates[i - 1]:
                continue
            current.append(candidates[i])
            # 之所以用i，而不是i + 1是因为可以重复使用
            # 注意理解：可以(重复使用)和(重复的解)是不一样的概念
            # 只能有一个[2, 2]因为不能有重复的解，但是可以有两个2是因为2可以重用
            self.helper(candidates, target - candidates[i], i, current, results)
            current.pop()
```

#### 40. Combination Sum II
```
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        if not candidates:
            return target == 0
        
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
            if target >= candidates[i]:
                self._dfs(
                    candidates,
                    target - candidates[i],
                    i + 1,
                    curr + [candidates[i]],
                    res,
                )
```

#### 41. First Missing Positive
```
class Solution:
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 这道题是unsorted的
        n = len(nums)

        for i in range(n):
            # 后面的条件是说nums[i]这个数字对应的位置应该在nums[nums[i] - 1]上
            # 比如nums = [1, 2, 3] 对应的位置就是完全吻合的
            while nums[i] > 0 and nums[i] <= n and nums[nums[i] - 1] != nums[i]: 
                # python交换操作
                # 先将等号左边的第一个数字存为temp
                # 然后所有对左边第一个数字的赋值操作（对应右边第一个数字）
                # 是使用temp的
                # 此时对左边第二个数字再进行赋值操作
                # 但是这时候引用的nums[i]是之前变化过的！！！！！
                # 这是个大坑要注意！！！
                # 这道题下面的交换顺序是通不过的！！！
                # nums[i], nums[nums[i] - 1] = nums[nums[i] - 1], nums[i]

                # 下面的交换就是说如果nums[i]不在应该在的位置上，就把num[i]放到应该放的位置上
                nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
        
        # 每个i位置对应的元素就应该是i + 1
        # 如果不是就立即返回
        # 类似贪心的思路
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        
        return n + 1
```

#### 42. Trapping Rain Water
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
        # 和11题container with most water也不同
        # 11题求的是最多的水
        # 这里是只要有坑就能加水
        stack = []
        res = 0
        i = 0
        while i < len(heights):
            if not stack or heights[i] <= heights[stack[-1]]:
                stack.append(i)
                i += 1
            else:
                # 注意此时不累加i了
                curr_bottom = stack.pop()
                if not stack:
                    continue
                # 核心：第一个括号里其实就是左右边界
                # heights[i]就是右边界
                # heights[stack[-1]]就是左边界
                # 刚刚pop出来的heights[curr_bottom]是此时的坑
                # 左右边界的最小值再减去坑的高度就是能装多少水
                res += (min(heights[i], heights[stack[-1]]) - heights[curr_bottom]) * (i - stack[-1] - 1)
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

        # 由于num3里面左边可能有很多0，所以要filter掉左边的0s
        left = 0
        while left < len_num1 + len_num2 - 1 and num3[left] == 0:
            left += 1
        
        return ''.join(str(i) for i in num3[left:])
```

#### 44. Wildcard Matching
```
class Solution:
    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        # 这道题是要支持？和*
        # ？是任何一个字符
        # * 是任意数量的任何字符
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
                    # 下面如果还能匹配，p中的当前字符应该是"?"或者是"*"
                    # 在这种情况下p中的这个*就表示可以没有
                    dp[i][j] |= dp[i - 1][j] and p[j - 1] in ('*', '?')
                if j > 0:
                    # 前i个s中的字符和前j - 1个p中的字符已经匹配
                    # 下面如果还能匹配，p中的当前字符也一定是*
                    # 在这种情况下p中的这个*就表示可以是任意一个字符(包括0个)
                    # 此时这个*就相当于去匹配了一个空字符
                    # 所以这时p[j - 1]不能为"?"
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
        jump I 是问能不能跳到最后
        """
        steps = 0
        n = len(nums)
        # curr_i表示遍历的nums里的坐标
        # curr_most_i表示当前能跳到的最远距离的坐标
        curr_i = curr_most_i = 0
        while curr_most_i < n - 1:
            steps += 1
            # 因为在下面的循环里要更新curr
            # 所以预先保存下pre
            pre_most_i = curr_most_i
            while curr_i <= pre_most_i:
                # 注意：这道题保证了curr_i一定是nums中valid index
                # 这道题的nums里存的内容也是index
                curr_most_i = max(curr_most_i, curr_i + nums[curr_i])
                curr_i += 1
            # 这里是说经过上面的遍历
            # 发现跳不远了
            # 所以直接返回-1
            if pre_most_i == curr_most_i:
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
            # 去除掉当前的nums[i] 递归下去看有多少种全排列
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
        # 这题是说nums中有可能有重复元素
        # 如果有重复元素 就加一次和i - 1的判断
        # 如果是可以重复使用元素 就再递归的时候不加1直接递归i
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
        # 这道题的image是n * n的
        # 将这个image顺时针旋转90度
        # 1 2   to  3 1
        # 3 4       4 2
        row = col =len(matrix)
        
        # 先转置
        # 1 2
        # 3 4
        # 变成
        # 1 3
        # 2 4
        for i in range(row):
            for j in range(i + 1, col):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        # 再垂直翻转（将image垂直flip）
        # 1 3
        # 2 4
        # 变成
        # 3 1
        # 4 2 
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
        # 这道题是要将strs这个list中同样pattern的词group起来
        # 用mapping存每一个词的pattern
        mapping = defaultdict(list)
        for each in strs:
            # ''.join(sorted(each)) 就是在hash当前each这个词的key
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

#### 51. N-Queens
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
    # positions的长度就是一共有多少行(同时也能表示一共有多少列)
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
            # 因为这道题是正方形的棋盘，所以可以用curr_row_inx来代替列的坐标
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
        # 和1不同的地方就是求出来有多少种答案
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
                # 跟上面一道题几乎一模一样
                # 只不过不需要求path了
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

    _INT_MIN = -2147483648

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ## 读题意：这道题是要返回最大的和，不是最大和的子序列长度！！！
        if not nums:
            return self._INT_MIN
        
        local_max = global_max = self._INT_MIN
        for num in nums:
            # local_max的这行判断实际上附带判断num是否大于0
            local_max = max(local_max + num, num)
            # global_max这行是计算全局
            global_max = max(global_max, local_max)
        
        return global_max
```

#### 54. Spiral Matrix	
```
class Solution:
    def spiralOrder(self, matrix: 'List[List[int]]') -> 'List[int]':
        # 好题！多看！
        if not matrix or not matrix[0]:
            return []
        
        m, n = len(matrix), len(matrix[0])
        res = []
        num_circles = (min(m, n) + 1) // 2
        
        curr_width, curr_height = n, m
        for i in range(num_circles):
            for col in range(i, i + curr_width):
                res.append(matrix[i][col])
            for row in range(i + 1, i + curr_height):
                res.append(matrix[row][i + curr_width - 1])
                
            if curr_width == 1 or curr_height == 1:
                break
                
            for col in range(i + curr_width - 2, i - 1, -1):
                res.append(matrix[i + curr_height - 1][col])
            for row in range(i + curr_height - 2, i, -1):
                res.append(matrix[row][i])
            
            curr_width -= 2
            curr_height -= 2
        
        return res
```

#### 55. Jump Game
```
class Solution:
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        I这道题是问能不能跳到最后
        jump game II是问最少跳几步能跳到最后
        """
        # reach表示当前能到达的最远坐标
        reach = 0
        n = len(nums)
        for i in range(n):
            # 第一个条件就是说从i跳不到reach了
            # 第二个条件是说能不能跳到最后（最后的位置就是n - 1）
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

#### 57. Insert Interval
```
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def insert(self, intervals, new_interval):
        """
        :type intervals: List[Interval]
        :type new_interval: Interval
        :rtype: List[Interval]
        """
        if not intervals:
            return [new_interval]
        
        res = []
        insert_pos = 0
        n = len(intervals)
        for current_interval in intervals:
            ## 说明new_interval在current interval的左边（不重叠）
            if current_interval.start > new_interval.end:
                res.append(current_interval)
            ## 说明new_interval在current interval的右边（不重叠）
            elif current_interval.end < new_interval.start:
                ## 注意：这个地方应该是+=1,而不是i,因为i是不准的!!!!!
                insert_pos += 1
                res.append(current_interval)
            ## 说明new_interval和current interval有重叠
            ## 在这种情况下重新update new_interval (这个思路很重要，往往出于思维定式不会想到去修改输入)
            else:
                new_interval.start = min(new_interval.start, current_interval.start)
                new_interval.end = max(new_interval.end, current_interval.end)
        # python list的insert函数，第一个参数是位置，第二个参数数插入的元素 
        res.insert(insert_pos, new_interval)
        return res
```

#### 62. Unique Paths
```
class Solution:
    def uniquePaths(self, m: 'int', n: 'int') -> 'int':
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] += dp[i - 1][j] + dp[i][j - 1]
        
        return dp[-1][-1]
```

#### 63. Unique Paths II
```
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: 'List[List[int]]') -> 'int':
        # 跟1不一样的地方就是有障碍物点
        if not obstacleGrid or not obstacleGrid[0]:
            return 0
        
        m, n = len(obstacleGrid), len(obstacleGrid[0])
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

#### 65. Valid Number
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
        
        # 当前字符串经过trim以后，每个字符有5种情况：
        # 1. 数字
        # 2. 小数点
        # 3. e
        # 4. 加减号
        # 5. 其他，直接返回False
        # 口诀：数点e sign （15.13）
        for i in range(n):
            if '0' <= s[i] <= '9':
                num = True
                allow_e = True
            elif s[i] == '.':
                # 前面如果出现过expr再出现"."是不合法的
                if dot or expr:
                    return False
                dot = True
            elif s[i] == 'e':
                # e前面必须有数字
                if expr or not num:
                    return False
                expr = True
                allow_e = False
            elif s[i] in '+-':
                # 如果+-号不是出现在第一个位置的话，前面必须是e
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
        # 可以把这个carry理解成要加的1
        # 这道题实际上和A+B那道题差不多
        carry = 1
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] + carry == 10:
                digits[i] = 0
            else:
                # 此时说明以后不需要进位了
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

#### 68. Text Justification
```
class Solution:
    def _format(self, line, max_width):
        # 下面的if说明此时line里只有一个词
        if len(line) == 1:
            return line[0] + ' ' * (max_width - len(line[0]))
        total_words_length = sum(len(w) for w in line)
        
        # 比如当前line里有5个词，所以应该有5 - 1 = 4个gap
        res_str, gaps = line[0], len(line) - 1
        # 注意index还是从0开始的
        for index, word in enumerate(line[1:]):
            if index < (max_width - total_words_length) % gaps:
                # max_width - total_words_length就是当前有多少个available的空格
                # 因为有gaps个gap，所以每个词之间分到了(max_width - total_words_length) // gaps个空格
                res_str += ' ' + ' ' * ((max_width - total_words_length) // gaps) + word
            else:
                res_str += ' ' * ((max_width - total_words_length) // gaps) + word
        return res_str

    def _format_last(self, line, max_width):
        res_str = ' '.join(line)
        return res_str + ' ' * (max_width - len(res_str))

    def fullJustify(self, words, max_width):
        """
        :type words: List[str]
        :type max_width: int
        :rtype: List[str]
        """
        curr_line, curr_line_length = [], 0
        results = []
        for w in words:
            # len(curr_line)是指由多少个空格
            if curr_line_length + len(w) + len(curr_line) <= max_width:
                curr_line_length += len(w)
                curr_line.append(w)
            else:
                # 说明加上现在这个词会越界
                # 直接返回format现有的行
                # 并重置为下一行
                results.append(self._format(curr_line, max_width))
                curr_line_length = len(w)
                curr_line = [w]
        if curr_line:
            results.append(self._format_last(curr_line, max_width))
        return results

        # 最简洁的做法！！！
        # lines = []
        # curr_line = []
        # one_line_letters = 0
        # for word in words:
        #     if len(word) + one_line_letters + len(curr_line) > maxWidth:
        #         for i in range(maxWidth - one_line_letters):
        #             curr_line[i % (len(curr_line) - 1 or 1)] += ' '
        #         lines.append(''.join(curr_line))
        #         curr_line = []
        #         one_line_letters = 0
        #     curr_line.append(word)
        #     one_line_letters += len(word)
        # lines.append(' '.join(curr_line).ljust(maxWidth))
        # return lines
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

#### 70. Climbing Stairs
```
class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if 0 <= n <= 1:
            return 1
        
        a, b = 1, 1
        for i in range(2, n + 1):
            a, b = b, a + b
        
        return b
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
            # 核心思路:从当前的/位置开始(如果有的话)找到下一个/的位置
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
        
        return '/' + '/'.join(res)
```

#### 72. Edit Distance
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
        这道题是说将所有0所在的行和列全部置为0
        """
        if not matrix or not matrix[0]:
            return
        
        # 因为要确定第一行（第一列）的零到底是由于中间
        # 元素变成的零还是原来就有的零
        # 所以开始就记录一下
        # 这样如果是开始就有零
        # 在最后就需要将第一行（第一列）全部置零
        rows, cols = len(matrix), len(matrix[0])
        empty_first_row = empty_first_col = False
        
        for i in range(cols):
            if matrix[0][i] == 0:
                empty_first_row = True
                break
        
        for i in range(rows):
            if matrix[i][0] == 0:
                empty_first_col = True
                break
        
        # 先遍历一次全体（除了第一行和第一列）
        # 将第一行和第一列相应的地方置0
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
        
        # 然后根据第一行和第一列的0，再反过来将i j里面的置0
        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # 最后如果开始第一行就有0存在
        # 说明在最后要将第一行全部置0,列也一样
        if empty_first_row:
            for i in range(cols):
                matrix[0][i] = 0
        
        if empty_first_col:
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
        # 左下角这个位置的元素是有两种选择的：向上或者向下
        # 每种选择都会使求出来的值有两种相反的变化方向
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
        # 三根指针 在大循环中遍历i
        # last_zero_pos含义是0的最后一个位置
        # first_two_pos含义是2的第一个位置
        # i是循环变量
        last_zero_pos = -1
        i = 0
        first_two_pos = len(nums)

        while i < len(nums):
            if nums[i] == 0:
                last_zero_pos += 1
                nums[last_zero_pos], nums[i] = nums[i], nums[last_zero_pos]
                i += 1
            elif nums[i] == 1:
                i += 1
            # 此时nums[i] == 2
            else:
                first_two_pos -= 1
                nums[i], nums[first_two_pos] = nums[first_two_pos], nums[i]
```

#### 76. Minimum Window Substring
```
from collections import defaultdict

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
        if not t or not s:
            return ''
    
        t_counter = defaultdict(int)
        for ch in t:
            t_counter[ch] += 1
        required = len(t_counter)
        
        l = r = 0
        matched = 0
        s_counter = defaultdict(int)
        min_len = 2 ** 31 - 1
        res = ''
        while r < len(s):
            s_counter[s[r]] += 1
            if s_counter[s[r]] == t_counter[s[r]]:
                matched += 1
            while matched == required:
                if min_len > r - l + 1:
                    min_len = r - l + 1
                    res = s[l:r + 1]
                s_counter[s[l]] -= 1
                if s_counter[s[l]] < t_counter[s[l]]:
                    matched -= 1
                l += 1
            r += 1
        
        return res
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
        self._dfs(n, k, 1, [], res)
        return res
    
    # df定义：从start开始到n，在现有curr基础上遍历
    # 往curr里添加数字
    def _dfs(self, n, k, start, curr, res):
        if len(curr) == k:
            res.append(curr[:])
            return
        
        for i in range(start, n + 1):
            curr.append(i)
            self._dfs(n, k, i + 1, curr, res)
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
        
        # 这个corner case开始就没考虑到！！！
        # 说明下面的DFS并没有cover住这个case
        # 比如board = [['a']], word = 'a'
        if m == 1 and n == 1 and len(word) == 1:
            return board[0][0] == word[0]

        visited = [[False] * n for _ in range(m)]

        for i in range(m):
            for j in range(n):
                
                if self._dfs(board, word, i, j, visited):
                    return True
        
        return False
    
    # dfs定义是在board上以当前的i j为起点，能不能找到word这个词
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

            # 核心：有重复的数值，所以要加一次判断
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

#### 84. Largest Rectangle in Histogram
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
        # 触发出栈操作，维护单调递减栈的性质，并更新res的结果，
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
        
        # 先判断下s1和s2所有的字符的数目是否相同
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
        # 这道题是保证了nums1足够装下nums2了
        # 所以在nums1上in-place修改，将nums放入
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
        # 留下的空肯定全部是属于nums2的
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
        if not s or s[0] == '0':
            return 0
        
        n = len(s)
        # dp[i]定义：前i个字符有多少种解码方式
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 9 if s[0] == '*' else 1
        # Python 1e9是一个float number
        M = int(1e9 + 7)
        
        for i in range(2, n + 1):
            # 核心思想：判断当前字符是不是0， 1-9，或者是*
            # 前一个有一个固定选择，就延续加一种
            if s[i - 1] == '0':
                if s[i - 2] == '1' or s[i - 2] == '2':
                    dp[i] = dp[i - 2]
                elif s[i - 2] == '*':
                    dp[i] = 2 * dp[i - 2]
                else:
                    return 0
            elif '1' <= s[i - 1] <= '9':
                if (s[i - 2] == '1') or (s[i - 2] == '2' and '0' <= s[i - 1] <= '6'):
                    dp[i] = dp[i - 1] + dp[i - 2]
                elif s[i - 2] == '*':
                    if '0' <= s[i - 1] <= '6':
                        dp[i] = dp[i - 1] + 2 * dp[i - 2]
                    else:
                        dp[i] = dp[i - 1] + dp[i - 2]
                else:
                    dp[i] = dp[i - 1]
            else:
                # 此时s[i - 1] == '*'
                if s[i - 2] == '1':
                    dp[i] = 9 * dp[i - 1] + 9 * dp[i - 2]
                elif s[i - 2] == '2':
                    dp[i] = 9 * dp[i - 1] + 6 * dp[i - 2]
                else:
                    dp[i] = 9 * dp[i - 1] + 15 * dp[i - 2]
            dp[i] %= M

        return dp[-1]       
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

#### 96. Unique Binary Search Trees
```
class Solution:
    def numTrees(self, n: 'int') -> 'int':
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        for i in range(2, n + 1):
            for j in range(i):
                dp[i] += dp[j] * dp[i - j - 1]
        return dp[n]
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
                # 当前s3里的字符是从s1里取的还是从s2里取的
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
        return self._is_valid(root, -infinity, infinity)
    
    def _is_valid(self, root, min_value, max_value):
        if root is None:
            return True
        
        if not min_value < root.val < max_value:
            return False
        
        # root的左子树的最大值不能大于root的值
        # root的右子树的最小值不能小于root的值
        return self._is_valid(root.left, min_value, root.val) and \
            self._is_valid(root.right, root.val, max_value)
```

#### 99. Recover Binary Search Tree
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
        # 这道题是说给定一个前序和中序的list
        # 构建BST
        # 思路就是通过前序找到根节点
        # 然后在中序中找到分界点
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
        # 根据中序和后序的list
        # 构建BST
        # 其实思路和105一样，通过后序的最后一个节点确定root
        # 然后将中序遍历的数组split
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

        # 这道题和核心是通过快慢指针找到中点
        prev = mid = fast = head
        while fast and fast.next:
            prev = mid
            mid = mid.next
            fast = fast.next.next

        # 要先断开，再将断开的链表传入递归函数
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
    def pathSum(self, root, target):
        """
        :type root: TreeNode
        :type target: int
        :rtype: List[List[int]]
        """
        # 这道题问的是所有从根节点到叶子节点的和为target的路径
        res = []
        self._helper(root, target, [], res)
        return res
    
    def _helper(self, node, target, curr, res):
        if not node:
            return
        
        curr.append(node.val)
        
        if target == node.val and not node.left and not node.right:
            res.append(curr[:])
        
        # DFS通常递归的就是多种选择
        # 比如这道题，有向左也有向右的选择
        self._helper(node.left, target - node.val, curr, res)
        self._helper(node.right, target - node.val, curr, res)
        
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
        # 这道题问的是s中有多少个子序列能够组成t
        ns, nt = len(s), len(t)
        dp = [[0] * (ns + 1) for _ in range(nt + 1)]
        for i in range(ns + 1):
            # 遍历第一行
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
                # s中前j个字符的子序列能凑成t的个数
                # 至少为前j - 1个字符的子序列能够凑成t中前i个字符的个数
                dp[i][j] = dp[i][j - 1]
                # 如果此时有两个字符相等
                # 说明当前结果还可以加上不用当前字符的结果
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
        # 这个条件表示当前层还有下一层
        # 在每层循环里处理的是下一层
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
                # 同一层平移
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
            # 注意最后一天可以直接丢掉
            # 不可能最后一天买
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

#### 123. Best Time to Buy and Sell Stock III
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
    ## 买入+卖出
    def maxProfit(self, prices):
        # write your code here
        if not prices:
            return 0
        n = len(prices)
        global_max = [[0] * 3 for _ in range(n)]
        local_max = [[0] * 3 for _ in range(n)]
        for i in range(1, n):
            diff = prices[i] - prices[i - 1]
            for j in (1, 2):
                # local_max就是卖出是当前天的情况
                # 比较的第一个量是全局到i-1天进行j-1次交易
                # 然后加上今天的交易，如果今天是赚钱的话（也就是前面只要j-1次交易，j - 1次交易中的最后一次交易的卖出时间是当前天）
                # 第二个量则是取local第i-1天j次交易，然后加上今天的差值
                # 这里因为local[i-1][j]比如包含第i-1天卖出的交易，所以现在变成第i天卖出，并不会增加交易次数
                # 而且这里无论diff是不是大于0都一定要加上，因为否则就不满足local[i][j]必须在最后一天卖出的条件了）
                local_max[i][j] = max(
                    global_max[i - 1][j - 1] + max(diff, 0),
                    local_max[i - 1][j] + diff,
                )
                # 当前局部最好的，和过往全局最好的中大的那个
                # 因为最后一次卖出如果包含当前天一定在局部最好的里面，否则一定在过往全局最优的里面
                global_max[i][j] = max(
                    local_max[i][j],
                    global_max[i - 1][j],
                )
        return global_max[-1][2]

        # 更好更容易理解的答案！！！
        # if not prices:
        #     return 0
        
        # # 核心思想
        # # 两次交易必定是错开的
        # # 所以某天我们要看两个值：
        # # 1是某天卖出之前买入的股票
        # # 2是某天买入之后卖出的股票
        # # 二者之和的最大值就是答案!!!
        # n = len(prices)
        # prev = [0] * n
        # future = [0] * n
        
        # min_v = prices[0]
        # for i in range(1, n):
        #     min_v = min(min_v, prices[i])
        #     prev[i] = max(prev[i - 1], prices[i] - min_v)
        
        # max_v = prices[-1]
        # for i in range(n - 2, -1, -1):
        #     max_v = max(max_v, prices[i + 1])
        #     future[i] = max(future[i + 1], max_v - prices[i])
        
        # res = 0
        # for i in range(n):
        #     res = max(res, prev[i] + future[i])
        
        # return res
```

#### 124. Binary Tree Maximum Path Sum
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
    
    # _helper函数返回的是以当前node为根的path sum
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

#### 126. Word Ladder II
```
from collections import deque
from collections import defaultdict

class Solution:
    def findLadders(self, start, end, word_list):
        """
        :type start: str
        :type end: str
        :type word_list: List[str]
        :rtype: List[List[str]]
        """
        # 这道题是说给定一个起始词和一个目标词，从word_list中找到所有的最短路径
        word_set = set(word_list)
        word_set.add(start)
        
        remove_one_char = self._remove_one_char_mapping(word_set)
        
        # 更新所有词到end这个词的更新距离(指的是需要更新几步)
        # 最终start这个词也肯定应该在dist里
        # 如果不在，说明无法变换，直接返回空即可
        # distance里面的定义是每个词（包括start）
        # 到end这个词的距离
        dist = {}
        self._bfs(end, dist, remove_one_char)
        
        if start not in dist:
            return []
        
        res = []
        self._dfs(start, end, dist, remove_one_char, [start], res)
        return res

    # bfs做的事情是更新每个在word_set里的词到begin_word的距离
    # 存到distance字典中
    def _bfs(self, begin_word, distance, remove_one_char):
        # 自身到自身的距离肯定是0
        # 这里的begin_word实际上是终点词
        # 我们将终点词当成begin word
        # 相当于反向遍历bfs
        distance[begin_word] = 0
        queue = deque()
        queue.append(begin_word)
        while queue:
            word = queue.popleft()
            for next_word in self._get_next_word(word, remove_one_char):
                # 核心：如果不在，说明当前找到了一个最短距离，更新即可
                # 如果在，说明之前找到过，由于我们是bfs，所以之前更新的距离一定是最短距离
                # 那就不用再更新了
                if next_word not in distance:
                    distance[next_word] = distance[word] + 1
                    queue.append(next_word)
    
    # 这道题实际上是反向dfs, 即从后（最终的end word）去递推start word
    def _dfs(self, start, end, dist, remove_one_char, curr_path, res):
        if start == end:
            res.append(curr_path[:])
            return
        for word in self._get_next_word(start, remove_one_char):
            # 剪枝，如果当前遍历的next word到最终的end word的距离不是只小于1
            # 直接不用管
            if dist[word] != dist[start] - 1:
                continue
            curr_path.append(word)
            self._dfs(word, end, dist, remove_one_char, curr_path, res)
            curr_path.pop()

    # 任意一个word替换掉一个字符都对应自身
    # 相当于变换的距离为1
    # 将某个词添加到这个词变化一个字母以后的key中去
    def _remove_one_char_mapping(self, word_set):
        remove_one_char = defaultdict(set)
        for word in word_set:
            for i in range(len(word)):
                one_ch_short_pattern = word[:i] + '%' + word[i + 1:]
                remove_one_char[one_ch_short_pattern].add(word)
        return remove_one_char
    
    # 输入word对word里每一个位置的字符用'%'代替
    # 看能对应remove_one_char里哪些词
    # 把这些词全部存起来返回成一个set
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
    def ladderLength(self, begin_word, end_word, word_list):
        """
        :type begin_word: str
        :type end_word: str
        :type word_list: List[str]
        :rtype: int
        """
        # 跟II不一样在于问需要变换几步
        word_set = set(word_list)
        n = len(begin_word)
        
        queue = [(begin_word, 1)]
        while queue:
            curr, count = queue.pop(0)
            if curr == end_word:
                return count
            for i in range(n):
                for ch in 'abcdefghijklmnopqrtsuvwxyz':
                    new_word = curr[:i] + ch + curr[i + 1:]
                    if new_word in word_set:
                        word_set.remove(new_word)
                        queue.append((new_word, count + 1))
        
        return 0
```

#### 128. Longest Consecutive Sequence
```
class Solution:
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        
        num_hash = set(nums)
        res = 0
        for num in nums:
            # 如果num不在当前的num_hash里
            # 就说明这个num一定之前是被访问过的(所以才会被remove掉)
            # 注意：num_hash里只会有nums里的数字，所以remove就意味着
            # 不再访问了
            if num not in num_hash:
                continue
            
            # 这道题的核心就是remove操作
            # remove就意味着之前被访问过 可以删除了
            # 而只要之前被访问过，这个num就已经属于某个序列
            # 不需要再查了
            num_hash.remove(num)
            
            # 下面的asc和desc的循环就是来看以当前nums这个词为中点
            # 能向上或者向下扩展多少
            # 实际上就有点像贪心的思想
            # 在每次局部的循环中尽量扩展
            asc = num + 1
            while asc in num_hash:
                num_hash.remove(asc)
                asc += 1
            
            desc = num - 1
            while desc in num_hash:
                num_hash.remove(desc)
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
        return self._dfs(root, curr_sum=0)
    
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
        # 这道题是要将board中所有被X包围的O变成X
        # 在边上的O是不会被变的，所以要特殊处理
        if not board or not board[0]:
            return
        
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if (i in (0, m - 1) or j in (0, n - 1)) and board[i][j] == 'O':
                    # 从边界开始DFS
                    # 这个DFS的作用就是将所有和边界O相连的区域用#标记出来
                    # 因为这部分是不需要变成X的
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

#### 132. Palindrome Partitioning II
```
class Solution:
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
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
        return self._helper(node, dict())
    
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
            for each_node in node.neighbors:
                new_node.neighbors.append(self._helper(each_node, node_map))
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
        # 第一遍从左向右遍历，如果右边的小盆友的等级高，就给右边小朋友加一个糖果，这样保证了一个方向上高等级的糖果多。
        # 然后再从右向左遍历一遍，如果相邻两个左边的等级高，而左边的糖果又少的话，则左边糖果数为右边糖果数加一。
        # 最后再把所有小盆友的糖果数都加起来返回即可。
        # 实际上这道题是两次贪心，分别从左到右再从右到左，这个思路挺重要
        n = len(ratings)
        res = 0
        nums = [1] * n

        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                nums[i] = nums[i - 1] + 1
        
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                # 反过来遍历时
                # 如果左边的rating高，而左边的糖果一定最少是右边的糖果再加1
                # 下面的max相当于确定了左边的最小值
                nums[i] = max(nums[i], nums[i + 1] + 1)
        
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
        # 当5第一次出现的时候，b=5, a=0,  b记录这个数字
        # 当5第二次出现的时候，b=0, a=5， a记录了这个数字
        # 当5第三次出现的时候，b=0, a=0， 都清空了，可以去处理其他数字了
        # 所以，如果有某个数字出现了1次，就存在b中，出现了两次，就存在a中，所以返回 a|b
        # 异或是两个位置上相同为0，不相同才为1
        # 特点:
        # X ^ X = 0
        # X ^ 0 = X
        # X ^ Y ^ Y = X ^ 0 = X
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
        
        ## 核心之一：先生成新Node，放在原来Node的后面
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
    def wordBreak(self, s, word_dict):
        """
        :type s: str
        :type word_dict: List[str]
        :rtype: bool
        """
        # 这题问的是能否用word_dict里的词来表示s(其中word_dict里的词可以重复使用)
        # 实际上就是背包问题
        if not s:
            # 这个边界不是很make sense
            return True

        if not word_dict:
            return False

        ## 思路：分隔板问题
        ## dp[i]表示字符串前i个字符是否可以用word_dict里的词表示
        ## j遍历所有0到i - 1的情况，如果dp[j]已经为True而且字符串j + 1到i也是word_dict中的某个词
        ## 则说明dp[i]可以用word_dict里的词表示了
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(1, n + 1):
            for j in range(i + 1):
                if dp[j] and s[j:i] in word_dict:
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

#### 140. Word Break II
```
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        # 这道题跟I不一样的地方在于要求出所有的组成
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
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        # 这道题是判断链表里有没有环
        if not head:
            return False

        slow = fast = head
        # 两种情况下确定有没有环
        # 要么fast指针到头了，要么fast指针追上slow指针了
        while True:
            if fast.next:
                slow = slow.next
                fast = fast.next.next
                if not fast:
                    return False
                elif fast == slow:
                    return True
            else:
                return False
```

#### 142. Linked List Cycle II
```
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            # 显式返回一个空
            return None
        
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                break
        
        if slow is not fast:
            return

        new_slow = head
        while new_slow is not slow:
            new_slow = new_slow.next
            slow = slow.next
        
        return new_slow
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
        # 说明链表长度应该大于等于3
        # 小于3直接return
        if not head or not head.next or not head.next.next:
            return
            
        slow = fast = head
        # 快慢指针
        # 如果链表是奇数
        # 则最后fast指针指向的是倒数第一个位置即最后一个位置(被fast.next终止)
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

#### 145. Binary Tree Postorder Traversal
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

#### 146. LRU Cache
```
from collections import OrderedDict

class LRUCache(object):

    # 这道题核心就是用python的OrderedDict
    # 每一次操作都先pop一遍，再添加一次
    # 这样就保证了最后被访问的（包括插入，查询）都会被放到最后一位
    # 换句话说第一位就是最久没被访问过的，可以被pop掉
    # OrderedDict是按照插入顺序保存的
    # 支持的API：
    # pop(key)
    # popitem() 这里和普通的dict不同，普通的dict的popitem()不支持参数，
    # OrderedDict的popitem(last = False)默认last = True,即默认从插入的最后一位pop
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
                self.cache.popitem(last=False)
        self.cache[key] = value

# 更标准的解法（面试官可能想考察双向链表）
class Node:
    def __init__(self, k, v):
        self.key = k
        self.val = v
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dict = dict()
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.dict:
            node = self.dict[key]
            self._remove(node)
            self._add(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dict:
            self._remove(self.dict[key])
        node = Node(key, value)
        self._add(node)
        self.dict[key] = node
        if len(self.dict) > self.capacity:
            temp = self.head.next
            self._remove(temp)
            self.dict.pop(temp.key)
    
    def _remove(self, node):
        _prev = node.prev
        _next = node.next
        _prev.next = _next
        _next.prev = _prev
    
    def _add(self, node):
        _prev = self.tail.prev
        _prev.next = node
        self.tail.prev = node
        node.prev = _prev
        node.next = self.tail


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
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

#### 149. Max Points on a Line
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
        # 这道题问的是最多有多少个点共线
        # 基本就是两两比较
        # 先fix一个点，然后去找跟这个点共线的一共有多少个点
        if not points:
            return 0

        n = len(points)
        res = 0
        for i in range(n):
            slopes_counts = defaultdict(int)
            same_points = 0
            for j in range(i + 1, n):
                if self._is_equal(points[i], points[j]):
                    same_points += 1
                else:
                    slope = self._get_slope(points[i], points[j])
                    slopes_counts[slope] += 1
            
            max_same_line_points = 0
            if slopes_counts:
                max_same_line_points = max(slopes_counts.values())
            res = max(res, max_same_line_points + same_points + 1)
        
        return res
    
    def _is_equal(self, p1, p2):
        return p1.x == p2.x and p1.y == p2.y
    
    def _get_slope(self, p1, p2):
        if p1.x == p2.x:
            return 2 ** 31 - 1
        
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        ## 单纯返回 dx / dy 会因为精度问题通过不了OJ
        ## 详情看
        ## http://www.cnblogs.com/grandyang/p/4579693.html
        gcd = self._gcd(dx, dy)
        return (dx // gcd, dy // gcd)

    def _gcd(self, a, b):
        if b == 0:
            return a
        
        return self._gcd(b, a % b)
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

        # 最终在stack里的一定只有一个string
        # 就是最终的结果
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
        # 这道题本质上是dp
        # pos[i]表示子数组[0, i]范围内并且一定包含nums[i]数字的最大子数组乘积
        # neg[i]表示子数组[0, i]范围内并且一定包含nums[i]数字的最小子数组乘积
        # 最后更新的res是全局，可以包括nums[i]也可以不包括nums[i]
        if not nums:
            return 0
        
        if len(nums) == 1:
            return nums[0]
        
        n = len(nums)
        # 这两个都是局部最优
        # 表示当遍历到i的时候，最大的pos和最小的neg是多少
        pos, neg = [0] * n, [0] * n
        
        if nums[0] > 0:
            pos[0] = nums[0]
        
        if nums[0] < 0:
            neg[0] = nums[0]
        
        ## 出了一个小bug：
        ## res注意在循环里是没有和pos[0]比较过的
        ## 所以要增加一次比较
        ## 因为最终答案可能就是第一个数字nums[0]
        res = -2 ** 31 if nums[0] < 0 else pos[0]
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
        # 这道题给定了没有重复元素
        # 而且数组长度大于等于3
        # 而且限定了肯定有转折点
        # 即不会出现单调递增的情况
        if len(nums) == 1 or nums[0] < nums[-1]:
            return nums[0]
        
        n = len(nums)
        start, end = 1, n - 2

        while start + 1 < end:
            mid = start + (end - start) // 2
            # 下面两个if都说明了找到了分界点
            # 因为如果在左右两段上肯定有nums[mid - 1] < nums[mid] < nums[mid + 1]
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

#### 154. Find Minimum in Rotated Sorted Array II
```
class Solution:
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 理论上这道题在前面应该加一个判断
        # nums[0] >= nums[-1]
        # 来确定nums到底是不是有折半的情况存在
        # 通过前面的判断以后，就可以在下面处理有折半的情况了
        # 这道题和153的区别就在于这道题可能有重复元素
        # 而153没有
        # 核心：用end比较确定折半方向
        if not nums:
            return 2 ** 31 - 1
        
        start, end = 0, len(nums) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            # 核心之一：用end
            if nums[mid] > nums[end]:
                start = mid + 1
            # 此时我们并不清楚mid是否是解
            # 所以不能轻易去掉mid
            # 但是我们确定的是mid+1以后的肯定不是解
            # 因为题目要求的是最小值
            elif nums[mid] < nums[end]:
                end = mid
            else:
                # 核心之二：可能有重复的数值
                end -= 1

        if nums[start] < nums[end]:
            return nums[start]
        
        return nums[end]
```

#### 155. Min Stack
```
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self._stack = []
        self._min = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self._stack.append(x)
        # 这里的>=就保证能多次push小数（比如多次push 0）
        # 都会存入到min中
        if not self._min or self._min[-1] >= x:
            self._min.append(x)

    def pop(self):
        """
        :rtype: void
        """
        val = self._stack.pop()
        if val == self._min[-1]:
            self._min.pop()

    def top(self):
        """
        :rtype: int
        """
        return self._stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self._min[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
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
        # 这道题是要干啥？
        # 相当于将旧root的最左孩子变成新root，
        # 旧root的最左孩子的兄弟右孩子变成新root的左孩子，
        # 旧root的最左孩子的祖先节点们变成新root的右孩子，
        # 这个root最终变成最右节点
        # 顺时针旋转180度
        #     1
        #    / \
        #   2   3
        #  / \
        # 4   5
        # 变成
        #   4
        #  / \
        # 5   2
        #    / \
        #   3   1 
        if not root or not root.left:
            return root
        # 先序遍历
        new_root = self.upsideDownBinaryTree(root.left)
        
        # 持续对root的左孩子操作
        # 下面是重点
        root.left.left = root.right
        root.left.right = root
        # 这个root最终变成最右节点
        root.left = root.right = None
        
        return new_root
        # 1. 对于一个parent来说，假如有right node，必须得有left node。
        # 而有left node，right node可以为空。而right node必须为叶子节点。所以该树
        # 每层至多有2个节点，并且2节点有共同的parent。
        # 2. 所以对于最底层来说，必有一个left node，而这个left node则为整个新树的根。
        # 3. 原树的根节点，变为了新树的最右节点。
        # 4. 对于子树1 2 3来说，需要在以2为根的子树2 4 5建立成新树4 5 2后，
        # 插入到新树的最右节点2下面。原树的根节点root为left child，原树root->right为新树的left node
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
        # 在当前函数域内设置一个大小为4个cache
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

#### 158. Read N Characters Given Read4 II - Call multiple times
```
# The read4 API is already defined for you.
# @param buf, a list of characters
# @return an integer
# def read4(buf):
    def __init__(self):
        self._buf4 = [None] * 4
        self._read_pos = 0
        self._write_pos = 0
    
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Maximum number of characters to read (int)
        :rtype: The number of characters read (int)
        """
        i = 0
        while i < n:
            # 这道题实际上是双指针的题
            # 追赶型双指针
            # 初始的时候read_pos是和write_pos重合的
            # 其实每次只要这两个位置重合
            # 就意味着buf4满了
            # write_pos意义是从read4中读到了多少个字符
            # 而read_pos是服务于往结果的buf里存数据的
            # 表示从当前的buf4里的哪个位置开始读
            # 换句话说，if里每次读4个字符
            # 而外层的while循环其实每次是往buf里存一个字符
            if self._read_pos == self._write_pos:
                self._read_pos = 0
                self._write_pos = read4(self.buf4)
                if self._write_pos == 0:
                    return i
            buf[i] = self._buf4[self._read_pos]
            self._read_pos += 1
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
        # 追赶型双指针
        # 注意这里不要用defaultdict了
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
                    # 核心之一
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
    def getIntersectionNode(self, head1, head2):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
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
            steps = len1 - len2
            while steps:
                head1 = head1.next
                steps -= 1
        elif len1 < len2:
            steps = len2 - len1
            while steps:
                head2 = head2.next
                steps -= 1
        
        while head1:
            if head1 is head2:
                return head1
            head1 = head1.next
            head2 = head2.next
        
        return None
    
    # 下面的是LC高票，思路实在太屌了
    # if not head1 or not head2:
    #     return
    
    # curr1, curr2 = head1, head2
    # while curr1 is not curr2:
    #     curr1 = curr1.next if curr1 else curr2
    #     curr2 = curr2.next if curr2 else curr1
    # return curr1
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
            # 当第一个s和t不相等的ch出现的时候
            # 肯定就是要返回的时候
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
        # 这道题是说数组有有很多的peak
        # 找出其中一个就好
        l, r = 0, len(nums) - 1

        # 这道题用l <= r就好
        while l <= r:
            mid = l + (r - l) // 2
            
            if (mid == 0 or nums[mid] > nums[mid - 1]) and \
                (mid == len(nums) - 1 or nums[mid] > nums[mid + 1]):
                return mid
            # 比较一端就好了
            elif mid > 0 and nums[mid] < nums[mid - 1]:
                r = mid - 1
            else:
                l = mid + 1
        
        return -1
```

#### 163. Missing Ranges
```
class Solution:
    def findMissingRanges(self, nums: 'List[int]', lower: 'int', upper: 'int') -> 'List[str]':
        res = []

        # 情况1：直接加上lower和upper这个区间
        if not nums:
            self._add_range(lower, upper, res)
            return res
        
        # 情况2：处理下lower到第一个数字
        self._add_range(lower, nums[0] - 1, res)
        
        # 情况3：一个一个处理nums里的间隔（前一个间隔与当前间隔）
        for i in range(1, len(nums)):
            self._add_range(nums[i - 1] + 1, nums[i] - 1, res)
        
        # 情况4：处理下最后一个数字到upper
        self._add_range(nums[-1] + 1, upper, res)
    
        return res
    
    def _add_range(self, lower, upper, res):
        if lower > upper:
            return
        
        if lower == upper:
            res.append(str(lower))
            return
        
        res.append('{}->{}'.format(lower, upper))
```

#### 165. Compare Version Numbers
```
class Solution:
    def compareVersion(self, version1: 'str', version2: 'str') -> 'int':
        v1 = [int(v) for v in version1.split('.')]
        v2 = [int(v) for v in version2.split('.')]  
        
        for i in range(max(len(v1), len(v2))):
            sub_v1 = v1[i] if i < len(v1) else 0
            sub_v2 = v2[i] if i < len(v2) else 0
            if sub_v1 > sub_v2:
                return 1
            elif sub_v1 < sub_v2:
                return -1
        
        return 0
```

#### 166. Fraction to Recurring Decimal
```
class Solution:
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        # 典型模拟法问题
        res = ''
        if numerator // denominator < 0:
            res += '-'
        if numerator % denominator == 0:
            return str(numerator // denominator)
        
        numerator = abs(numerator)
        denominator = abs(denominator)
        res += str(numerator // denominator)
        res += '.'
        
        numerator %= denominator
        i = len(res)
        mapping = dict()
        
        while numerator != 0:
            if numerator not in mapping:
                # 实际上就是记录下上次出现的分子位置
                # 注意不会出现0.6767676767的情况
                # 只可能出现0.66666...
                mapping[numerator] = i
            else:
                i = mapping[numerator]
                res = res[:i] + '(' + res[i:] + ')'
                return res
            numerator = numerator * 10
            res += str(numerator // denominator)
            numerator %= denominator
            i += 1
        
        return res
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
                # 这道题比较坑
                # 要返回的是数组里“第几个”数字，所以要把结果+1
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
        self._data = defaultdict(int) 

    def add(self, number):
        """
        Add the number to an internal data structure..
        :type number: int
        :rtype: void
        """
        self._data[number] += 1
        
    def find(self, value):
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        :type value: int
        :rtype: bool
        """
        for curr in self._data:
            # 支持a + a == 2a的情况
            if curr * 2 == value and self._data[curr] > 1:
                return True
            
            if curr * 2 != value and value - curr in self._data:
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
        self._stack = []
        self._push_left(root)

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self._stack) > 0
        
    def next(self):
        """
        :rtype: int
        """
        top = self._stack.pop()
        # 核心：类似中序遍历的思想
        # 在每次访问的时候，同时要将该节点的右孩子push到栈中
        # 注意这里要调循环的push_left
        # 不能直接append
        self._push_left(top.right)
        return top.val
    
    def _push_left(self, node):
        while node:
            self._stack.append(node)
            node = node.left
```

#### 174. Dungeon Game
```
class Solution:
    def calculateMinimumHP(self, dungeon: 'List[List[int]]') -> 'int':
        if not dungeon or not dungeon[0]:
            return 0
        
        m, n = len(dungeon), len(dungeon[0])
        # 这道题的精髓就是从后往前遍历
        # dp[i][j]表示要到达目的地i j点要最少多少血量
        dp = [[2 ** 31 - 1] * (n + 1) for _ in range(m + 1)]
        dp[m][n - 1] = 1
        dp[m - 1][n] = 1
        
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                dp[i][j] = max(1, min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j])
        
        return dp[0][0]
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
    # 即比较器里输入参数里的第一个认为是大的(-1)
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
        # 这道题是说将s中所有出现次数大于10的子串拿出来
        if not s or len(s) < 10:
            return []
        ## 基本思路就是遍历所有的10个字符长度的子串，将其添加到mapping里
        ## 如果此时该key对应在mapping里的值大于1，说明出现了重复，将其添加到一个结果set里
        mapping = defaultdict(int)
        res = set()
        for i in range(len(s) - 9):
            mapping[s[i:i + 10]] += 1
            if mapping[s[i:i + 10]] > 1:
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

        # 比如10天 6次交易
        # 而且每天只能买入或者卖出一次
        # 实际上就是在10天里
        # 此时说明没有限制（因为每天只能一次买入或者一次卖出）
        # 这种情况下就是股票1
        if k > n // 2:
            return self._quick_solve(prices)
        
        # dp[j]表示完成j次交易时的最大收益
        # dp[j] = max(dp[j], dp[j - 1] + prices[i] * [1, -1][j % 2])
        dp = [-2 ** 31] * (2 * k + 1)
        dp[0] = 0
        for i in range(n):
            # 题目给出了K次交易
            # 则一共有2 * K次操作
            # 由于每天最多只能有一次操作
            # 所以这里的j上限肯定是min(2 * K, i + 1)
            for j in range(min(2 * k, i + 1), 0, -1):
                # j的定义是第几次操作（可以是买入操作也可以是卖出操作）
                # j为偶数天卖出（赚钱），奇数天买入（少钱）
                if j % 2 == 0:
                    dp[j] = max(dp[j], dp[j - 1] + prices[i])
                else:
                    dp[j] = max(dp[j], dp[j - 1] - prices[i])
        
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
        # 注意k是可能大于nums的长度的
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
        # 就是要算n中有多少个1
        res = 0
        for i in range(32):
            # 说明此时在n的i位置上发现了一个1
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
            # 此时队列里的最右边的元素就是right view里的结果
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
        # 上下左右4个方向说明有4种选择
        # 递归下去就好
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
        # 这道题问的是能不能修完全部课程
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
        # 这道题问的是nums中和大于s的最短子序列的长度是多少
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
        :type num_courses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        edges = defaultdict(list)
        indegrees = [0] * num_courses
        
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
        if len(res) != num_courses:
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

#### 212. Word Search II
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
            index = ord(ch) - ord('a')
            if not curr.children[index]:
                curr.children[index] = Node()
            curr = curr.children[index]
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
                # 注意由于在dfs里有回溯
                # 所以这里每次的visited矩阵都是被还原成全False的矩阵
                self._dfs(board, trie.root, res, visited, i, j)
        
        return list(res)
    
    # 递归定义：从棋盘的i,j点开始走，将能找到的词全部加到results中
    def _dfs(self, board, root, results, visited, i, j):
        visited[i][j] = True

        # 递归核心：当前遍历的棋盘上的点是否在当前root的孩子节点中
        # 这决定了是否需要剪枝
        index = ord(board[i][j]) - ord('a')
        if root.children[index]:
            root = root.children[index]
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

#### 213. House Robber II
```
class Solution:
    def rob(self, nums: 'List[int]') -> 'int':
        # 核心思路还是差不多
        # 只不过这里用两个dp数组
        # 第一个数组表示抢第一家（即没有可能最后一家）
        # 第二根数组表示不抢第一家（即有可能抢最后一家）
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums)
        
        rob_first = [0] * (n - 1)
        rob_no_first = [0] * (n - 1)
        
        rob_first[0] = nums[0]
        rob_first[1] = max(nums[0], nums[1])
        # 此时不会去考虑抢最后一家(n - 1)
        for i in range(2, n - 1):
            rob_first[i] = max(
                nums[i] + rob_first[i - 2],
                rob_first[i - 1],
            )
        
        rob_no_first[0] = nums[1]
        rob_no_first[1] = max(nums[1], nums[2])
        # 此时可以考虑去抢最后一家(n - 1)
        for i in range(3, n):
            rob_no_first[i - 1] = max(
                nums[i] + rob_no_first[i - 3],
                rob_no_first[i - 2],
            )
        
        return max(
            max(rob_first),
            max(rob_no_first),
        )
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
                # 此时rev_s[i:]就是可以往s前面加的字符串
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
        # 必背经典模板题目！！！
        left = 0
        right = len(nums) - 1
        
        while True:
            candidate_pos = self._partition(nums, left, right)
            # 因为candidate_pos是0-index，而题目问的k是1-index
            if candidate_pos + 1 == k:
                return nums[candidate_pos]
            elif candidate_pos + 1 > k:
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
        #   3.1 如果是start，二话不说加入堆，并判断当前堆顶(curr)是否前一次的堆顶元素(pre)一样
        #       如果不一样，说明这次操作堆顶发生了变化，加入到res中，并更新pre
        #   3.2 如果是end，说明需要从堆中pop掉
        
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
                # 桶中只保留index没有过期的桶号
                # TODO: 还不是很清晰
                del d[nums[i - k] // w]
        return False
```

#### 221. Maximal Square
```
class Solution:
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0
        
        m, n = len(matrix), len(matrix[0])
        # dp[i][j]定义：以i j为右下角的正方形的最大边长
        # 审题！matrix里面的是str而不是int
        dp = [[0] * n for _ in range(m)]
        res = 0
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    if matrix[i][j] == '1':
                        dp[i][j] = 1
                elif matrix[i][j] == '1':
                    dp[i][j] = min(
                        dp[i - 1][j - 1],
                        dp[i][j - 1],
                        dp[i - 1][j],
                    ) + 1
                res = max(res, dp[i][j])
        
        return res * res
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
        # res含义是当前的值
        # curr含义是当前在+-或者()之间的值
        res = curr = 0
        stack = []
        sign = 1
        
        for ch in s:
            if '0' <= ch <= '9':
                curr = 10 * curr + int(ch)
            # 核心一直：当遇到+-号的时候
            # 不是执行操作
            # 而是将当成正负号使用
            # 赋给后面的curr
            elif ch in '+-':
                res += sign * curr
                curr = 0
                sign = 1 if ch == '+' else -1
            elif ch == '(':
                # 注意是将sign和res同时入栈！！！
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
    
        # 先保存下之前的left和right
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

#### 228. Summary Ranges
```
class Solution:
    def summaryRanges(self, nums: 'List[int]') -> 'List[str]':
        n = len(nums)
        i = 0
        
        res = []
        while i < n:
            j = 1
            while i + j < n and nums[i + j] - nums[i] == j:
                j += 1
            if j == 1:
                res.append(str(nums[i]))
            else:
                res.append('{}->{}'.format(nums[i], nums[i + j - 1]))
            i += j
        
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
        # 典型二分法
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
        # 此时的slow.next已经是翻转后的右边链表的表头了
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

#### 239. Sliding Window Maximum
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
            if i >= k - 1:
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
        # 这道题和I一模一样
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
        # 这道题是说在words这个list中，找到word1和word2出现的index最短距离
        # http://www.cnblogs.com/grandyang/p/5187041.html
        # 我们用两个变量p1,p2初始化为-1
        # 然后我们遍历数组，遇到单词1，就将其位置存在p1里
        # 若遇到单词2，就将其位置存在p2里
        # 如果此时p1, p2都不为-1了，那么我们更新结果
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

            # 此时说明找到了p1和p2都不再是-1
            # 指向了word1和word2
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
```

#### 245. Shortest Word Distance III
```
class Solution(object):
    def shortestWordDistance(self, words, word1, word2):
        """
        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        p1 = p2 = -1
        res = 2 ** 31 - 1

        for i in range(len(words)):
            temp = p1
            if words[i] == word1:
                p1 = i
            if words[i] == word2:
                p2 = i
            if p1 != -1 and p2 != -1:
                # 最后一个条件temp != p1实际上隐含着说本次循环中
                # p1被更新了。因为在if条件中同时满足了word1 == word2
                # 就是说此时p2也是被更新了
                if word1 == word2 and temp != -1 and temp != p1:
                    res = min(res, abs(temp - p1))
                # 此时就是常规每轮比较一下
                elif p1 != p2:
                    res = min(res, abs(p1 - p2))
        
        return res
```

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
            # 当inner_width == outer_width的时候
            # 就是主调函数里一开始dfs(n, n)的情况
            # 说明此时是最外层
            # 是不能左右加0的
            if inner_width != outer_width:
                res.append('0' + each + '0')
            res.append('1' + each + '1')
            res.append('6' + each + '9')
            res.append('9' + each + '6')
            res.append('8' + each + '8')
        
        return res
```

#### 248. Strobogrammatic Number III
```
class Solution:
    def strobogrammaticInRange(self, low, high):
        """
        :type low: str
        :type high: str
        :rtype: int
        """
        # 这道题问的是在low和high之间有多少个strbogrammatic的number
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

#### 249. Group Shifted Strings
```
from collections import defaultdict

class Solution:
    def groupStrings(self, strings: 'List[str]') -> 'List[List[str]]':
        mapping = defaultdict(list)
        for each in strings:
            if len(each) == 1:
                mapping[(-1, )].append(each)
            else:
                orders = []
                for i in range(1, len(each)):
                    orders.append((ord(each[i]) - ord(each[i - 1]) + 26) % 26)
                mapping[tuple(orders)].append(each)
        
        res = []
        for _, words in mapping.items():
            res.append(words)
        
        return res
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
        res = self._data[self._row][self._col]
        self._col += 1
        return res

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
        # 这道题问的是一个人能不能参加完全部的会议
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
        # 这道题问的是需要的最少会议房间是多少
        # 实际上就是在找有多少个重叠
        if not intervals:
            return 0
        
        # 这道题核心需要一个有顺序的hash表(按照时间点从小到大排序)
        # 里面需要存每个时间点上的room count
        rooms_at_time = defaultdict(int)
        for interval in intervals:
            rooms_at_time[interval.start] += 1
            rooms_at_time[interval.end] -= 1
        
        rooms_list = []
        for time_point, counts in rooms_at_time.items():
            rooms_list.append((time_point, counts))
        rooms_list.sort()
        
        # local_res的含义是在遍历时候记录当前需要的room数目
        # 然后每次都去更新全局最大的global_res
        global_res = local_res = 0
        for _, counts in rooms_list:
            local_res += counts
            global_res = max(global_res, local_res)
        
        return global_res
```

#### 254. Factor Combinations
```
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
        
        # 这道题单纯这么做在Lintcode会MLE/TLE
        # range的上限不用是n这么大
        # 可以是平方根
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
        # 这道题是要求返回从根节点到叶子节点的路径
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
        # 因为写了string的拼接，所以不需要回溯了
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
        nums中有多少个三元组合，sum和小于target
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
        # 这道题是说数组中有两个数字只出现了一次
        # 其他的数字都出现了两次
        # 这道题有几个点要注意：
        # 异或的特点：（1）a ^ 0 = a, (2) a ^ a = 0
        # 异或就是说不一样的bit就为1，一样的bit就为0
        # diff &= -diff得到的是一个2的阶乘（二进制00001000只有一个1）
        # 很好的一个技巧：一个数与上它自身的相反数，就能得到这个数字最后一位的二进制1的位置
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
            # diff这里可以理解为两个目标数字不一样的地方的最后一个1的位置
            # 利用这个位置来分流nums
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
                # 因为这道题是无向图
                graph[each].remove(curr_point)
        
        return len(visited) == n
```

#### 265. Paint House II
```
class Solution:
    
    _INT_MAX = 2 ** 31 - 1
    
    def minCostII(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        if not costs or not costs[0]:
            return 0
        
        num_of_houses, num_of_colors = len(costs), len(costs[0])
        dp = [[self._INT_MAX] * num_of_colors for _ in range(num_of_houses)]
        dp[0] = costs[0][:]
        
        for i in range(1, num_of_houses):
            for j in range(num_of_colors):
                dp[i][j] = costs[i][j] + \
                    min(dp[i - 1][k] for k in range(num_of_colors) if k != j)
        
        return min(dp[-1])
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

#### 269. Alien Dictionary
```
from collections import defaultdict
from collections import deque

class Solution:
    def alienOrder(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        # 这道题是说words里的单词是已经排过序的
        # 求输出字母的顺序
        if not words:
            return ''

        hash_map = defaultdict(set)
        hash_indegree = defaultdict(int)
        result = ''

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
                    # 这道题是按照顺序小的字母指向顺序大的字母来建图的
                    # 建的是有向图
                    # 发现一条有向边，这条边是之前的不一样
                    # 是从一个词的字母指向另一个词的字母了
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
        # 好题！注意体会
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

#### 271. Encode and Decode Strings
```
class Codec:

    def encode(self, strs):
        """Encodes a list of strings to a single string.
        
        :type strs: List[str]
        :rtype: str
        """
        res = ''
        for word in strs:
            res += str(len(word)) + '/' + word
        return res

    def decode(self, s):
        """Decodes a single string to a list of strings.
        
        :type s: str
        :rtype: List[str]
        """
        res = []
        i = 0
        # 为什么这么编解码是可以的？
        # 比如原始单词就是6/asdd
        # 之所以这么做可行，
        # 就是因为我们是从左往右走的
        # 优先碰到的"/"会被认为是人为附加的！！！
        while i < len(s):
            slash = s.index('/', i)
            length = int(s[i:slash])
            i = slash + length + 1
            res.append(s[slash + 1:i])
        return res

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))
```

#### 272. Closest Binary Search Tree Value II
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

#### 273. Integer to English Words
```
from collections import deque

class Solution:
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        # 这道题是要数字转化为文字
        # 基本思路就是分为四级：
        # 0到19
        # 20到90
        # 100
        # 一千，一百万，十亿
        level1 = ('Zero One Two Three Four Five Six Seven Eight Nine Ten' + \
                 ' Eleven Twelve Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen').split()
        level2 = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()
        level3 = 'Hundred'
        level4 = 'Thousand Million Billion'.split()
        
        words, digits = deque(), 0
        while num:
            # 注意：这道题是从小到大三位三位数字处理的
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

#### 274. H-Index
```
class Solution:
    def hIndex(self, citations: 'List[int]') -> 'int':
        citations.sort(reverse=True)
        # i从小到大，但是citations[i]从大到小
        # 交点就是h-index
        for i in range(len(citations)):
            if i >= citations[i]:
                return i
        return len(citations)
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
        # 名人是说在这n个人里（序号1到n）
        # 这个名人被其他所有人认识
        # 同时不认识其他所有人
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
        # 这道题是说输入一个未排序的数组
        # 原地生成num[0] <= nums[1] >= nums[2] <= nums[3]...的样子
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
        # 这道题是说有两个数组
        # 交替一个一个的返回
        # 基本思路就是说每次操作的时候
        # 从队列中popleft出来 操作 并在操作完成之后再次入队
        # 这样就完成了交替操作的效果
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

#### 282. Expression Add Operators
```
class Solution:
    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]
        """
        # 这道题是说输入的数字num string 123
        # 用每个0-9的数字来只用加减乘生成target
        if not num:
            return []

        res = []
        self._dfs(num, target, start_pos=0, last_str='', last_val=0, last_diff=0, res=res)
        return res
    
    def _dfs(self, num, target, start_pos, last_str, last_val, last_diff, res):
        if start_pos == len(num):
            if last_val == target:
                res.append(last_str)
            return

        for i in range(start_pos, len(num)):
            curr_str = num[start_pos:i + 1]
            if num[start_pos] == '0' and i > start_pos:
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
                    last_str=last_str + '*' + curr_str,
                    last_val=last_val - last_diff + last_diff * int(curr_str),
                    last_diff=last_diff * int(curr_str),
                    res=res,
                )
                self._dfs(
                    num, target, i + 1,
                    last_str=last_str + '+' + curr_str,
                    last_val=last_val + int(curr_str),
                    last_diff=nt(curr_str),
                    res=res,
                )
                self._dfs(
                    num, target, i + 1,
                    last_str=last_str + '-' + curr_str,
                    last_val=last_val - int(curr_str),
                    last_diff=-int(curr_str),
                    res=res,
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
        # 这道题是说将0移动的数组的后面
        available_pos_for_0 = -1
        # i指针始终在available_pos_for_0之后的
        # 不停的将i指针指向的非零数字移到available_pos_for_0上去
        for i in range(len(nums)):
            if nums[i] != 0:
                available_pos_for_0 += 1
                nums[i], nums[available_pos_for_0] = nums[available_pos_for_0], nums[i]
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
        # 这道题是要找p节点的后继节点
        # 实际上等价于寻找bst中第一个大于p值的节点
        res = None
        while root:
            if root.val > p.val:
                res = root
                root = root.left
            else:
                # 此时root的值比p小
                # 所以p的后继也一定出现在root的右子树上（因为后继的值一定比p大）
                root = root.right
        
        return res
```

#### 286. Walls and Gates
```
# # DFS 用来练习很好
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
#                     self._dfs(rooms, i, j, current_depth=0)
    
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
        # 这道题是说-1是墙
        # 0是门
        # INF是空房间
        # 将每个INF替换成它到最近的门的距离
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
                if not 0 <= x < len(rooms) or \
                    not 0 <= y < len(rooms[0]) or \
                        rooms[x][y] < rooms[curr_i][curr_j] + 1:
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
        # 好题！多多体会
        # 快慢指针题
        # 这道题假定了nums中一定有重复的点
        # 否则解法就会死循环
        # 一个长度为n+1的数组
        # 里面的数字的范围是1到n
        # 则必定有至少一个数字是出现重复的
        # 将这个数字求出来
        # 这道题可以排序，可以用set，可以二分，可以用找环的思想
        # 比如：1 -> 3 -> 2 -> 4 -> 2 （2, 4, 2）就形成了环 
        # 已知0肯定是不在list里的
        # 所以0可以用来做初始化
        slow = fast = 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        # 此时就找到了环中的某个点 slow就是(2, 4, 2)环中的某个点
        # 下面需要找到环里的重复值
        temp = 0
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

#### 288. Unique Word Abbreviation
```
class ValidWordAbbr:

    def __init__(self, dictionary: 'List[str]'):
        self._mapping = dict()
        for word in dictionary:
            shortened = self._shorten(word)
            if shortened in self._mapping:
                self._mapping[shortened].add(word)
            else:
                self._mapping[shortened] = set([word])
        
    def isUnique(self, word: 'str') -> 'bool':
        shortened = self._shorten(word)
        if shortened not in self._mapping:
            return True
        # python的set和list都支持直接比较
        if self._mapping[shortened] == set([word]):
            return True
        return False
    
    def _shorten(self, word):
        if len(word) <= 2:
            return word
        return word[0] + str(len(word) - 2) + word[-1]


# Your ValidWordAbbr object will be instantiated and called as such:
# obj = ValidWordAbbr(dictionary)
# param_1 = obj.isUnique(word)
```

#### 291. Word Pattern II
```
class Solution:
    def wordPatternMatch(self, pattern, string):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        return self._dfs(pattern, string, dict(), set())
    
    # 递归定义：每次以pattern的首字母为目标(并更新mapping和used)
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

        # 此时说明ch不在mapping里，而且没有用过
        # 为什么这道题要用mapping和used?
        # mapping实际上表示当前的映射
        # used就表示全局这个词找没找过 起到剪枝的作用
        for i in range(len(string)):
            word = string[:i + 1]
            if word in used:
                continue

            mapping[ch] = word
            used.add(word)

            if self._dfs(pattern[1:], string[i + 1:], mapping, used):
                return True

            # 回溯
            del mapping[ch]
            used.remove(word)

        return False
```

#### 295. Find Median from Data Stream
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
        # 同时保证最大堆的长度等于最小堆的长度或者+1(就是说最大堆的长度大于最小堆)
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

#### 296. Best Meeting Point
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

#### 297. Serialize and Deserialize Binary Tree
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
        return ' '.join(res)

    def _serialize(self, node, res):
        if node:
            res.append(str(node.val))
            self._serialize(node.left, res)
            self._serialize(node.right, res)
        else:
            res.append('#')

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

#### 298. Binary Tree Longest Consecutive Sequence
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def longestConsecutive(self, root: 'TreeNode') -> 'int':
        if not root:
            return 0
        # 审题！！！这道题只要求递增！！！
        return self._dfs(root, parent=None, res=0)
    
    # 递归函数的定义一般就是输入某些状态，让递归函数去根据这些状态递归下去
    # 这道题里需要定义的状态就是当前node，当前node的parent，以及当前的result
    def _dfs(self, node, parent, res):
        if not node:
            return 0
        
        if parent and node.val - 1 == parent.val:
            res += 1
        else:
            # 表示断开了，从头计数
            res = 1
        
        return max(
            res,
            self._dfs(node.left, node, res),
            self._dfs(node.right, node, res)
        )
```

#### 299. Bulls and Cows
```
class Solution:
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        # 这道题就是根据serect和猜的词
        # 输出1A3B这种提示
        # A是指多少个字符位置对
        # B是指多个个字符对但是位置不对
        # 假设都是数字，并且长度一致
        mapping = [0] * 256
        n = len(secret)
        bulls = cows = 0
        for i in range(n):
            if secret[i] == guess[i]:
                bulls += 1
            else:
                if mapping[ord(secret[i])] < 0:
                    cows += 1
                if mapping[ord(guess[i])] > 0:
                    cows += 1
                mapping[ord(secret[i])] += 1
                mapping[ord(guess[i])] -= 1
        
        return str(bulls) + 'A' + str(cows) + 'B'
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
                    # 这个1就是当前的nums[i]
                    dp[i] = max(dp[i], 1 + dp[j])
        
        return max(dp)
```

#### 301. Remove Invalid Parentheses
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
            if self._is_valid(curr):
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
            # 去掉一个括号的选择放入队列
            # 去掉两个括号的选择放入队列
            # 以此类推下去。。。
            for i in range(len(curr)):
                if curr[i] in ('(', ')'):
                    # 去掉一个括号
                    new_str = curr[:i] + curr[i + 1:]
                    if new_str not in visited:
                        visited.add(new_str)
                        queue.append(new_str)
        
        return res
    
    def _is_valid(self, s):
        # 判断括号合不合法的好写法！！！
        # 不合法的括号唯一的情况就是右括号数目多于了左括号
        count = 0
        for c in s:
            if c == '(':
                count += 1
            elif c == ')':
                count -= 1
            # 核心之一
            if count < 0:
                return False
        # 注意：最后要检查count是否为0的
        return count == 0
```

#### 302. Smallest Rectangle Enclosing Black Pixels
```
class Solution:
    def minArea(self, image, x, y):
        """
        :type image: List[List[str]]
        :type x: int
        :type y: int
        :rtype: int
        """
        # 这道题问的是最小的矩形能够包括所有的黑色像素
        if not image or not image[0]:
            return 0
        
        m, n = len(image), len(image[0])

        # 1. 检查上边界
        start, end = 0, x
        while start + 1 < end:
            mid = start + (end - start) // 2
            # 如果在mid这一行有black pixel
            # 说明mid到x都是有黑色像素的
            if self._check_row(mid, image):
                end = mid
            # 反之，说明0到mid都是没有黑色像素的
            else:
                start = mid
        top_level = start if self._check_row(start, image) else end
        
        # 2. 检查下边界
        start, end = x, m - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_row(mid, image):
                start = mid
            else:
                end = mid
        # 核心之一： 为什么这里要先查end？
        # 因为我们要找包括所有的黑色像素的矩形
        # 所以必须尽可能扩展范围
        # 就是说在找bottom边界时候end要优先于start
        # 同理在找right边界的时候end也要优先于start
        bottom_level = end if self._check_row(end, image) else start
        
        # 3. 检查左边界
        start, end = 0, y
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_column(mid, image):
                end = mid
            else:
                start = mid
        left_level = start if self._check_column(start, image) else end
        
        # 4. 检查右边界
        start, end = y, n - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if self._check_column(mid, image):
                start = mid
            else:
                end = mid
        right_level = end if self._check_column(end, image) else start
        
        return (right_level - left_level + 1) * (bottom_level - top_level + 1)
    
    def _check_row(self, row, image):
        if any(i == '1' for i in image[row]):
            return True
        return False
    
    def _check_column(self, col, image):
        if any(i[col] == '1' for i in image):
            return True
        return False
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
        # 注意索引 典型presum解法
        return self._sum[row2 + 1][col2 + 1] - \
        self._sum[row2 + 1][col1] - \
        self._sum[row1][col2 + 1] + \
        self._sum[row1][col1]

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)
```

#### 305. Number of Islands II
```
# 并查集必背好题
class Union:
    def __init__(self, n):
        self.father = [i for i in range(n)]
        self.count = 0
    
    def find(self, a):
        if self.father[a] == a:
            return self.father[a]
        # 这一行是核心
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
        # 就是x中二进制的最后一位
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

#### 310. Minimum Height Trees
```
class Solution:
    def findMinHeightTrees(self, n: 'int', edges: 'List[List[int]]') -> 'List[int]':
        # 开始的思路是遍历每个点，并以每个点为根求深度
        # 但是实际上比较巧妙的做法是遍历每个叶子
        # 并类似剥洋葱的方式将叶子删除掉
        # 最终剩下的1到2层的节点就是答案
        if n == 1:
            return [0]
        
        # 每个点跟它相连的点的集合
        graph = [set() for _ in range(n)]
        for i, j in edges:
            graph[i].add(j)
            graph[j].add(i)
        
        leaves = [i for i in range(n) if len(graph[i]) == 1]
        
        while n > 2:
            n -= len(leaves)
            new_leaves = []
            for i in leaves:
                # 不常见的操作
                # 从集合中随机删除一个
                j = graph[i].pop()
                graph[j].remove(i)
                # 核心之一：
                # 什么是叶子节点？
                # 树种的叶子节点就是跟它相连的边只有1条
                # 如果此时跟j相连的点只有一个
                # 说明j就是一个叶子节点
                if len(graph[j]) == 1:
                    new_leaves.append(j)
            leaves = new_leaves
        
        return leaves
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
        # 比如打了nums[i]这个气球
        # 可以得到nums[i - 1] * nums[i] * nums[i + 1]的分数
        # 求最多的分数
        n = len(nums)
        nums = [1] + nums + [1]
        # dp[i][j]定义：
        # nums中i到j气球全部打爆能有多少分数
        dp = [[0] * (n + 2) for _ in range(n + 2)]
        
        # i遍历的意义是当前剩下几个气球?
        # 剩余气球数在1到n之间
        for i in range(1, n + 1):
            # left是从下标1开始的一直到n - i + 1
            for left in range(1, n - i + 2):
                right = left + i - 1
                for k in range(left, right + 1):
                    dp[left][right] = max(
                        dp[left][right],
                        # 核心之一:
                        # 判断left到right之间所气球都打爆能有多少分数
                        # 实际上到最后最剩下一个气球k
                        # 我们遍历这个k
                        # 最终结果应该分三部分：
                        # 1. 最后这个气球k和left的left以及right的right的乘积
                        # 2. left到k-1之间气球全部打爆的分数
                        # 3. k+1到right之间全部气球打爆的分数
                        nums[left - 1] * nums[k] * nums[right + 1] + \
                            dp[left][k - 1] + \
                            dp[k + 1][right],
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

#### 315. Count of Smaller Numbers After Self
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
        # 需要把nums变成index, num的形式
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
            # 每次只在循环里pop出一个东西
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
        # 这道题是要移除重复字符，同时保证原来的相对位置
        # 同时要保证最终的结果是最小的字典序
        # 就是说尽量把ascii小的字符放在前面

        # 比如输入 "bcabc"
        # 在下面的循环中的输出是：
        # b
        # bc
        # a
        # ab
        # abc
        m = [0] * 256
        visited = [False] * 256
        # 注意这里的res存放的是ascii码
        res = [0]
        
        for ch in s:
            m[ord(ch)] += 1
        
        for ch in s:
            m[ord(ch)] -= 1
            if visited[ord(ch)]:
                continue
            # 说明后面还会出现res[-1]这个字符
            # 重点！！！
            # 核心就是这个ord(ch) < res[-1]
            # 说明此时出现了一个更小的字典序的字符
            # 而且当前res[-1]位置的是完全可以被pop掉的（因为后面还再出现）
            while ord(ch) < res[-1] and m[res[-1]] > 0:
                visited[res[-1]] = False
                res.pop()
            visited[ord(ch)] = True
            res.append(ord(ch))
        
        return ''.join(chr(i) for i in res[1:])
```

#### 317. Shortest Distance from All Buildings
```
from collections import deque

class Solution:
    
    _DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def shortestDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # 好题
        # 这道题是说找一个空地建楼
        # 这个楼到grid里所有的楼的距离之和最小(并且能到达所有的楼)
        # 求出这个最小距离
        # 0是空地，1是已经建好的楼，2是障碍物
        if not grid or not grid[0]:
            return 0

        m, n = len(grid), len(grid[0])
        res = 2 ** 31 - 1
        building_cnt = 0
        # 这里的distance就相当于一个结果矩阵
        # 存的是所有的0 grid到所有的楼的距离之和
        distance = [[0] * n for _ in range(m)]
        reached_buildings = [[0] * n for _ in range(m)]
        
        # 基本思路就是先定位到1
        # 然后从这个1出发更新周围所有的楼的距离(用+=)
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

#### 320. Generalized Abbreviation
```
class Solution:
    def generateAbbreviations(self, word: 'str') -> 'List[str]':
        res = []
        res.append('' if not word else str(len(word)))
            
        for i in range(len(word)):
            left_res = '' if i == 0 else str(i)            
            right_res_list = self.generateAbbreviations(word[i + 1:])
            for right_each in right_res_list:
                res.append(left_res + word[i] + right_each)
        
        return res
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
        # 这道题是可以无限用coins
        # 维护一个一维动态数组dp，其中dp[i]表示钱数为i时的最小硬币数的找零，递推式为：
        # dp[i] = min(dp[i], dp[i - coins[j]] + 1);
        
        # 感觉下面的解法比书影的解法更好理解
        # 注意初始化一定要用一个大值
        dp = [amount + 1 for i in range(amount + 1)]
        dp[0] = 0
        
        for i in range(1, amount + 1):
            for j in range(len(coins)):
                if i - coins[j] >= 0:
                    # 具体就是在说用还是不用当前硬币
                    # 后面的就是使用当前硬币
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1)
        
        return dp[-1] if dp[-1] <= amount else -1

        # 书影博客
        # 这家伙一向喜欢在遍历当前index时候去更新未来的状态
        # dp = [-1] * (amount + 1)
        # dp[0] = [0]
        # for x in range(amount):
        #     # 当遍历到当前dp[x]状态时候，应该已经考虑过了x左边所有的情况
        #     # 所以此时如果仍然dp[x] == -1，则当前状态一定不能凑齐找零钱数
        #     if dp[x] == -1:
        #         continue
        #     for c in coins:
        #         # 面值为c加到x上会越界 直接跳过
        #         if x + c > amount:
        #             continue
        #         # x + c一定在当前x的右边，
        #         # 所以可以用现在的状态dp[x]去更新未来
        #         # 第一个条件是说没有更新过
        #         # 第二个条件是说现在即使dp[x + c]之前被更新过
        #         # 但是此时dp[x] + 1可以使用更少个数的硬币
        #         # 所以用dp[x] + 1来更新dp[x + c]
        #         if dp[x + c] == -1 or dp[x + c] > dp[x] + 1:
        #             dp[x + c] = dp[x] + 1
        # return dp[amount]
```

#### 323. Number of Connected Components in an Undirected Graph
```
# 这道题需要多默写几遍
class Union:
    def __init__(self, n):
        self._father = [i for i in range(n)]
        self._count = n
    
    def _find(self, a):
        if a == self._father[a]:
            return a
        # 重点！！！
        self._father[a] = self._find(self._father[a])
        return self._father[a]
    
    def connect(self, a, b):
        root_a = self._find(a)
        root_b = self._find(b)
        if root_a != root_b:
            self._father[root_a] = self._father[root_b]
            self._count -= 1
    
    def query(self):
        return self._count

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

#### 324. Wiggle Sort II
```
class Solution:
    def wiggleSort(self, nums: 'List[int]') -> 'None':
        """
        Do not return anything, modify nums in-place instead.
        """
        temp = nums[:]
        temp.sort()
        
        # 思路很巧妙
        # 先把大的从后往前pop出来放到奇数位置
        # 再把小的pop出来放到偶数位置
        # 这样就保证了小-大-小-大...
        for i in range(1, len(nums), 2):
            nums[i] = temp.pop()
        
        for i in range(0, len(nums), 2):
            nums[i] = temp.pop()
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

#### 329. Longest Increasing Path in a Matrix
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
        # dp解法
        # if not matrix or not matrix[0]:
        #     return 0
        
        # res = 1
        # m, n = len(matrix), len(matrix[0])
        # # dp[i][j]表示从matrix的i j开始最大递增路径的长度
        # dp = [[0] * n for _ in range(m)]
        
        # for i in range(m):
        #     for j in range(n):
        #         if dp[i][j] > 0:
        #             continue
        #         queue = deque([(i, j)])
        #         length = 1
        #         while queue:
        #             length += 1
        #             q_len = len(queue)
        #             for _ in range(q_len):
        #                 ci, cj = queue.popleft()
        #                 for di, dj in self._DIRECTIONS:
        #                     newi, newj = ci + di, cj + dj
        #                     if 0 <= newi < m \
        #                         and 0 <= newj < n \
        #                         and matrix[newi][newj] > matrix[ci][cj] \
        #                         and dp[newi][newj] < length:
        #                         queue.append((newi, newj))
        #                         dp[newi][newj] = length
        #                         res = max(res, dp[newi][newj])
        
        # return res
                      
        if not matrix or not matrix[0]:
            return 0

        m, n = len(matrix), len(matrix[0])
        sequence = []
        for i in range(m):
            for j in range(n):
                sequence.append((matrix[i][j], i, j))
        sequence.sort()
        
        # 注意这里的定义
        # 指的是能走到这个点的个数(能走过来的值一定比这个点小)
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
        # 这道题说的是返回一个行程计划
        graph = defaultdict(list)
        for from_, to_ in tickets:
            graph[from_].append(to_)
        
        for each in graph:
            graph[each].sort(reverse=True)
        
        res = []
        self._dfs(graph, 'JFK', res)
        return res[::-1]
    
    # dfs定义：在图中，以from开始
    # 将所有未访问过的地点添加到res中
    def _dfs(self, graph, from_, res):
        # 典型递归
        # 从from出发我们有很多种新起点
        # 每一个新起点又是一次新的递归
        for _ in range(len(graph[from_])):
            # 跟visit数组的作用是一样的
            curr = graph[from_].pop()
            self._dfs(graph, curr, res)
        # while graph[from_]:
        #     curr = graph[from_].pop()
        #     self._dfs(graph, curr, res)
        # 本质上就是图的后序遍历 
        res.append(from_)
```

#### 334. Increasing Triplet Subsequence
```
class Solution:
    def increasingTriplet(self, nums: 'List[int]') -> 'bool':
        # 审题！！！这道题要求的是子序列
        # 应该想到使用两个全局变量的解法！！
        lowest = second_lowest = 2 ** 31 - 1
        for num in nums:
            if num <= lowest:
                lowest = num
            elif num <= second_lowest:
                second_lowest = num
            else:
                # lowest和second_lowest代表当前已经递增的两个出现过的数字
                # 同时lowest < second_lowest
                # 则如果出现了另外一个既大于lowest又大于second_lowest的数字
                # 说明就找到了一个递增的三元组
                return True
        
        return False
```

#### 336. Palindrome Pairs
```
class Solution:
    def palindromePairs(self, words):
        """
        :type words: List[str]
        :rtype: List[List[int]]
        """
        # 这道题是说从一个不重复的单词数组words中
        # 找出所有的inx对儿，每个inx对儿对应的两个词
        # 能够拼接成一个回文串
        mapping = {word: index for index, word in enumerate(words)}
        
        res = set()
        for inx, word in enumerate(words):
            
            # case 1: 检查空字符串是否是一个解
            if '' in mapping and word != '' and \
                self._is_palin(word):
                res.add((mapping[''], inx))
                res.add((inx, mapping['']))
            
            # case 2: 检查当前word的反词是否能组成一个解
            reversed_word = word[::-1]
            if reversed_word in mapping and mapping[reversed_word] != inx:
                res.add((mapping[reversed_word], inx))
                res.add((inx, mapping[reversed_word]))
            
            # case 3: 检查当前词拆成左右两部分
            # 核心之一
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
                if self._is_palin(left_word) and reversed_right in mapping:
                    res.add((mapping[reversed_right], inx))
                # (left, right, reversed_left)的形式
                if self._is_palin(right_word) and reversed_left in mapping:
                    res.add((inx, mapping[reversed_left]))
        
        return list(res)
    
    def _is_palin(self, word):
        left, right = 0, len(word) - 1
        while left < right:
            if word[left] != word[right]:
                return False
            left += 1
            right -= 1
        return True
```

#### 337. House Robber III
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # 跟1和2不一样的地方就在于这道题是在一棵二叉树上偷窃
    def rob(self, root: 'TreeNode') -> 'int':
        return self._dfs(root, cache=dict())
    
    def _dfs(self, node, cache):
        if not node:
            return 0
        
        if node in cache:
            return cache[node]
        
        val = node.val
        if node.left:
            val += self._dfs(node.left.left, cache) + self._dfs(node.left.right, cache)
        if node.right:
            val += self._dfs(node.right.left, cache) + self._dfs(node.right.right, cache)
        
        val = max(
            val,
            self._dfs(node.left, cache) + self._dfs(node.right, cache),
        )
        
        cache[node] = val
        return val
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
        # 感觉这道题用stack或者用queue都是可以的
        while stack:
            curr, level = stack.pop()
            if curr.isInteger():
                res += curr.getInteger() * level
            else:
                for t in curr.getList():
                    stack.append((t, level + 1))
        
        return res
```

#### 340. Longest Substring with At Most K Distinct Characters
```
from collections import defaultdict

class Solution:
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        这道题是说在s中找一个子串
        这个子串中最多有k个独特的字符
        返回能找到的最长的子串长度
        """
        # 滑动窗口好题！！！
        # 必背
        counter = defaultdict(int)
        res = left = right = 0
        
        while right < len(s):
            counter[s[right]] += 1
            while len(counter) > k:
                counter[s[left]] -= 1
                if counter[s[left]] == 0:
                    # 注意counter.pop(key)
                    # 和del counter[key]
                    # 是一样的
                    counter.pop(s[left])
                left += 1
            # 走到这儿时保证left和right之间的子串最多只有k个独特字符
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
        # 核心之一
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
        
        res = []
        for key, value in nums_count.items():
            heapq.heappush(res, (value, key))
            if len(res) > k:
                heapq.heappop(res)
        
        res.sort(reverse=True)
        return [i[1] for i in res]
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
                # 如果nums2中有重复的在nums1中出现的数字
                # 下次就不用再找了（因为已经被添加过了）
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
        # 这道题是返回所有重复的数字
        # 不需要去重
        # 所以两根指针可以做
        # 如果要求不能重复
        # 需要辅助set
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

#### 353. Design Snake Game
```
from collections import deque

class SnakeGame:

    def __init__(self, width: 'int', height: 'int', food: 'List[List[int]]'):
        """
        Initialize your data structure here.
        @param width - screen width
        @param height - screen height 
        @param food - A list of food positions
        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], 
        the second is at [1,0].
        """
        self._width = width
        self._height = height
        # 注意每个food的值都是1
        self._food = deque(tuple(i) for i in food)
        self._score = 0
        self._snake = deque([(0, 0)])

    def move(self, direction: 'str') -> 'int':
        """
        Moves the snake.
        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down 
        @return The game's score after the move. Return -1 if game over. 
        Game over when snake crosses the screen boundary or bites its body.
        """
        head_x, head_y = self._snake[0]
        tail_x, tail_y = self._snake[-1]

        if direction == 'U':
            head_x -= 1
        if direction == 'L':
            head_y -= 1
        if direction == 'R':
            head_y += 1
        if direction == 'D':
            head_x += 1
        # 蛇头移动完之后
        # 移动蛇尾
        # 直接pop出来就好
        self._snake.pop()

        # 第一个条件说明碰撞上了
        # 剩下的条件表示越界了
        if self._snake.count((head_x, head_y)) > 0 or \
            head_x < 0 or head_x >= self._height or \
            head_y < 0 or head_y >= self._width:
            return -1
        
        self._snake.appendleft((head_x, head_y))
        if self._food and (head_x, head_y) == self._food[0]:
            self._food.popleft()
            self._snake.append((tail_x, tail_y))
            self._score += 1

        return self._score
        

# Your SnakeGame object will be instantiated and called as such:
# obj = SnakeGame(width, height, food)
# param_1 = obj.move(direction)
```

#### 354. Russian Doll Envelopes
```
# 理解的不好！
# 回头再看看
class Solution:
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        求最终能套出多少个信封
        """
        # 二分法
        # envelopes里面存的是信封的宽和高(w, h)
        envelopes_copy = [(i[0], -i[1]) for i in envelopes]
        # 排序是按照宽度从小到大，高度从大到小（绝对值）
        envelopes_copy.sort()
        n = len(envelopes_copy)
        # res定义是在最终排好序的信封中，高度h从小到大的排序
        # 所以res的长度就是最终的答案
        res = []
        for i in range(n):
            l, r = 0, len(res)
            w, h = envelopes_copy[i][0], -envelopes_copy[i][1]
            while l < r:
                mid = l + (r - l) // 2
                # 这里的高度h是从大到小排列的
                # 核心：要找到最后一个大于等于t的坐标right
                # 表示外面的信封h能装的下mid
                if h > res[mid][1]:
                    l = mid + 1
                else:
                    r = mid
            if r == len(res):
                res.append((w, h, i))
            else:
                res[r] = (w, h, i)
        return len(res)

        # python TLE
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
                # 否则就重新push入堆
                if count > 0:
                    temp.append((-count, ch))
                # n是指当前剩余的字符个数
                n -= 1
            for each in temp:
                heappush(hp, each)
        
        return ''.join(res)
```

#### 359. Logger Rate Limiter
```
class Logger:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._mapping = dict()

    def shouldPrintMessage(self, timestamp: 'int', message: 'str') -> 'bool':
        """
        Returns true if the message should be printed in the given timestamp, 
        otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """
        if message not in self._mapping:
            self._mapping[message] = timestamp
            return True
        
        if timestamp - self._mapping[message] >= 10:
            self._mapping[message] = timestamp
            return True
        
        return False
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
        # 这道题是说给一个抛物曲线
        # 然后输入nums 输出sorted的结果
        # 要求O(n)实现
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

#### 361. Bomb Enemy
```
class Solution:
    def maxKilledEnemies(self, grid: 'List[List[str]]') -> 'int':
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        h1 = [[0] * n for _ in range(m)]
        h2 = [[0] * n for _ in range(m)]
        v1 = [[0] * n for _ in range(m)]
        v2 = [[0] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                temp = 0 if j == 0 or grid[i][j] == 'W' else h1[i][j - 1]
                h1[i][j] = temp + 1 if grid[i][j] == 'E' else temp
            for j in range(n - 1, -1, -1):
                temp = 0 if j == n - 1 or grid[i][j] == 'W' else h2[i][j + 1]
                h2[i][j] = temp + 1 if grid[i][j] == 'E' else temp
        
        for j in range(n):
            for i in range(m):
                temp = 0 if i == 0 or grid[i][j] == 'W' else v1[i - 1][j]
                v1[i][j] = temp + 1 if grid[i][j] == 'E' else temp
            for i in range(m - 1, -1, -1):
                temp = 0 if i == m - 1 or grid[i][j] == 'W' else v2[i + 1][j]
                v2[i][j] = temp + 1 if grid[i][j] == 'E' else temp
        
        res = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '0':
                    res = max(res, h1[i][j] + h2[i][j] + v1[i][j] + v2[i][j])
        
        return res
```

#### 364. Nested List Weight Sum II
```
class Solution:
    def depthSumInverse(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        # http://www.cnblogs.com/grandyang/p/5615583.html
        # Leetcode - StefanPochmann
        # 这道题是说外层的weighting大 
        weighted = unweighted = 0
        
        while nestedList:
            next_level_list = []
            for each in nestedList:
                if each.isInteger():
                    unweighted += each.getInteger()
                else:
                    next_level_list += each.getList()
            # 核心之一
            # 这样就保证了外层的weight会被不停的加进去
            weighted += unweighted
            nestedList = next_level_list
        
        return weighted
```

#### 365. Water and Jug Problem	
```
class Solution:
    def canMeasureWater(self, x: 'int', y: 'int', z: 'int') -> 'bool':
        # 这道题可以抽象为
        # z = m * x + n * y
        # 正就是倒入，负就是倒出
        # 核心就是看z是不是x和y的倍数
        # 别忘记了限制条件x + y >= z 因为无论如何都称不出比小于x y之和的水
        return z == 0 or x + y >= z and z % self._gcd(x, y) == 0
    
    def _gcd(self, a, b):
        if b == 0:
            return a
        
        return self._gcd(b, a % b)
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
#             # 实际上这里就是一层一层的遍历
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

#### 369. Plus One Linked List
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def plusOne(self, head: 'ListNode') -> 'ListNode':
        if not head:
            return head
        
        reversed_head = self._reverse(head)
        curr = reversed_head
        carry = 1
        
        while curr:
            pre = curr
            new_val = curr.val + carry
            carry = new_val // 10
            curr.val = new_val % 10
            curr = curr.next
        
        if carry:
            pre.next = ListNode(carry)
        
        return self._reverse(reversed_head)
    
    def _reverse(self, node):
        dummy_node = ListNode(-1)
        dummy_node.next = node
        
        curr = node
        while curr.next:
            temp = curr.next
            curr.next = temp.next
            # 翻转的核心！
            temp.next = dummy_node.next
            dummy_node.next = temp
        
        return dummy_node.next
```

#### 370. Range Addition
```
class Solution:
    def getModifiedArray(self, length: 'int', updates: 'List[List[int]]') -> 'List[int]':
        nums = [0] * (length + 1)
        for i in range(len(updates)):
            # 核心思路：
            # 区间的开头加数字，区间的结尾的下一位减数字
            # 因为最终往res里append的是累加和
            # 在新的区间上的操作时候需要先减去这个数字
            nums[updates[i][0]] += updates[i][2]
            nums[updates[i][1] + 1] -= updates[i][2]
        
        total = 0
        res = []
        for i in range(length):
            total += nums[i]
            res.append(total)
        
        return res
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
        # 这道题是说不用加好实现两个数字相加
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
        # 这里的nums1和num2都是sorted
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

#### 375. Guess Number Higher or Lower II
```
class Solution:
    def getMoneyAmount(self, n: 'int') -> 'int':
        # dp[i][j]定义：在数字i到数字j之间猜数字的最小花费
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(2, n + 1):
            for j in range(i - 1, -1, -1):
                global_min = 2 ** 31 - 1
                for k in range(j + 1, i):
                    # 核心之一
                    # 当前花费为k
                    # 再加上左右两部分的最大值作为local_min
                    local_max = k + max(dp[j][k - 1], dp[k + 1][i])
                    # 最终的global_min就是dp[j][i]的最小花费
                    global_min = min(global_min, local_max)
                dp[j][i] = j if j + 1 == i else global_min
        return dp[1][n]
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
            # 核心
            # 当前要比较的数字是mid
            # 看下当前的matrix里有多少个不大于这个数字的count
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
        self._data = []
        self._hash = {}

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self._hash:
            self._data.append(val)
            self._hash[val] = len(self._data) - 1
            return True
        else:
            return False
        
    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self._hash:
            # if self.hash[val] == len(self.data) - 1:
            if val == self._data[-1]:
                self._data.pop()
            else:
                ## 为了保证O(1)
                ## 将data中最后一个pop出，保存
                ## 然后覆盖掉data里真正需要pop的value
                ## 别忘了修改hash
                last_value = self._data.pop()
                self._data[self._hash[val]] = last_value
                self._hash[last_value] = self._hash[val]
            # 核心之一
            # 别忘了不管val是不是data的最后一位
            # 都要在hash里删除掉
            # 写成del self._hash[val]也是一样的
            self._hash.pop(val)
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

#### 381. Insert Delete GetRandom O(1) - Duplicates allowed
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
        self._map = dict()
        self._nums = list()

    def insert(self, val):
        """
        Inserts a value to the collection. Returns true if the collection did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        self._nums.append(val)
        if val in self._map:
            self._map[val].append(len(self._nums) - 1)
            return False
        else:
            self._map[val] = [len(self._nums) - 1]
            return True

    def remove(self, val):
        """
        Removes a value from the collection. Returns true if the collection contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self._map:
            pos = self._map[val].pop()
            if not self._map[val]:
                del self._map[val]
            if pos != len(self._nums) - 1:
                self._map[self._nums[-1]][-1] = pos
                self._nums[pos], self._nums[-1] = self._nums[-1], self._nums[pos]
            self._nums.pop()
            return True
        else:
            return False

    def getRandom(self):
        """
        Get a random element from the collection.
        :rtype: int
        """
        return choice(self._nums)


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
        # 这道题是说在不知道当前链表长度的情况下
        # 等概率的返回一个节点
        self._head = head

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        # 蓄水池典型题
        res = self._head
        node = self._head.next
        index = 1
        while node:
            # 这里的0就是hard coded一个点，fix住以后用来确定if的条件的
            # 由于index是每次递增的
            # 下面的res概率是1/2. 1/3, 1/4 ...
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

        # python shuffle函数
        # from random import shuffle
        # shuffle(self._data)
        # return self._data

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

#### 388. Longest Absolute File Path
```
import re
from collections import defaultdict

class Solution:
    def lengthLongestPath(self, input: 'str') -> 'int':
        # 求最长的绝对文件名路径
        # 注意不是最深的文件名路径
        lines = input.split('\n')
        level_mapping = defaultdict(str)
        res = 0
        
        for i in range(len(lines)):
            curr_level = lines[i].count('\t')
            lines[i] = level_mapping[curr_level - 1] + re.sub('\\t+', '/', lines[i])
            level_mapping[curr_level] = lines[i]
            
            # 说明此时的lines[i]代表一个文件名字
            if '.' in lines[i]:
                res = max(res, len(lines[i]))
        
        return res
```

#### 391. Perfect Rectangle
```
# 理解的不好
from collections import defaultdict

class Solution:
    def isRectangleCover(self, rectangles):
        """
        :type rectangles: List[List[int]]
        :rtype: bool
        """
        # rectangles里的点
        # 每个点都是左下角和右下角的坐标，一共4个值，每个坐标按照横纵轴排列
        left = min(x[0] for x in rectangles)
        bottom = min(x[1] for x in rectangles)
        right = max(x[2] for x in rectangles)
        top = max(x[3] for x in rectangles)

        points = defaultdict(int)
        for l, b, r, t in rectangles:
            # a b c d是左下，右下，右上，左上
            a, b, c, d = (l, b), (r, b), (r, t), (l, t)
            for p, q in zip((a, b, c, d), (1, 2, 4, 8)):
                if points[p] == q:
                    return False
                points[p] |= q

        for px, py in points:
            # px和py不能是边界点
            # 边界点可以为1， 2， 3， 4
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
        # 这道题是问s不是t的子序列
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
        # 输入的s形如s="3[a]2[bc]" return "aaabcbc"
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
        # 返回这个子串的长度
        # 1. 常规解法， 类似two pointers, i是起点，j是终点
        # 由于字母只有26个，而整型mask有32位，足够用了
        # 每一位代表一个字母，如果为1，表示该字母不够k次，如果为0就表示已经出现了至少k次
        res = 0
        n = len(s)
        i = 0
        while i + k <= n:
            # 大循环里遍历子串的起点i
            # 在小循环里遍历子串的终点j
            # 如果当前i到j的子串满足了条件
            # 就去更新res
            m = [0] * 26
            # mask = 0初始意味着32位上全是0
            # 实际上这里的mask和m起到的作用是一样的
            # 只不过如果不用这个mask
            # 我们需要将m遍历看m里的每个非零的元素满不满足大于等于k
            mask = 0
            max_idx = i
            for j in range(i, n):
                t = ord(s[j]) - ord('a')
                m[t] += 1
                if m[t] < k:
                    # 将mask的二进制第t位变成1(不管以前是不是1)
                    mask |= 1 << t
                else:
                    # 将mask的二进制第t位设位0(不管以前是不是0)
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
        return [6.0, 0.5, -1.0, 1.0, -1.0].
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
        # 这道题说的是从num这个string中remove掉k个字符
        # 使的num最小
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

#### 403. Frog Jump
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
                # 这个`if stone + jumps - 1 != stone`的条件不可或缺
                # 否则当[0, 1...]头两块石头就会出错！！
                # 因为在1石头上会在循环中更新pos 1位置的set（额外加上一个0）
                # 相当于在循环中我们去更新了这个set 是不允许的
                if stone + jumps - 1 != stone \
                    and stone + jumps - 1 > 0 \
                    and stone + jumps - 1 in stones_hash:
                    stones_hash[stone + jumps - 1].add(jumps)
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
        # 先把四周的点的高度以及坐标入队
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
                if curr_sea_level > heightMap[newi][newj]:
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

#### 418. Sentence Screen Fitting
```
class Solution:
    def wordsTyping(self, sentence, rows, cols):
        # 暴力模拟法
        # python3通不过
        # java版本能通过
        res = 0
        word_inx = 0
        i = 0
        while i < rows:
            j = 0
            while j < cols:
                if j + len(sentence[word_inx]) <= cols:
                    j += len(sentence[word_inx]) + 1
                    word_inx += 1
                else:
                    j = cols
                if word_inx == len(sentence):
                    res += 1
                    word_inx = 0
            i += 1
        
        return res
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
        # 核心之一
        # 则如果是开头，其左边和上边一定没有x
        # 这道题实际上用了贪心
        # 用BFS也可以做
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

#### 424. Longest Repeating Character Replacement
```
from collections import defaultdict

class Solution:
    def characterReplacement(self, s: 'str', k: 'int') -> 'int':
        # 这道题是说可以修改k个字符串
        # 问这样修改以后最长的相同字符的子串长度是多少
        # 实际上就是滑动窗口问题
        # 问的是最长的滑动窗口，里面满足
        # 这个窗口里面除了最多的字符以外的其他字符的总和小于等于K
        # 这道题开始做的时候考虑思路是反着的
        # 去想最少的字符个数要小于等于k
        # 这样就会限制字符种类在窗口里只能有两个
        # 实际上应该想出现次数最多的个数要大于等于win_length - k
        n = len(s)
        l = r = 0
        max_count = 0
        mapping = defaultdict(int)
        res = 0
        while r < n:
            mapping[s[r]] += 1
            max_count = max(max_count, mapping[s[r]])
            # 核心之一
            # 就是说除了出现次数最多以外的字符个数应该小于等于k
            while r - l + 1 - max_count > k:
                mapping[s[l]] -= 1
                l += 1
            res = max(res, r - l + 1)
            r += 1
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
        # 就是中序遍历
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
            # 这里是中序遍历的操作
            curr = curr.right
        
        first.left = last
        last.right=  first
        return first
```

#### 428. Serialize and Deserialize N-ary Tree
```
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
from collections import deque

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
        
        tokens = deque(data.split(' '))
        root = Node(int(tokens.popleft()), [])
        
        self._bfs(root, tokens)
        return root
    
    def _bfs(self, node, tokens):
        if not tokens:
            return
        
        # add child nodes with subtrees
        while tokens[0] != '#':
            value = tokens.popleft()
            child = Node(int(value), [])
            node.children.append(child)
            self._bfs(child, tokens)
        
        # discard the "#"
        tokens.popleft()

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
#  1---2---3---4---5---6--NULL
#          |
#          7---8---9---10--NULL
#              |
#              11--12--NULL
# child只有一个（比如1,3或者8）
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

#### 432. All O`one Data Structure
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
        # 初始化的时候
        # 先把这两个node连接起来
        self.head_node.next = self.tail_node
        self.tail_node.prev = self.head_node
    
    # 往x后面加一个node
    # 这里的x已经保证在这个double linkedlist里面了
    def insert_after(self, x):
        new_node = Node()
        temp = x.next
        x.next = new_node
        new_node.next = temp
        new_node.prev = x
        temp.prev = new_node
        return new_node

    # 往x前面加一个node
    # 同样的，这里的x已经保证在这个double linkedlist里面了
    def insert_before(self, x):
        new_node = Node()
        temp = x.prev
        x.prev = new_node
        new_node.prev = temp
        new_node.next = x
        temp.next = new_node
        return new_node
    
    def remove(self, x):
        # 因为是双向链表
        # 所以remove就比较简单了
        # 注意这里的x依旧是在这个double linkedlist里面的
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
        # 这道题的核心
        # 就是用一个双向链表存递增的存frequency
        # 这里的双向链表
        # head是当前frequency最小的node
        # tail是当前frequency最大的node
        self.dll = DoubleLinkedList()
        # key到frequency的映射
        self.key_counter = defaultdict(int)
        # frequency到某个节点的映射
        # 而这个节点用来承装所有的key（使用key_set）
        self.node_freq = {0: self.dll.get_dummy_head()}

    # prev_freq就是前一个frequency的意思
    # 能执行到下面的函数
    # 说明prev_freq一定已经在node_freq里面了
    # 所以不需要再检查
    def _remove_key_pf_node(self, prev_freq, key):
        node = self.node_freq[prev_freq]
        node.remove_key(key)
        if node.is_empty():
            self.dll.remove(node)
            self.node_freq.pop(prev_freq)
        
    def inc(self, key):
        """
        Inserts a new key <Key> with value 1. Or increments an existing key by 1.
        :type key: str
        :rtype: void
        """
        # 生成当前的，调整过去的
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
        # 同理也是生成当前的，调整过去的
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
        
        for curr in range(1, n):
            curr_start, curr_end = ins[curr]
            last_start, last_end = ins[last]
            if curr_start < last_end:
                # 此时要移除一个了
                # 实际上并没有移除，但是将last的位置变成当前end最小的那个i
                # 为了保证我们总体去掉的区间数最小，我们去掉那个end值较大的区间
                # 就是说如果我们保留的是大的区间
                # 下一次curr的start还是可能会小于last_end
                # 就会导致又增加了res
                # 我们要尽量少的增加res
                # end值较大的区间意味着可能会overlapping后面的区间
                res += 1
                if curr_end < last_end:
                    last = curr
            else:
                last = curr
        
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
    def pathSum(self, root, target):
        """
        :type root: TreeNode
        :type target: int
        :rtype: int
        """
        # 这里的pathSum定义是可以包括也可以不包括当前root的target种类数目
        if not root:
            return 0
        return self.pathSum(root.left, target) + \
            self.pathSum(root.right, target) + \
            self._contain_root(root, target)
    
    # _contain_root的定义就是必须包括当前root的target种类数目
    def _contain_root(self, root, target):
        if not root:
            return 0
        res = 0
        if root.val == target:
            res += 1
        res += self._contain_root(root.left, target - root.val) + \
            self._contain_root(root.right, target - root.val)
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
        # 这道题是说找出s中的所有start index
        # 使得s的这个子串可以成为p的anagram
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

#### 444. Sequence Reconstruction
```
from collections import deque

class Solution:
    def sequenceReconstruction(self, org: 'List[int]', seqs: 'List[List[int]]') -> 'bool':
        # 这道题用拓扑排序来解
        # 在每个seq中的每个数字都是一个node，分别指向后面的数字
        # 这样就建立了先后顺序
        # 实际上这道题是问能不能从这个图中重建出唯一的一个顺序
        graph = self._build_graph(seqs)
        topo_order = self._topological_sort(graph)
        return topo_order == org
    
    def _build_graph(self, seqs):
        graph = dict()

        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = set()
        
        for seq in seqs:
            for i in range(1, len(seq)):
                graph[seq[i - 1]].add(seq[i])
        
        return graph

    
    def _build_indegrees(self, graph):
        indegrees = {node: 0 for node in graph}
        
        for node in graph:
            for neighbour in graph[node]:
                indegrees[neighbour] += 1
                
        return indegrees
    
    def _topological_sort(self, graph):
        indegrees = self._build_indegrees(graph)
        queue = deque()
        
        for node, degrees in indegrees.items():
            if degrees == 0:
                queue.append(node)
        
        res = []
        while queue:
            if len(queue) > 1:
                return None
            
            node = queue.popleft()
            res.append(node)
            for nei in graph[node]:
                indegrees[nei] -= 1
                if indegrees[nei] == 0:
                    queue.append(nei)

        # 核心之一
        # 当前图中一共有len(graph)个点
        # 则最终的结果应该全部遍历完
        if len(res) == len(graph):
            return res
        return None
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
        # 注意这道题的顺序
        # 这里的两个链表是从高位到低位
        # 所以要先用两个stack接着
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

#### 446. Arithmetic Slices II
```
from collections import defaultdict

class Solution:
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        # 这道题要求返回的是A里面的等差子序列个数
        # 注意子序列的长度要求大于等于3
        if not A:
            return 0

        n = len(A)
        # 这里的dp[0]永远是一个空的dict
        # 因为以索引0结尾的只是一个数，不会有等差数列
        # 这里的dp[i]是一个hashmap
        # 核心之一：（重点）表示以i结尾的数组存在以diff为等差的等差数列的个数
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

#### 447. Number of Boomerangs
```
from collections import defaultdict

class Solution:
    def numberOfBoomerangs(self, points: 'List[List[int]]') -> 'int':
        # 回力镖问题
        res = 0
        for i1, j1 in points:
            mapping = defaultdict(int)
            for i2, j2 in points:
                distx = i1 - i2
                disty = j1 - j2
                mapping[distx ** 2 + disty ** 2] += 1
            for dist, count in mapping.items():
                # 排列组合问题
                # 比如fix了a点，有b c d三个点到a的距离都相等
                # 则答案是3 * 2
                res += count * (count - 1)
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
from collections import deque

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
        node_list = deque(data.split(', '))
        return self._deserialize(node_list)
        
    def _deserialize(self, node_list):
        if not node_list:
            return
        
        # 核心：这里要popleft()，注意将node_list是一个deque
        curr = node_list.popleft()
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
        # 输入tree，输出eert
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

#### 457. Circular Array Loop
```
class Solution:
    def circularArrayLoop(self, nums: 'List[int]') -> 'bool':
        # 这道题是说如果遍历的num是整数
        # 往前走num步
        # 如果是负数
        # 往后走num步
        # 问是否有环
        next_inx_mapping = {}
        n = len(nums)
        
        for i in range(n):
            # 循环每次都以i为起点来判断有没有环
            curr = i
            while True:
                next_inx = (curr + nums[curr] + n) % n
                if next_inx == curr or nums[curr] * nums[next_inx] < 0:
                    break
                if next_inx in next_inx_mapping:
                    return True
                next_inx_mapping[curr] = next_inx
                curr = next_inx
        
        return False
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
        # acc1和acc2都是0到n的序号
        mapping = defaultdict(int)
        for acc1, acc2, money in transactions:
            mapping[acc1] -= money
            mapping[acc2] += money
            
        accounts = []
        # cnt统计的是accounts现在有多少个不平衡的账户
        for acc, money in mapping.items():
            if money != 0:
                accounts.append(money)
        
        return self._dfs(accounts, start=0, old_steps=0)
    
    def _dfs(self, accounts, start, old_steps):
        new_steps = 2 ** 31 - 1
        
        # 先找第一个不平衡的账户的位置
        # 注意第一次进入这个递归函数的时候，start肯定是0
        # 但是后面由于在for循环中更新了account的元素就不一定了
        # 所以我们这里需要再求一下
        while start < len(accounts) and accounts[start] == 0:
            start += 1
        
        # 核心递归：遍历所有account里start以后的不平衡情况(fix start，从start以后的所有的options)
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

#### 475. Heaters
```
from bisect import bisect_left
from bisect import bisect_right

class Solution:
    def findRadius(self, houses: 'List[int]', heaters: 'List[int]') -> 'int':
        heaters.sort()
        res = 0
        for house_pos in houses:
            rad = 2 ** 31 - 1
            # 核心：
            # 利用二分查找
            # 分别找到不大于house的最大加热器坐标le
            # 以及不小于house的最小加热器坐标ge
            # 则当前房屋所需的最小加热器半径
            # rad = min(house - le, ge - house)
            le = bisect_right(heaters, house_pos)
            if le != 0:
                rad = min(rad, house_pos - heaters[le - 1])
            ge = bisect_left(heaters, house_pos)
            if ge < len(heaters):
                rad = min(rad, heaters[ge] - house_pos)
            
            res = max(res, rad)
        
        return res
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
        # dp[i][j]定义
        # 在s的i到j之间最长的回文串长度是多少
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
        last_end = points[0][1]
        for curr_start, curr_end in points[1:]:
            if last_end >= curr_start:
                # 不能是max
                # 比如例子[[1, 6], [2, 8], [7, 12], [10, 16]]
                # 本来第一只箭在1和6之间
                # 如果max的话就会把last_end变成8
                # 导致7和12这个范围也被认为可以被第一只箭包括
                # which is not因为第一只箭的范围在1到6之间
                # 从而使得res值少了
                last_end = min(last_end, curr_end)
            else:
                last_end = curr_end
                res += 1
        
        return res
```

#### 460. LFU Cache
```
from collections import defaultdict

class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = self.next = None

class DoubleLinkedList:
    def __init__(self):
        self.dummy_head = Node(0, 0)
        self.dummy_tail = Node(0, 0)
        self._count = 0
        self.dummy_head.next = self.dummy_tail
        self.dummy_tail.prev = self.dummy_head
    
    # 将new_node插入到existing_node后面
    def insert(self, existing_node, new_node):
        temp = existing_node.next
        existing_node.next = new_node
        new_node.prev = existing_node
        new_node.next = temp
        temp.prev = new_node
        self._count += 1
    
    # 将这个node从双向链表中移除
    def remove(self, node):
        prev = node.prev
        prev.next = node.next
        node.next.prev = prev
        self._count -= 1

    def size(self):
        return self._count
    
    def get_head(self):
        if self._count > 0:
            return self.dummy_head.next
        return None
    
    def get_tail(self):
        if self._count > 0:
            return self.dummy_tail.prev
        return None

    def append(self, node):
        self.insert(self.dummy_tail, node)
        
    def append_left(self, node):
        self.insert(self.dummy_head, node)
    
    def pop(self):
        self.remove(self.get_tail())
    
    def pop_left(self):
        self.remove(self.get_head())

class LinkedHashMap:
    # 实质上LFU就是一个二级key的hash map
    # 第一级的key是frequency
    # 第二级的key才是真正的key
    def __init__(self):
        # 基本数据结构就是用一个dict
        # key就是key，value是对应着在一个双向链表中的节点(这个节点也用key和value生成)
        # 这样就很方便的可以用O（1）来定位节点的位置
        # 方便插入删除操作
        self.node_map = dict()
        self.ddl = DoubleLinkedList()
    
    def size(self):
        return len(self.node_map)
    
    def contains(self, key):
        return key in self.node_map
    
    def get(self, key):
        return self.node_map[key].value
    
    # 注意：
    # 下面的三个涉及到移动的操作：remove, move_left, move_right
    # 都是针对key来移动的
    # 将key对应的node从双向链表中移除
    def remove(self, key):
        node = self.node_map[key]
        self.ddl.remove(node)
        self.node_map.pop(key)
    
    # 将key对应的node移动到双向链表的最左端
    def move_left(self, key):
        node = self.node_map[key]
        self.ddl.remove(node)
        self.ddl.append_left(node)
    
    # 将key对应的node移动到双向链表的最右端
    def move_right(self, key):
        node = self.node_map[key]
        self.ddl.remove(node)
        self.ddl.append(node)

    def append(self, key, value):
        if not self.contains(key):
            node = Node(key, value)
            self.ddl.append(node)
            self.node_map[key] = node
        else:
            self.node_map[key].value = value
            self.move_right(key)
    
    def append_left(self, key, value):
        if not self.contains(key):
            node = Node(key, value)
            self.ddl.append_left(node)
            self.node_map[key] = node
        else:
            self.node_map[key].value = value
            self.move_left(key)
    
    def pop(self):
        key = self.ddl.get_tail().key
        self.remove(key)
        return key
    
    def pop_left(self):
        key = self.ddl.get_head().key
        self.remove(key)
        return key

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = -1
        # cache里存的形式是：key: [value, frequency]
        self.cache = dict()
        # freq_map里存的形式是：freq: LinkedHashMap(其中里面是key：（key, value）形式的node)
        self.freq_map = defaultdict(LinkedHashMap)

    def get(self, key: int) -> int:
        if key in self.cache:
            value, old_freq = self.cache[key]
            self.cache[key][1] += 1
            
            old_freq_is_zero = False
            self.freq_map[old_freq].remove(key)
            if self.freq_map[old_freq].size() == 0:
                old_freq_is_zero = True
                self.freq_map.pop(old_freq)
            self.freq_map[old_freq + 1].append_left(key, value)
            
            if old_freq == self.min_freq and old_freq_is_zero:
                self.min_freq += 1
            return value
    
        return -1
            
    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return
        if key in self.cache:
            self.cache[key][0] = value
            # 主要是为了触发一下key的freq增值
            self.get(key)
        else:
            if len(self.cache) == self.capacity:
                min_freq_linkedlisthashmap = self.freq_map[self.min_freq]
                min_freq_node = min_freq_linkedlisthashmap.pop()
                self.cache.pop(min_freq_node)
            self.cache[key] = [value, 1]
            self.freq_map[1].append_left(key, value)
            self.min_freq = 1
            
# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
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
                # 下面处理grid[i][j] == 1的情况
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
            for i in range(n + 1 - step):
                j = i + step - 1
                dp[i][j] = s[i:i + step]
                for k in range(i, j):
                    left = dp[i][k]
                    right = dp[k + 1][j]
                    if len(left) + len(right) < len(dp[i][j]):
                        dp[i][j] = left + right
                temp = s[i:j + 1]
                replace = ''
                # 核心之一:
                # find函数： S.find(sub[, start[, end]]) -> int
                # 应该从1开始找，避免找到0
                pos = (temp * 2).find(temp, 1)
                if pos >= len(temp):
                    replace = temp
                else:
                    # 核心之一：len(temp) // pos 就是temp中有多少个重复的pattern！
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
            # 核心之一
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

#### 486. Predict the Winner
```
class Solution:
    def PredictTheWinner(self, nums: 'List[int]') -> 'bool':
        n = len(nums)
        mem = [[-1] * n for _ in range(n)]
        # 实际上最终dfs返回的是第一个人获得的价值减去第二个人获得的价值
        # 所以如果大于等于0就不会输
        return self._dfs(nums, start=0, end=n - 1, mem=mem) >= 0
    
    # dfs的定义是在start和end的状态下
    # 先取石子的人会比后取石子的人多多少分
    def _dfs(self, nums, start, end, mem):
        if mem[start][end] != -1:
            return mem[start][end]
        
        if start == end:
            mem[start][end] = nums[start]
        else:
            mem[start][end] = max(
                nums[start] - self._dfs(nums, start + 1, end, mem),
                nums[end] - self._dfs(nums, start, end - 1, mem)
            )
        
        return mem[start][end]
```

#### 489. Robot Room Cleaner
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
        self._dfs(robot, x=0, y=0, dx=0, dy=1, visited=visited)
    
    # 注意：当每次递归调用完dfs后，robot是回到原处的
    def _dfs(self, robot, x, y, dx, dy, visited):
        robot.clean()
        visited.add((x, y))

        for _ in range(4):
            # 向四个方向递归下去
            # 这道题不能使用传统的方向数组的原因在于
            # 需要一直走下去直到走到头
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
        # 这道题要求出所有的递增子序列
        # 典型的DFS题目
        res = set()
        self._dfs(nums, start=0, curr=[], res=res)
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

#### 493. Reverse Pairs
```
class Solution:
    def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # 这道题也可以用树状数组来解
        # 假设nums里的数字是有上限的
        # 则用一个数组（或者哈希表）c，数组c里的下标就是nums里的值
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
            # 为什么这里是j - (mid + 1)？
            # 因为相当于在fix i坐标
            # 得到以i坐标开头有一个满足条件的对儿
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
        # 只能用加号或者减号
        # 有多少种能凑成S的方案
        # dp定义是以某个start作为下标
        # 对于凑成total这个值（作为key）有多少种方案（作为value）
        dp = [dict() for _ in range(len(nums))]
        return self._dfs(nums, total=S, start=0, dp=dp)
    
    # nums从start开始凑成total有多少种答案
    # 好题！需要多多体会
    # 典型的自顶向下的记忆化搜索
    # dfs定义就是以nums的start下标为开始，有多少种凑成total的方案
    def _dfs(self, nums, total, start, dp):
        if start == len(nums):
            return 1 if total == 0 else 0
        
        if total in dp[start]:
            return dp[start][total]

        cnt1 = self._dfs(nums, total - nums[start], start + 1, dp)
        cnt2 = self._dfs(nums, total + nums[start], start + 1, dp)
        # 凑成total有两种选择
        # 一种是当前total减去当前nums[start] -> cnt1
        # 另一种是当前的total加上nums[start] -> cnt2
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
        # 思路就是模拟法
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
        # Maze I是问能不能到达
        # 这道题是问最短的距离是多少
        # 当时遇到这道题的时候由于思路定式不知道小球该往哪儿走
        # 被这里卡住了
        # 其实方向随便选，设定一个有序的方向数组就好了
        # 核心思路BFS
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
                # 注意这里跟常规的BFS求极值不同
                # 不能直接return当前的dist
                # 因为很可能后面还会有更小的值
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
        # 暴力解或者前缀和 感觉都差不多
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

#### 524. Longest Word in Dictionary through Deleting
```
from functools import cmp_to_key

def _cmp(a, b):
    # 对于cmp函数来说，输入是a b，如果返回-1，则a排在前面
    if len(a) > len(b):
        return -1
    elif len(a) < len(b):
        return 1
    else:
        if a > b:
            return 1
        if a < b:
            return -1
        return 0

class Solution:
    def findLongestWord(self, s, d):
        # 这道题是说从s中删除字符，返回能变成的d中的最长的单词
        d.sort(key=cmp_to_key(_cmp))
        for word in d:
            i = 0
            for ch in s:
                if i < len(word) and ch == word[i]:
                    i += 1
            if i == len(word):
                return word
        
        return ''
```

#### 527. Word Abbreviation
```
class Solution:
    def wordsAbbreviation(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        # 缩写但是不能有重复的字符串
        # 解法也是暴力解
        # 一个一个试前缀使用多长
        n = len(words)
        res = list(map(self._abbrev, words))
        prefix = [0] * n
        
        for i in range(n):
            while True:
                # O(n^2)遍历
                # 在小循环中将所有重复的索引加入一个set
                # 然后将set中所有的词增加一个k
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
    # 保留0到索引i的字符，从第i+1到最后执行缩写操作
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

#### 529. Minesweeper
```
class Solution:

    _DIRS = [
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ]

    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """
        if not board or not board[0]:
            return board
        
        m, n = len(board), len(board[0])
        row, col = click
        
        if board[row][col] == 'M':
            board[row][col] = 'X'
        else:
            mine_counts = 0
            for di, dj in self._DIRS:
                newi, newj = row + di, col + dj
                if not 0 <= newi < m or not 0 <= newj < n:
                    continue
                if board[newi][newj] == 'M':
                    mine_counts += 1
            if mine_counts > 0:
                board[row][col] = str(mine_counts)
            else:
                board[row][col] = 'B'
                for di, dj in self._DIRS:
                    newi, newj = row + di, col + dj
                    if not 0 <= newi < m or not 0 <= newj < n:
                        continue
                    if board[newi][newj] == 'E':
                        self.updateBoard(board, [newi, newj])
                
        return board
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
        # 左边最大的深度和右边最大的深度
        left_dia = self._dfs(node.left)
        right_dia = self._dfs(node.right)
        self.res = max(self.res, left_dia + right_dia)
        return max(left_dia, right_dia) + 1
```

#### 544. Output Contest Matches
```
class Solution:
    def findContestMatch(self, n: 'int') -> 'str':
        v = [str(i) for i in range(1, n + 1)]
        while n > 1:
            for i in range(n // 2):
                # 注意下标
                # 这样就保证了1-8， 2-4
                # 这道题的n限制了一定是2的k次幂
                v[i] = '({},{})'.format(v[i], v[n - i - 1])
            n //= 2
        return v[0]
```

#### 547. Friend Circles
```
class Solution:
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        # 就是number of island的
        # 这道题尝试用DFS来解
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

#### 549. Binary Tree Longest Consecutive Sequence II
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def longestConsecutive(self, root: 'TreeNode') -> 'int':
        # 非常好的递归题目
        # 基本思想就是递归经过当前节点和不经过当前结果
        if not root:
            return 0

        res = self._dfs(root, 1) + self._dfs(root, -1) + 1

        return max(
            res,
            self.longestConsecutive(root.left),
            self.longestConsecutive(root.right),
        )
    
    def _dfs(self, node, diff):
        if not node:
            return 0
        
        left = right = 0
        if node.left and node.val - node.left.val == diff:
            left = 1 + self._dfs(node.left, diff)
        if node.right and node.val - node.right.val == diff:
            right = 1 + self._dfs(node.right, diff)
        
        return max(left, right)
```

#### 552. Student Attendance Record II
```
class Solution:
    def checkRecord(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 假设一共有n天
        # 问一共有多少种可行的出席记录 A P L
        # 可行的记录是指最多一次A，最多两次L
        if n == 0:
            # 只有一种就是空
            return 1
        if n == 1:
            # A P L一共3种
            return 3
        
        # 我们先考虑不含'A'的情况
        # num[i + 1]表示前n中前i个字符在不包含'A'的时候能凑成多少种答案
        # 则此时只有如下3种结尾的方案：
        # 1. 以'P'结尾，这个结尾长度为1，所以我们要回溯到nums[i]
        # 2. 以'PL'结尾，这个结尾长度为2，所以我们要回溯到nums[i - 1]
        # 3. 以'PLL'结尾，这个结尾长度为3，所以我们要回溯到nums[i - 2]
        nums = [1, 1, 2]
        for i in range(2, n):
            nums.append((nums[i] + nums[i - 1] + nums[i - 2]) % 1000000007)
        
        # 先求一遍此时的答案
        # 下面三项就是分别表示以P结尾，以PL结尾，以PLL结尾的答案
        res = (nums[n] + nums[n - 1] + nums[n - 2]) % 1000000007
        
        # 然后insert 'A'到答案中，假设insert的位置就是i
        # 此时左边表示n中i个字符有多少种答案
        # 右边表示n中n - i - 1个字符有多少种答案
        # 注意总长度还是i + (n - i - 1) + 1 = n的
        for i in range(n):
            res += nums[i + 1] * nums[n - i] % 1000000007
            res %= 1000000007
        
        return res
    
    # 还有思路（从grandyang的答案中）
    # 设定三个DP数组A, P, L
    # 分别表示以A结尾 以P结尾 以L结尾的答案数目
    # 则最终的答案就是A[n - 1] + P[n - 1] + L[n - 1]
    # 递归式分别为
    # A[i] = A[i-1] + A[i-2] + A[i-3]
    # P[i] = A[i-1] + P[i-1] + L[i-1]
    # L[i] = A[i-1] + P[i-1] + A[i-2] + P[i-2]
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
        # 这道题是说从wall中画一条竖线
        # 问break最少的砖头数是多少
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
                # res实际上指的是最多的缝
                # 则最终的len(wall) - res
                # 就是最少能break多少块砖头
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
        # 这道题根next permutation那道题几乎一样
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

        # 这里用while循环然后乘10相加也是一样的
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
        curr_sum = 0
        # 某个sum值（这个值是从最右边0开始算的）
        # 基本思路就是如果这个值出现过两次
        # 则中间的坐标之间的和就是0
        # 推广一下
        # 如果curr_sum出现过，而且curr_sum-target**在之前的遍历中**也出现过
        # 则中间的差就是target！！！所以此时答案数目可以加上此时curr_sum-target出现的次数
        mapping = defaultdict(int)
        
        for num in nums:
            mapping[curr_sum] += 1
            curr_sum += num
            # curr_sum - (curr_sum - target) == target!!!
            res += mapping[curr_sum - target]
    
        return res
```

#### 562. Longest Line of Consecutive One in Matrix
```
class Solution:
    def longestLine(self, M: 'List[List[int]]') -> 'int':
        if not M or not M[0]:
            return 0
        
        m, n = len(M), len(M[0])
        dp = [[[0] * 4 for _ in range(n)] for _ in range(m)]
        
        res = 0
        for i in range(m):
            for j in range(n):
                if M[i][j] == 0:
                    continue
                for k in range(4):
                    dp[i][j][k] = 1
                if j > 0:
                    # 垂直方向
                    dp[i][j][0] += dp[i][j - 1][0]
                if i > 0:
                    # 水平方向
                    dp[i][j][1] += dp[i - 1][j][1]
                if i > 0 and j < n - 1:
                    dp[i][j][2] += dp[i - 1][j + 1][2]
                if i > 0 and j > 0:
                    dp[i][j][3] += dp[i - 1][j - 1][3]
                res = max(res, max(dp[i][j]))
        
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
        # 这道题是说给一个数字，找出跟这个数字最近的回文数字
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
        # 因为是找最近的
        # 所以我们只根据左半部分来确定有半部分就好
        pre_fix = int(n[:(n_len + 1) // 2])
        
        for i in (-1, 0, 1):
            pre = str(pre_fix + i)
            if n_len % 2 == 0:
                new_str = pre + pre[::-1]
            else:
                new_str = pre + pre[-2::-1]
            candidates.add(int(new_str))

        if n_num in candidates:
            # 去掉这个数字本身
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
        # 这道题是说s1的任意一个全排列的字符串是不是s2的子串
        # 典型滑动窗口问题
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

#### 568. Maximum Vacation Days
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
        # flights里面用0和1来确定能不能到达
        # days里面存的是在n城市第k周能获得多少节假日
        # 非常典型的dp题目
        n = len(flights)
        k = len(days[0])
        # dp[n][k]的定义是在第k周当前在n城市，未来将会获得的最大假期
        # 相当于如果通过这一个点，未来将会获得的最大假期
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
        # 这道题实际上就是模拟法
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
                # 全大写的
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

#### 604. Design Compressed String Iterator
```
class StringIterator:

    def __init__(self, compressedString: 'str'):
        # 输入是L1e2t1C1o1d1e1
        # 后面的数字就是前面的字母的重复个数
        self._string = compressedString
        self._n = len(compressedString)
        self._pos = 0
        self._count = 0
        self._c = ' '

    def next(self) -> 'str':
        if self.hasNext():
            self._count -= 1
            return self._c
        return ' '

    def hasNext(self) -> 'bool':
        if self._count > 0:
            return True
        if self._pos >= self._n:
            return False
        
        self._c = self._string[self._pos]
        self._pos += 1
        
        while self._pos < self._n and '0' <= self._string[self._pos] <= '9':
            self._count = 10 * self._count + int(self._string[self._pos])
            self._pos += 1
        return True


# Your StringIterator object will be instantiated and called as such:
# obj = StringIterator(compressedString)
# param_1 = obj.next()
# param_2 = obj.hasNext()
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
        # 这道题是问能不能种n朵花在flowerbed里面
        # 前提是不能种相邻的花
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
        # 这道题是问nums能凑成多少个三角形
        nums.sort()
        n = len(nums)
        res = 0

        for i in range(n - 1, 1, -1):
            left, right = 0, i - 1
            # 实际上这里的循环是来确定right的
            while left < right:
                if nums[left] + nums[right] > nums[i]:
                    res += right - left
                    right -= 1
                else:
                    left += 1
        
        return res
```

#### 616. Add Bold Tag in String
```
class Solution:
    def addBoldTag(self, s: 'str', words: 'List[str]') -> 'str':
        n = len(s)
        mask = [False] * n
        for i in range(n):
            prefix = s[i:]
            for word in words:
                if prefix.startswith(word):
                    for j in range(i, min(i + len(word), n)):
                        mask[j] = True
        
        res = ''
        i = 0
        while i < n:
            if mask[i] is False:
                res += s[i]
                i += 1
            else:
                res += '<b>'
                while i < n and mask[i] is True:
                    res += s[i]
                    i += 1
                res += '</b>'
        
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
        而不是每个task都需要相隔n个间隔
        """
        # 问最终一共需要最短的多少个间隔
        cnt = [0] * 26
        for task in tasks:
            cnt[ord(task) - ord('A')] += 1
        
        cnt.sort()
        mx = cnt[25]
        length = len(tasks)
        # first_less_mx_inx是sort过后的cnt数组中从左到右最后一个值小于mx的
        # 注意python里的list的index方法是从左到右扫的
        # 比如cnt=[0, 0, ...3, 3]
        # 会找到的是第一个3的位置作为cnt.index(mx)
        first_less_mx_inx = cnt.index(mx) - 1
        
        # n + 1就是每个子集的长度
        # mx是出现的最多次数
        # 因为第一项是不考虑最后一大坨出现次数相等并且出现次数都是最多的task
        # 所以mx - 1
        # 25 - first_less_mx_inx是指在最后要排剩余的不包含mx任务那那一堆任务
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
        # 这道题是说输入是k个有序的数组
        # 找出一个最小的range
        # 能够至少包含每个数组中的一个数字
        # 这道题实际上可以转换为滑动窗口问题
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
                # 出现了更小的间隔
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
                    # 此时还需要更新一下上一次执行函数的开始时间
                    # why?
                    # 因为求的是独家的运行时间
                    stack[-1][1] = timestamp + 1
        
        return res
```

#### 640. Solve the Equation
```
class Solution:
    def solveEquation(self, equation: 'str') -> 'str':
        left, right = equation.split('=')
        lhs = rhs = 0
        
        for each in self._break(left):
            if 'x' in each:
                lhs += int(self._coeff_x(each))
            else:
                rhs -= int(each)
        
        for each in self._break(right):
            if 'x' in each:
                lhs -= int(self._coeff_x(each))
            else:
                rhs += int(each)
        
        if lhs == 0:
            if rhs == 0:
                return 'Infinite solutions'
            else:
                return 'No solution'
        
        return 'x=' + str(rhs // lhs)
    
    def _coeff_x(self, string):
        if len(string) > 1 and '0' <= string[-2] <= '9':
            # +3x变成+3
            return string.replace('x', '')
        # 比如+x或者-x变成+1或者-1
        return string.replace('x', '1')
    
    def _break(self, string):
        res = []
        r = ''
        for ch in string:
            if ch in '+-':
                if r:
                    res.append(r)
                r = ch
            else:
                r += ch
        res.append(r)
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
        if not root:
            return res
        
        if root.is_end:
            res.append((root.hotness, root.sentence))

        for _, node in root.children.items():
            res.extend(self._dfs(node))
        return res
```

#### 644. Maximum Average Subarray II
```
class Solution:
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        # 这道题是说找出一个平均值数组 要求这个数组长度大于等于k
        # 而且平均值最大
        # 基本思路：平均值一定小于等于数组中的最大值，并且大于等于数组中的最小值
        low, high = min(nums), max(nums)
        while high - low > 1e-7:
            mid = (low + high) / 2
            # 这是表示能找到一个长度大于k的子数组
            # 该子数组的和大于mid
            # 说明当前的mid小了，可以增大
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

#### 647. Palindromic Substrings
```
class Solution:
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        # 这道题是问有多少个substring
        # http://www.cnblogs.com/grandyang/p/7404777.html
        # 在s[i]和s[j]相等这个条件下，去判断[i...j]是否是回文的时候，i和j的位置关系很重要，
        # 如果i和j相等了，那么dp[i][j]肯定是true；
        # 如果i和j是相邻的，那么dp[i][j]也是true；
        # 如果i和j中间只有一个字符，那么dp[i][j]还是true；
        # 如果中间有多余一个字符存在，那么我们需要看dp[i+1][j-1]是否为true，若为true，那么dp[i][j]就是true。
        # 赋值dp[i][j]后，如果其为true，结果res自增1
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
                # 核心：递推方程
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
        # 这道题是说以nums中最大的数字为头结点
        # 递归生成新树
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

#### 655. Print Binary Tree
```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def printTree(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[str]]
        """
        # 典型分治法思维
        h = self._get_height(root)
        w = 2 ** h - 1
        res = [[''] * w for _ in range(h)]
        self._dfs(root, res, curr_h=0, height=h, start=0, end=w - 1)
        return res
    
    def _dfs(self, root, res, curr_h, height, start, end):
        if not root or curr_h == height:
            return
        
        res[curr_h][(start + end) // 2] = str(root.val)
        self._dfs(root.left, res, curr_h + 1, height, start, (start + end) // 2)
        self._dfs(root.right, res, curr_h + 1, height, (start + end) // 2 + 1, end)
    
    def _get_height(self, root):
        if not root:
            return 0
        return 1 + max(
            self._get_height(root.left), 
            self._get_height(root.right),
        )
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
        # 这道题是说arr是sorted
        # 找出arr中k个跟x值最接近的
        # 好题！多看，注意理解这里的思路
        # 应该想到最终的返回的数组也是有序的
        res = arr[:]
        while len(res) > k:
            if x - res[0] <= res[-1] - x:
                # 说明最右边的数字和目标差距太大
                # 移除最后一位
                res.pop()
            else:
                # 反之移除第一位
                res.pop(0)
        
        return res
```

#### 659. Split Array into Consecutive Subsequences
```
from collections import defaultdict

class Solution:
    def isPossible(self, nums: 'List[int]') -> 'bool':
        freq = defaultdict(int)
        need = defaultdict(int)
        
        for num in nums:
            freq[num] += 1

        for num in nums:
            # 表示此时的num已经被前面用过了（凑成对儿了）
            if freq[num] == 0:
                continue
            # 作为某个连儿的结尾
            elif need[num] > 0:
                need[num] -= 1
                need[num + 1] += 1
                freq[num] -= 1
            # 作为某个新三连儿的开头
            elif freq[num + 1] > 0 and freq[num + 2] > 0:
                freq[num] -= 1
                freq[num + 1] -= 1
                freq[num + 2] -= 1
                need[num + 3] += 1
            else:
                # 此时说明出现了一个数字
                # 这个数字不能和任何的其他数字凑成连儿
                # 是个多余的 所以可以直接返回False
                return False
        
        return True
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
        1	2	3
        2	4	6
        3	6	9
        for m = n = 3, k = 5
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
        # 这道题是问num中交换一次可以获得的最大数字
        num_list = list(int(i) for i in str(num))
        n = len(num_list)

        # 核心之一:记录下每个数字最后出现的位置
        # last是指num_list中每个数字最后出现的index
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
        # 这道题是求最长递增子序列的个数
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

#### 676. Implement Magic Dictionary
```
from collections import defaultdict

class MagicDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._mapping = defaultdict(list)

    def buildDict(self, words: 'List[str]') -> 'None':
        """
        Build a dictionary through a list of words
        """
        for word in words:
            self._mapping[len(word)].append(word)

    def search(self, target: 'str') -> 'bool':
        """
        Returns if there is any word in the trie that equals to the given 
        word after modifying exactly one character
        """
        if len(target) not in self._mapping:
            return False

        for word in self._mapping[len(target)]:
            diff_count = 0
            i = 0
            # 模拟法
            # 计数有多少个不一样的字符
            while i < len(word):
                if word[i] != target[i]:
                    diff_count += 1
                    if diff_count == 2:
                        break
                i += 1
            if i == len(word) and diff_count == 1:
                return True
        
        return False

# Your MagicDictionary object will be instantiated and called as such:
# obj = MagicDictionary()
# obj.buildDict(dict)
# param_2 = obj.search(word)
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
        # 如果此时计数器大于0了，我们暂时不能确定这个string非法
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
        # 这道题是问nums能不能组成24
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
                # 将nums[j]的值合并到nums[i]上去
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
            res.append(1.0 * a / b)
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
        # 这道题是允许删除一个字符
        # 判断s还能不能是palindrome
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
        # 这道题是说只使用time里的数字，找出下一个时间
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
                   # for digit in divmod(hour, 10))
                   # for hour in divmod(curr, 60):
                   # 但是由于这里的'hour'第一次使用时候还没有定义
                   # 所以会报错
                   # 这里需要将这两句反过来
                   # 跟one-liner if-else不太一样
                   # res[0] if res else -1
                   # 会先去检查if的条件满不满足再去取res[0]
                   # 所以不会出现越界问题
                   for hour in divmod(curr, 60)
                   for digit in divmod(hour, 10)):
                return '{:02d}:{:02d}'.format(*divmod(curr, 60))
```

#### 684. Redundant Connection
```
from collections import defaultdict

class Solution:
    def findRedundantConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        graph = defaultdict(set)
        for v1, v2 in edges:
            # 第一次进入递归
            # 选一个肯定不会在edges中出现的点就好了
            if self._has_cycle(v1, v2, graph, root=-1):
                return [v1, v2]
            graph[v1].add(v2)
            graph[v2].add(v1)
        
        return []
    
    # has_cycle的定义就是当添加v1 v2这条边的时候
    # 图中会不会出现环
    def _has_cycle(self, v1, v2, graph, root):
        if v2 in graph[v1]:
            return True

        # 具体思路就是不停的递归下去，看是否有某个子孙后代就是v2
        for nei in graph[v1]:
            # 避免在1 -> 2和2 -> 1之间出现死循环
            if root == nei:
                continue
            if self._has_cycle(nei, v2, graph, root=v1):
                return True
        
        return False
```

#### 685. Redundant Connection II
```
# 理解的不好！！
class Solution:
    def findRedundantDirectedConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        # 和I不一样的地方在于这道题是有向图
        # 点是从1到N的
        # 正常的树N个点只有N - 1条边
        # 这道题要考虑三种情况：
        # 第一种：无环，但是有结点入度为2的结点
        # 第二种：有环，没有入度为2的结点
        # 第三种：有环，且有入度为2的结点
        n = len(edges)
        parents = [0] * (n + 1)
        candidate1 = []
        candidate2 = []
        
        # step 1: 检查是否存在某个点有两个parents(即入度为2)
        for i in range(n):
            start, end = edges[i]
            # 说明end这个点不在parents中
            if parents[end] == 0:
                parents[end] = start
            else:
                # 此时说明发现了一个入度为2的点
                # candidate1是老边
                candidate1 = [parents[end], end]
                # cnadidate2是新边
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

#### 686. Repeated String Match
```
class Solution:
    def repeatedStringMatch(self, A: 'str', B: 'str') -> 'int':
        # 感觉这里的判断是对的 答案有点问题？？
        # if len(A) > len(B):
        #     return -1
        
        cnt = 1
        new_A = A
        while len(new_A) < len(B):
            new_A += A
            cnt += 1
        
        pos = new_A.find(B)
        if pos != -1:
            return cnt
        
        new_A += A
        pos = new_A.find(B)
        if pos != -1:
            return cnt + 1
        
        return -1
```

#### 687. Longest Univalue Path
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def longestUnivaluePath(self, root: 'TreeNode') -> 'int':
        if not root:
            return 0
        
        not_contain_root = max(
            self.longestUnivaluePath(root.left),
            self.longestUnivaluePath(root.right),
        )
        
        contain_root = self._dfs(root.val, root.left) + self._dfs(root.val, root.right)
        return max(not_contain_root, contain_root)
    
    def _dfs(self, target, node):
        if not node or node.val != target:
            return 0
        
        return 1 + max(self._dfs(target, node.left), self._dfs(target, node.right))
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
        # 核心：我们其实将步骤这个维度当成了时间维度在不停更新
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

#### 689. Maximum Sum of 3 Non-Overlapping Subarrays
```
class Solution:
    def maxSumOfThreeSubarrays(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # 核心思想就是分段做
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
from collections import deque

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
        queue = deque([(i, j)])
        visited[i][j] = True

        res = 1
        while queue:
            curr_i, curr_j = queue.popleft()
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
        return self._dfs(nums, k, target, start=0, curr_sum=0, visited=visited)
    
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
        
        # 一定要pre后面插入
        # 因为pre要么是链表中间某个点(此时curr在后面)
        # 要么是结尾(此时curr是开头)
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
        # 题目要求nums中所有乘积小于k的子数组的个数
        # 其中k大于等于0，nums全正数
        # 则如果k小于1肯定是没有乘积为0的
        if k == 0:
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
            # 考虑此时实际上加上的是所有l和r之间以r为结尾的子数组
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

#### 715. Range Module
```
from bisect import bisect
from bisect import bisect_left

class RangeModule:

    def __init__(self):
        self.X = [0, 10 ** 9]
        self.track = [False] * 2
    
    def _index(self, x):
        i = bisect_left(self.X, x)
        if x != self.X[i]:
            self.X.insert(i, x)
            self.track.insert(i, self.track[i - 1])
        return i
        
    def addRange(self, left, right, track=True):
        """
        :type left: int
        :type right: int
        :rtype: void
        """
        i = self._index(left)
        j = self._index(right)
        # 相当于合并了X里面的元素
        self.X[i:j] = [left]
        self.track[i:j] = [track]
        
    def queryRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: bool
        """
        i = bisect(self.X, left) - 1
        j = bisect_left(self.X, right)
        return all(self.track[i:j])

    def removeRange(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: void
        """
        self.addRange(left, right, False)


# Your RangeModule object will be instantiated and called as such:
# obj = RangeModule()
# obj.addRange(left,right)
# param_2 = obj.queryRange(left,right)
# obj.removeRange(left,right)
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
        # 注意这里不能用while
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
        # 这道题是说
        # accounts里第一个元素是名字，其他都是邮箱
        # 如果某两个accounts的邮箱出现了交集，那么这两个account就是同一个人的
        # 这道题用accounts数组的index作为accounts的id
        self.initialize(len(accounts))
        email_to_ids = self.get_email_to_ids(accounts)
        
        for email, ids in email_to_ids.items():
            root_id = ids[0]
            for _id in ids[1:]:
                # 说明这些id都和root_id一样，属于同一个账户的
                self.union(root_id, _id)
        
        id_to_email_set = self.get_id_to_email_set(accounts)
        merged_accounts = []
        for user_id, email_set in id_to_email_set.items():
            merged_accounts.append([
                # 最后再根据id（因为id是index数字）再取出email
                accounts[user_id][0],
                *sorted(email_set),
            ])
        return merged_accounts
    
    def initialize(self, number_of_total_accounts):
        self.father = dict()
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
        # 并查集的核心！！！必背
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
        in_comment = False
        res = []
        for line in source:
            i = 0
            if not in_comment:
                # 只有当前不在comment（块）中
                # 才新建一个newline
                # 否则沿用老的
                newline = []
            while i < len(line):
                # 表示一个comment（块）开始
                if line[i:i + 2] == '/*' and not in_comment:
                    in_comment = True
                    i += 2
                # 表示一个comment（块）结束
                # 此时为何不需要break？
                # 因为/**/形式的注释可能出现在行中
                # 比如`int a; /*this is a demo*/ int b;`
                elif line[i:i + 2] == '*/' and in_comment:
                    in_comment = False
                    i += 2
                # 表示一个行级注释开始
                # 后面的肯定不用考虑
                elif not in_comment and line[i:i + 2] == '//':
                    break
                else:
                    if not in_comment:
                        newline.append(line[i])
                    i += 1
            if newline and not in_comment:
                res.append(''.join(newline))
        return res
```

#### 723. Candy Crush
```
class Solution:
    def candyCrush(self, board: 'List[List[int]]') -> 'List[List[int]]':
        if not board or not board[0]:
            return board
        
        # 核心思路就是先找该消除的点
        # 然后将这些该消除的点都置为0
        # 然后逐列将0移动到顶端
        m, n = len(board), len(board[0])
        while True:
            del_pos = []
            for i in range(m):
                for j in range(n):
                    if board[i][j] == 0:
                        continue
                    x0 = x1 = i
                    y0 = y1 = j
                    while x0 >= 0 and x0 > i - 3 and board[x0][j] == board[i][j]:
                        x0 -= 1
                    while x1 < m and x1 < i + 3 and board[x1][j] == board[i][j]:
                        x1 += 1
                    while y0 >= 0 and y0 > j - 3 and board[i][y0] == board[i][j]:
                        y0 -= 1
                    while y1 < n and y1 < j + 3 and board[i][y1] == board[i][j]:
                        y1 += 1
                    if x1 - x0 > 3 or y1 - y0 > 3:
                        del_pos.append((i, j))
            if not del_pos:
                break
            for i, j in del_pos:
                board[i][j] = 0
            for j in range(n):
                t = m - 1
                for i in range(m - 1, -1, -1):
                    if board[i][j] != 0:
                        board[t][j], board[i][j] = board[i][j], board[t][j]
                        t -= 1
        return board
```

#### 724. Find Pivot Index
```
class Solution:
    def pivotIndex(self, nums: 'List[int]') -> 'int':
        total = sum(nums)
        curr_sum = 0
        for i, num in enumerate(nums):
            if total - num == 2 * curr_sum:
                return i
            curr_sum += num
        return -1
```

#### 730. Count Different Palindromic Subsequences
```
class Solution:
    def countPalindromicSubsequences(self, S):
        """
        :type S: str
        :rtype: int
        """
        # 这道题是求在一个有重复字符的字符串里找出回文子序列的个数
        # http://www.cnblogs.com/grandyang/p/7942040.html
        n = len(S)
        M = 10 ** 9 + 7
        
        # dp[i][j]表示i到j之间不同的回文子序列个数
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
                    # 实际上此时s[i] == s[j]
                    while left <= right and S[right] != S[j]:
                        right -= 1
                    if left > right:
                        # 说明此时i和j之间没有和S[i]相等的字符
                        # a - bcd - a
                        # 为什么要乘以2？
                        # 因为里面的字符可以单独存在，也可以和外层一起出现，所以答案相当于double了
                        # 注意是回文子序列
                        # 加2是因为只算外层的a和aa这两种情况也要算上
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 2
                    elif left == right:
                        # 说明此时i和j之间只有一个和S[i]相等的字符
                        # a - bad - a
                        # 加1是因为单个字符a已经包括在里面(dp[i + 1][j - 1])了，所以只加aa
                        dp[i][j] = dp[i + 1][j - 1] * 2 + 1
                    else:
                        # 此时是a - d - (a - bc - a) - h - a的情况
                        # left指向从左到右第二个a，right指向从右到左第二个a
                        # 为什么要减去dp[left + 1][right - 1]?
                        # 因为当包括最外层aa的情况时候，会出现重复
                        # 比如当不选那个d和h的时候，通常外层的aa也同样有两种情况：选或者不选
                        # 同样的此时里面的aa也有两种选项：选或者不选
                        # 通常应该有4种答案：外层a里层a里层a外层a 里层a里层a 外层a外层a 和空
                        # 此时里层a里层a 外层a外层a这两种情况都是aa，所以相当于多余计算了
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
        # 这道题是找S中最短的子串W，使得T是W的子序列
        m, n = len(S), len(T)
        start = -1
        min_len = 2 ** 31 - 1
        # 这道题比较难想的就是dp的状态定义，而且这道题的dp定义确实比较特殊
        # 核心之一：
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
                    # 核心之二：
                    # 如果此时S[i] == T[j](为了方便理解，这里假设1-index)
                    # 则S中前i个字符包含T中前j个字符的在S中的起始位置
                    # 实际上跟S中前i - 1个字符包含T中前j - 1个字符在S中的起始位置是一样的
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

#### 729. My Calendar I	
```
class MyCalendar:

    def __init__(self):
        self._events = []

    def book(self, start: 'int', end: 'int') -> 'bool':
        for booked_start, booked_end in self._events:
            # 重点！！
            # 必背的条件
            if booked_start < end and booked_end > start:
                return False
        # 这道题可以通过自动有序的数据结构（TreeMap，AVL）降低复杂度
        # 这道题不要求正确的插入时间
        # 就是说events可以是无序的
        self._events.append([start, end])
        return True


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)
```

#### 731. My Calendar II
```
class MyCalendarTwo:

    def __init__(self):
        # 注意审题！！！
        # 跟I不一样的是这道题是说重叠的区域上最多有两个event
        # 用两个set
        # s1存的是完整区间的集合
        # s2存的是重叠区间（即已经有两个event在这个区间里了）
        # 所以如果新的start, end在这个区域里
        # 下面的book函数之间返回False
        self._s1 = set()
        # s2存的就是当前有重叠的区域（已经有两个event在了）
        # 所以如果新event和任何s2中的区域重叠
        # 直接返回False即可
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
                # 核心之一：
                # 重叠的区域就是两个start中的最大值和两个end中的最小值
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
        # 要求返回当前日历里最多重叠事件的个数
        self._freq = defaultdict(int)

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: int
        """
        # 好题！仔细体会
        # 这里的做法和370. Range Addition以及meeting rooms比较像
        # 用来计算在某个区间里的个数
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
        # 这道题是说将旧颜色以及相邻的旧颜色变成新颜色
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
                # 这里别忘了如果新点不是旧颜色（old_color）
                # 或者已经是新颜色（new_color）时候，都不需要入队
                # 尤其是后者，如果已经是新颜色还入队的话
                # 会使得死循环(反复将这个点入队只因为它不是旧颜色)
                if not 0 <= newi < m or not 0 <= newj < n or \
                    image[newi][newj] != old_color or \
                    image[newi][newj] == new_color:
                    continue
                queue.append((newi, newj))
                # 也相当于起到了visit数组的作用
                image[newi][newj] = new_color
        
        return image
```

#### 734. Sentence Similarity
```
class Solution:
    def areSentencesSimilar(self, words1, words2, pairs):
        """
        :type words1: List[str]
        :type words2: List[str]
        :type pairs: List[List[str]]
        :rtype: bool
        """
        if len(words1) != len(words2):
            return False

        pairs_set = set()
        for p1, p2 in pairs:
            pairs_set.add((p1, p2))
            pairs_set.add((p2, p1))
        
        n = len(words1)
        for i in range(n):
            w1, w2 = words1[i], words2[i]
            pairs_set.add((w1, w1))
            pairs_set.add((w2, w2))
            if (w1, w2) not in pairs_set and (w2, w1) not in pairs_set:
                return False
        
        return True
```

#### 737. Sentence Similarity II
```
from collections import defaultdict

class UnionFind:
    def __init__(self, pairs):
        self.father = {}
        for w1, w2 in pairs:
            self.father[w1] = w1
            self.father[w2] = w2

    def find(self, word):
        if word not in self.father:
            return
        root_word = self.father[word]
        if root_word != word:
            self.father[word] = self.find(self.father[word])
        return self.father[word]

    def union(self, word1, word2):
        root1 = self.find(word1)
        root2 = self.find(word2)
        if root1 != root2:
            self.father[root1] = root2

class Solution:
    def areSentencesSimilarTwo(self, words1, words2, pairs):
        if len(words1) != len(words2):
            return False

        uf = UnionFind(pairs)
        for p1, p2 in pairs:
            uf.union(p1, p2)
        
        for w1, w2 in zip(words1, words2):
            if w1 == w2:
                continue
            root1 = uf.find(w1)
            root2 = uf.find(w2)
            if not root1 or not root2:
                return False
            if not root1 == root2:
                return False

        return True

#       # DFS解法
        if len(words1) != len(words2):
#             return False

#         graph = defaultdict(set)
#         for p1, p2 in pairs:
#             graph[p1].add(p2)
#             graph[p2].add(p1)
        
#         for w1, w2 in zip(words1, words2):
#             stack = [w1]
#             visited = set([w1])
#             while stack:
#                 word = stack.pop()
#                 if word == w2:
#                     break
#                 for nei in graph[word]:
#                     if nei not in visited:
#                         stack.append(nei)
#                         visited.add(nei)
#             else:
#                 return False
        
#         return True
```

#### 739. Daily Temperatures
```
class Solution:
    def dailyTemperatures(self, T: 'List[int]') -> 'List[int]':
        # 这道题就是说T[i]之后第几个数字比他大
        # 核心思路：单调递减栈
        # 实际上就是tapping water题的变种
        # 好题！多多体会
        n = len(T)
        res = [0] * n
        desc_stack = []
        for curr_day in range(n):
            # 相当于将T[curr_day]当成一个标准
            # 去把之前所有遍历过的日子update
            while desc_stack and T[curr_day] > T[desc_stack[-1]]:
                last_day = desc_stack.pop()
                res[last_day] = curr_day - last_day
            desc_stack.append(curr_day)
        return res
```

#### 742. Closest Leaf in a Binary Tree
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
    def findClosestLeaf(self, root: 'TreeNode', k: 'int') -> 'int':
        # 这道题是说k是root中的某个节点
        # 树中的哪个叶子节点到k的距离最小
        # 基本思路是先将树变成一个无向图
        # 然后以这个k节点为开头做bfs直到某个叶子节点
        graph = defaultdict(list)
        self._dfs(root, graph, parent_node=None)
        visited = set()
        queue = deque()
        for node in graph:
            if node and node.val == k:
                queue.append(node)
                visited.add(node)
                break
        
        while queue:
            node = queue.popleft()
            if node:
                if len(graph[node]) == 1:
                    # 说明此时的node是一个叶子节点
                    return node.val
                for nei in graph[node]:
                    if nei not in visited:
                        queue.append(nei)
                        visited.add(nei)
    
    def _dfs(self, node, graph, parent_node):
        if node:
            graph[node].append(parent_node)
            graph[parent_node].append(node)
            self._dfs(node.left, graph, node)
            self._dfs(node.right, graph, node)
```

#### 744. Find Smallest Letter Greater Than Target
```
class Solution:
    def nextGreatestLetter(self, letters: 'List[str]', target: 'str') -> 'str':
        start, end = 0, len(letters) - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if letters[mid] <= target:
                start = mid + 1
            else:
                end = mid
        if letters[start] > target:
            return letters[start]
        if letters[end] > target:
            return letters[end]
        return letters[0]
```

#### 745. Prefix and Suffix Search
```
class WordFilter:

    def __init__(self, words):
        """
        :type words: List[str]
        返回同时具有前缀 prefix 和后缀suffix 的词的最大权重。
        如果没有这样的词，返回 -1。
        """
        self._map = dict()
        for weight, word in enumerate(words):
            for i in range(len(word) + 1):
                prefix = word[:i]
                for j in range(len(word) + 1):
                    suffix = word[j:]
                    # 由于我们是按照顺序遍历word的
                    # 所以出现在后面具有一样key的词一定有更大的权重
                    self._map[prefix + '#' + suffix] = weight

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

#### 750. Number Of Corner Rectangles
```
class Solution:
    def countCornerRectangles(self, grid: 'List[List[int]]') -> 'int':
        m, n = len(grid), len(grid[0])
        # 核心思路用两根线扫描
        res = 0
        for i in range(m):
            for j in range(i + 1, m):
                cnt = 0
                for k in range(n):
                    if grid[i][k] == grid[j][k] == 1:
                        cnt += 1
                # 有n个格子
                # 这n个格子就能组成1 + 2 + ... + n = n * (n + 1) / 2个矩形
                res += cnt * (cnt - 1) // 2
        return res
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
        # 换句话说，每个CIDR块都尽量覆盖最多的ips
        # 而且要注意，这个CIDR块里的地址是以原始的ip为起始地址往上增加的
        # 比如原始ip是192.168.10.100，要求覆盖50个ip
        # 则这个ip范围就是192.168.10.100到192.168.10.149
        # 也就是说我们只能把最后的0变成1，不能把1变成0

        # A.B.C.D/N就是一个CIDR块，这里的N是前缀个数(0-32)
        # 当然N越小，可以覆盖的区域就越大
        # 这道题是问要覆盖n个ip，最少需要多少个CIDR块
        ips = ip.split('.')
        x = 0
        for i in range(len(ips)):
            x = x * 256 + int(ips[i])
        
        res = []
        while n > 0:
            # 核心之一：
            # x & -x 能求出x中最后一位1的位置
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
        # 密码有n位，每位上的数字都是0到k - 1（0 <= k <= 10）
        # 密码箱的工作方式是记录最后n位，只要最后n位match了，就会解开
        # 这道题要求找保证输入以后能打开的最短的密码串
        if k < 0:
            return -1
        
        # 密码每位数字就是0-K之间（K是0到9）
        # 最短的密码也应该有n位
        res = '0' * n
        # 注意：这里要写set([res])
        # 如果写成set(res)就相当于把res这个字符串里的每个字母都加到set里了
        # 因为res本身就是一个iterable
        visited = set([res])
        
        # 遍历k的n次方
        for i in range(pow(k, n)):
            # 取出当前密码的倒数n - 1位
            # 再遍历加上一个备选
            # 得到一个新的密码组合
            # 如果这个密码组合没有出现过
            # 就直接放入到res中，并同时放入到visit set中
            # 核心之一：
            # 为什么要从后往前？因为这样能保证每次遍历的覆盖最多
            # 最终能保证拼接起来的字符串最短
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
        # 这道题输入是n个员工的不重叠的并且排过序的schedule
        # 求所有的空闲时间间隔
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
                # 典型贪心思路
                prev_end = max(prev_end, curr_end)
        
        return res
```

#### 760. Find Anagram Mappings
```
from collections import defaultdict

class Solution:
    def anagramMappings(self, A: 'List[int]', B: 'List[int]') -> 'List[int]':
        B_mapping = defaultdict(list)
        for inx, b in enumerate(B):
            B_mapping[b].append(inx)
        
        res = []
        for a in A:
            res.append(B_mapping[a].pop())
        
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
        # n是偶数(表示n // 2对儿couple)
        # 求最少的交换次数
        # 使得每对儿couple坐在一起
        # (1, 2)是一对儿，(3, 4)是一对儿
        # 这道题思路就是暴力循环
        res = 0
        n = len(row)
        for i in range(0, n, 2):
            # 每次的i和i+1 suppose是一对儿
            # 核心之一：
            # 非常棒的trick：比如2和3应该是一对儿
            # 则2^1正好是3，而3^1正好是2
            # 偶数异或和奇数异或的问题
            if row[i + 1] == row[i] ^ 1:
                continue
            for j in range(i + 1, n):
                if row[j] == row[i] ^ 1:
                    row[j], row[i + 1] = row[i + 1], row[j]
                    res += 1
                    break
        return res
```

#### 766. Toeplitz Matrix
```
class Solution:
    def isToeplitzMatrix(self, matrix: 'List[List[int]]') -> 'bool':
        if not matrix or not matrix[0]:
            return False
        
        m, n = len(matrix), len(matrix[0])
        for i in range(m - 1):
            for j in range(n - 1):
                if matrix[i][j] != matrix[i + 1][j + 1]:
                    return False
        
        return True
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
        # 重构这个字符串，使得相同的字符不会相邻
        # 如果做不到返回空
        n = len(s)
        ch_counts = [(counts, ch) for ch, counts in Counter(s).items()]

        sorted_s = []
        for counts, ch in sorted(ch_counts):
            if counts > (n + 1) // 2:
                return ''
            # 'a' * 5 = 'aaaaa'
            sorted_s += ch * counts
        
        # python string是immutable
        # 先转成字符串数组再进行interleaving的操作
        res_list = [None] * n
        sorted_s_list = list(sorted_s)
        # 核心！！！
        # res_list从零开始每隔两位，从1开始每隔两位
        # 分别用sorted_s_list的前后两半来填充
        # 支持slicing的语言都可以这么做
        # 注意：5 // 2 == 2, 相当于sorted_s_list[2:]的长度是3
        # 同时res_list[::2]的长度也是等于res_list[1::2]的长度或者加1
        res_list[::2], res_list[1::2] = sorted_s_list[n // 2:], sorted_s_list[:n // 2]
        return ''.join(res_list)
```

#### 769. Max Chunks To Make Sorted
```
class Solution:
    def maxChunksToSorted(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        max_num = -2 ** 31
        res = 0
        for i, num in enumerate(arr):
            max_num = max(max_num, num)
            # 如果此时在当前i这个索引之前出现的最大数字正好和i相等
            # 说明我们找到了一个块儿
            # 这个块儿可以以i结尾断开
            # res可以直接加1
            # 怎么想到的？？
            # 挺难想的，但是面试时候可以通过观察数据得出结论
            # 也可以这么想：
            # 当遍历到i的时候，有两种选择，能断开还是不能断开？
            # 如果能断开，说明断开以后的这个数组里的最大的数字肯定属于这个区间
            # 不会落到右边的区间
            # 所以当之前的最大数字正好和当前索引相等的时候，就是能断开的时候了
            if i == max_num:
                res += 1
        return res
```

#### 772. Basic Calculator III
```
class Solution:
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        这道题是既有加减乘除又有括号的
        """
        # 很好的题目！
        # 多看！！！
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
                # 此时已经出现了一个左括号了，所以cnt = 1
                cnt = 1
                j = i + 1
                while j < n:
                    if s[j] == '(':
                        cnt += 1
                    if s[j] == ')':
                        cnt -= 1
                    if cnt == 0:
                        break
                    j += 1
                # j是最后右括号的位置
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
                # 如果此时ch是乘除
                # 不急着将curr_res放入到res中
                # 因为需要后面的数字乘除完了再放入到res中（因为乘除优先级高）
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
        # 给定一个board的样子
        # 凑成target需要的最少的步数
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
        # queue里面放的是当前的状态
        # 很重要的思路！！！
        queue.append(start)
        visited = set()

        res = 0
        while queue:
            # 注意这里重点要分层遍历才能更新res！！！！
            # 所以要for循环而不是单纯的每次popleft
            q_len = len(queue)
            for _ in range(q_len):
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
        # 这道题说给了我们n个加油站，两两之间相距不同的距离
        # 然后我们可以在任意地方新加K个加油站，问能使得任意两个加油站之间的最大距离的最小值是多少
        # 太绕了，实际上就是去minimize任意两个加油站之间的最大距离
        # 这道题正解是二分法
        # 比较难想
        # 二分的是**最小的任意两个加油站之间的最大距离**(
        # 最大距离的最小值，其实就是题目要求解的值)
        # 好处是这道题给定了上下界
        # 因为加油站的位置是在0到10 ** 8之间
        # 所以任意两个加油站之间的距离就在0到10 ** 8之间
        # 我们需要在这个范围内找到一个最小值，
        # 以这个最小间隔，在现有的两两加油站之间插入一共K个新油站
        
        # 不要被题目的开头minimize这个词误导了
        # 这道题实际上是寻找所有能放的下K个加油站的解里面的最大值
        l, r = 0, 1e8
        while r - l > 1e-6:
            mid = l + (r - l) / 2
            # 说明以现有的mid为interval，在stations
            # 两两之间插入油站的话做不到插入K个新油站
            # 此时说明间隔太大了，需要缩小上界
            if self._helper(stations, K, mid):
                r = mid
            else:
                l = mid
        
        return l
    
    # 核心重点：helper求的是在两两相邻的加油站之间，
    # 能够存放下多少个以当前mid为间隔的加油站！！！
    # 则如果以当前的间隔interval能插入的新加油站数量小于K(放不下K个加油站)
    # 说明间隔太大了；反之则说明间隔太小了
    def _helper(self, stations, K, interval):
        cnt = 0
        for i in range(len(stations) - 1):
            # 这里求的是能放得下多少个加油站
            cnt += floor((stations[i + 1] - stations[i]) / interval)
        return cnt <= K
```

#### 776. Split BST
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def splitBST(self, root: 'TreeNode', V: 'int') -> 'List[TreeNode]':
        res = [None, None]
        if not root:
            return res
        
        # 非常好的一道BST递归题目
        # 遇到树的问题尽量先去想递归
        if root.val <= V:
            # 思路就是假设我们有一个来自未来的函数
            # 能够帮助我们在root.right上分割
            # 则返回的结果我们在这个函数里处理
            # 一直返回的res[0]就是小树
            # 直接把这个小树加到当前树的右子树上，而且当前树就是新的小树
            res = self.splitBST(root.right, V)
            root.right = res[0]
            res[0] = root
        else:
            res = self.splitBST(root.left, V)
            root.left = res[1]
            res[1] = root
        
        return res
```

#### 777. Swap Adjacent in LR String
```
class Solution:
    def canTransform(self, start, end):
        """
        :type start: str
        :type end: str
        :rtype: bool
        """
        # 这道题是说XL -> LX，RX -> XR
        # L永远只能往左移，R永远只能往右移
        if len(start) != len(end):
            return False

        n = len(start)
        i = j = 0
        while i < n and j < n:
            while i < n and start[i] == 'X':
                i += 1
            while j < n and end[j] == 'X':
                j += 1
            # 下面两个都是边界条件
            if i == n and j == n:
                return True
            if i == n or j == n:
                return False
            # 此时start[i]和end[j]必定要么都是L，要么都是R
            # 由于L R只能和X交换，他们之间的相对顺序是不变的
            # 所以如果此时它俩不等，一定是False
            if start[i] != end[j]:
                return False
            # 由于L只能往左移
            # 就是说start里的i要变成end里的j
            # start里的L只能和X交换往左移动
            # 所以如果此时i的位置小于j的位置
            # 往左移是不能到达end的
            # 所以直接返回False
            if start[i] == 'L' and i < j:
                return False
            # R的情况同理
            if start[i] == 'R' and i > j:
                return False
            i += 1
            j += 1
        
        return True
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
        # 这道题是问当前的graph是不是一个二分图
        colors = [0] * len(graph)
        for i in range(len(graph)):
            # 如果已经被染过色了，直接跳过
            if colors[i] != 0:
                continue
            queue = deque()
            queue.append(i)
            # 先染色为1
            colors[i] = 1
            while queue:
                curr = queue.popleft()
                for each in graph[curr]:
                    # 如果没染过色
                    # 就染色为另外一种颜色
                    if colors[each] == 0:
                        queue.append(each)
                        colors[each] = -1 * colors[curr]
                    else:
                        # 否则如果已经被染过色了而且跟相邻的颜色一样
                        # 就不是二分图，可以世界返回False
                        if colors[each] == colors[curr]:
                            return False

        return True
```

#### 787. Cheapest Flights Within K Stops
```
from collections import defaultdict
from heapq import heappop
from heapq import heappush

class Solution:
    def findCheapestPrice(self, n: 'int', flights: 'List[List[int]]', src: 'int', dst: 'int', K: 'int') -> 'int':
        # 这道题是说n个城市，最多k站，最小的花费从src到达dst
        # 拓扑排序，但是由于是求花费，用最小堆
        # 实际上这种做法是dijkstra
        graph = defaultdict(dict)
        for s, e, c in flights:
            graph[s][e] = c
        
        min_heap = []
        # [curr_cost, curr_k, curr_place]
        heappush(min_heap, [0, 0, src])
        
        # best_cost的key是一个tuple(k, place)
        # 指当前已经走了k站到达place的花费是多少(对应的value)
        best_cost = dict()
        while min_heap:
            curr_cost, curr_k, curr_place = heappop(min_heap)
            if curr_k > K + 1:
                continue
            # 核心之一：
            # 正常的dijkstra的dict的key只需要用node表示即可
            # 但是这里由于需要考虑多少站（K）
            # 所以要附加上curr_k作为key的一部分
            # dijkstra: 核心是（1）bfs，（2）用最小堆，（3）用dict.get方法设置default距离为无穷大
            if curr_cost > best_cost.get((curr_k, curr_place), 2 ** 31 - 1):
                continue
            
            if curr_place == dst:
                return curr_cost
            
            for next_place, next_cost in graph[curr_place].items():
                new_cost = curr_cost + next_cost
                if new_cost < best_cost.get((curr_k + 1, next_place), 2 ** 31 - 1):
                    heappush(min_heap, [new_cost, curr_k + 1, next_place])
                    best_cost[(curr_k + 1, next_place)] = new_cost
        
        return -1
```

#### 788. Rotated Digits
```
class Solution:
    def rotatedDigits(self, N: 'int') -> 'int':
        # dp[i]表示当前数字i的3中状态
        # dp[i] == 0表示数字i翻转后不合法
        # dp[i] == 1表示数字i翻转后和原来数字一样
        # dp[i] == 2表示数字i是一个好数字
        dp = [0] * (N + 1)
        res = 0
        for i in range(N + 1):
            if i < 10:
                if i in (0, 1, 8):
                    dp[i] = 1
                elif i in (2, 5, 6, 9):
                    dp[i] = 2
                    res += 1
            else:
                a, b = dp[i // 10], dp[i % 10]
                # 比如i=81
                # 则此时a = 81 // 10 = 8, b = 81 % 10 = 1
                # 则我们知道dp[8]一定是1，dp[1]也一定是1
                # 这两位翻转后不变还是81
                # 重要的审题：这里翻转180度是独立的每一位数字分别翻转
                # 而不是整体的180度！！！
                if a == 1 and b == 1:
                    dp[i] = 1
                elif a >= 1 and b >= 1:
                    dp[i] = 2
                    res += 1
        
        return res
```

#### 790. Domino and Tromino Tiling
```
class Solution:
    def numTilings(self, N: 'int') -> 'int':
        M = 1e9 + 7
        # dp[n]表示要堆成2 * n的board有多少种堆法
        # dp[n]是由之前的dp值组成的，其中dp[n-1]和dp[n-2]各自能贡献一种组成方式
        # 而dp[n-3]，一直到dp[0]，都能各自贡献两种组成方式，所以状态转移方程为：
        # dp[n] = dp[n-1] + dp[n-2] + 2 * (dp[n-3] + ... + dp[0])
        #       = dp[n-1] + dp[n-3] + dp[n-2] + dp[n-3] + 2 * (dp[n-4] + ... dp[0])
        #       = dp[n-1] + dp[n-3] + dp[n-1]
        #       = 2 * dp[n-1] + dp[n-3]
        if N <= 1:
            return 1
        dp = [0] * (N + 1)
        dp[0] = 1
        dp[1] = 1
        dp[2] = 2
        for i in range(3, N + 1):
            dp[i] = (2 * dp[i - 1] + dp[i - 3]) % M
        
        return int(dp[-1])
```

#### 791. Custom Sort String
```
from collections import defaultdict

class Solution:
    def customSortString(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: str
        """
        # 给定两个字符串S和T
        # 将T按照S的顺序重新排列
        # 这道题是运用hash map的好题
        # 巧妙的运用hash map记录个数
        # 再顺序遍历S保证顺序
        count = defaultdict(int)
        for ch in T:
            count[ch] += 1

        res = []
        # 按顺序遍历S就是能拿到S里字符的顺序
        # 因为这道题说了S中没有重复的字符
        for ch in S:
            # 这里是将T中同时也在S中出现的字符给先给添加到res中
            # 并将T自己的count置为0
            # 核心之一：
            # count里记录的是T的count
            # 但是当我们按照顺利遍历S的时候，也从这个hash里取这个出现的S中的字符
            # 在T中出现了多少次
            res.append(ch * count[ch])
            count[ch] = 0
        
        # 核心之二：
        # 别忘了T中还有些字符没有在S中出现过
        # 由于我们巧妙的上面遍历中count[ch] = 0
        # 所以就避开了对最终结果的影响
        for ch in count:
            res.append(ch * count[ch])
        
        # 这里不用filter掉''，因为空字符串join还是空字符串
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
        # 这道题给了我们一个字符串S，又给了一个单词数组，问我们数组有多少个单词是字符串S的子序列。
        # 核心思路：将words里的词的iterator加入一个map
        # 遍历一次S,每次遍历到的ch对应的iterator往后移动
        # 如果某个iterator全部跑完了，res加1
        # 这个iterator的思路要take out，非常巧妙
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
            # 注意这里一定要置为空
            # 相当于清零，因为在while中要重新添加iterator到相应的index上
            heads[inx] = []
            while prev:
                it = prev.pop()
                next_ch = next(it, None)
                if next_ch:
                    heads[ord(next_ch) - ord('a')].append(it)
                else:
                    # 此时说明iterator里已经全部都找到了
                    # next_ch为None了
                    # 而且当前的prev也一定是空，下一次while的判断中就会跳出
                    res += 1
        
        return res
```

#### 796. Rotate String
```
class Solution:
    def rotateString(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """
        if not A:
            return not B
        
        if len(A) != len(B):
            return False
        
        for i in range(len(A)):
            if A[i:] + A[:i] == B:
                return True
        
        return False
```

#### 799. Champagne Tower
```
class Solution:
    def champagneTower(self, poured: 'int', query_row: 'int', query_glass: 'int') -> 'float':
        # 基本思路就是一层一层处理
        dp = [[0] * 101 for _ in range(101)]
        dp[0][0] = poured
        for i in range(query_row + 1):
            for j in range(i + 1):
                if dp[i][j] >= 1:
                    dp[i + 1][j] += (dp[i][j] - 1) / 2
                    dp[i + 1][j + 1] += (dp[i][j] - 1) / 2
        
        return min(1, dp[query_row][query_glass])
```

#### 800. Similar RGB Color
```
class Solution:
    def similarRGB(self, color: 'str') -> 'str':
        # 这道题是说返回跟color最近的并且有shorthand的颜色
        # 输入的一定不是shorthand
        res = []
        for i in range(1, len(color), 2):
            res.append(self._get_close_color(color[i:i + 2]))
        return '#' + ''.join(res)
    
    def _get_close_color(self, ss):
        shorthands = [
            '00', '11', '22', '33',
            '44', '55', '66', '77',
            '88', '99', 'aa', 'bb',
            'cc', 'dd', 'ee', 'ff',
        ]
        
        return min(shorthands, key=lambda xx: abs(int(ss, 16) - int(xx, 16)))
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
        # 只有顶端的以及和不会掉落的砖头相连的砖头不会掉落
        # 问给定给一个hit的数组，返回每次hit掉落砖块的个数（不包括这个hit的砖块）
        # 核心：先hit打掉，再标记剩余不会掉的砖块为2
        res = [0] * len(hits)
        if not grid or not grid[0]:
            return []
        m, n = len(grid), len(grid[0])

        # 注意grid里只有0和1两种情况
        # 所以进行过下面的操作
        # 是不会生成新的砖块的（为1）
        # 只是将原有的砖块（为1的点）打掉（变成0）
        # 或者标记原来的empty点（值是0的点）为-1
        for i, j in hits:
            grid[i][j] -= 1
        
        # 先将第一行(index为0)里所有现存的砖块以及和它们相连的砖块给标记成2
        for i in range(n):
            self._dfs(0, i, grid)

        # 核心之一：从后往前考虑
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
            # 这步里是说此时这个砖块是衔接四周为1(重要！此时的grid[i][j]只能必须得是1，不能是2, 因为2是不会掉的)和顶部的纽带
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
    
    # 判断i j点的周围4个点是否和其他值为2的点连接过，或者本身就是顶层的点（不会掉落）
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
        # 这道题是问将A(全部是非负元素)中的元素分别放到B和C中，能不能让凑成的B和C有一样的平均值
        # dp定义：
        # 从A中拿出一些元素
        # 核心就是dp的定义
        # 键值key表示这些元素的sum和
        # dp的value表示这些元素的个数
        # 初始状态0的sum和不需要任何元素，因此value也为0
        dp = {0: 0}
        len_A, total_A = len(A), sum(A)
        
        for a in A:
            # sub_sum一定要从大到小遍历
            # 因为这道题有个限定A一定是非负元素
            # 而且当前要更新的的状态sub_sum + a
            # 是依赖于之前遍历过的状态sub_sum的
            # 换句话说大的状态是依赖于小的状态的，而a是全正数，则sub_sum是递增的
            # 所以要dp的key要反着遍历
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

#### 807. Max Increase to Keep City Skyline
```
class Solution:
    def maxIncreaseKeepingSkyline(self, grid: 'List[List[int]]') -> 'int':
        m, n = len(grid), len(grid[0])
        max_row = [0] * m
        max_col = [0] * n
        
        for i in range(m):
            for j in range(n):
                max_row[i] = max(max_row[i], grid[i][j])
                max_col[j] = max(max_col[j], grid[i][j])
        
        res = 0
        for i in range(m):
            for j in range(n):
                min_h = min(max_row[i], max_col[j])
                res += min_h - grid[i][j]
        
        return res
```

#### 809. Expressive Words
```
class Solution:
    def expressiveWords(self, S: 'str', words: 'List[str]') -> 'int':
        # 输入是一个words
        # 返回words中有多少个词可以拉伸为S
        res = 0
        m = len(S)
        for word in words:
            i = j = 0
            while i < m:
                if j < len(word) and S[i] == word[j]:
                    j += 1
                elif i > 0 and S[i] == S[i - 1] and i + 1 < m and S[i] == S[i + 1]:
                    i += 1
                elif not (i > 1 and S[i] == S[i - 1] and S[i] == S[i - 2]):
                    break
                i += 1
            if i == m and j == len(word):
                res += 1
        
        return res
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
        # routes[i]表示第i号bus能到达哪些车站
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

#### 817. Linked List Components
```
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def numComponents(self, head: 'ListNode', G: 'List[int]') -> 'int':
        # 这题是问G中有多少个相连的子链表
        res = 0
        node_set = set(G)
        
        curr = head
        while curr:
            if curr.val not in node_set:
                curr = curr.next
                continue
            
            # 此时说明找到了一个在G中的点
            # res可以加1
            res += 1
            
            # 再去找到下一个在G中的点作为下一次的curr
            # 因为当前这个点已经遍历完了
            while curr and curr.val in node_set:
                curr = curr.next
        
        return res
```

#### 818. Race Car
```
class Solution:
    def racecar(self, target):
        """
        :type target: int
        :rtype: int
        题意：
        遇到A，speed *= 2, 并且position += speed
        遇到R，speed = -1如果当前speed是1；speed = 1(注意不移动车子)如果
        当前speed已经是-1了
        给定最终的位置target，问最短的sequence（由A和R组成）能够到达target
        初始的第一个A使得车子从0到1
        """
        # 这道题最native的思路就是bfs
        # 假设我们最多最多有n条指令（其中每条指令就是从A或者R两种选择中选）
        # 然后把每次指令下能到达的candidate_target和target比较即可

        # dp[i]表示到达i点需要的最少指令
        # 注意：到达2这个点需要4条指令：AARA
        dp = [0, 1, 4] + [2 ** 31 - 1] * target
        for t in range(3, target + 1):
            k = t.bit_length()
            # 当前的t就是1111111的二进制表示
            # 如果t等于3， 7， 15， 31的话
            # 最短的指令就是AA, AAA, AAAA, AAAAA
            if t == 2 ** k - 1:
                dp[t] = k
                continue
            
            # 情况1：走2 ** (k - 1)的距离
            # 然后用一个R和m个A去往回走，走到一个能完美发射的点
            # 这时候已经走了(k - 1) + 1 + (m - 1) + 1 + 2步（即A**(k - 1)RA**(m - 1)R
            # 然后去dp里找剩余的步数t - 2 ** (k - 1) + 2 ** m
            # 说白了就是在没到target之前的某个位置停住，然后一路发射AAAAA冲到重点
            # 核心就是得找到这个完美停住的点，怎么停住？就是得先走k - 1步，然后一次R，继续走m步，然后再一次R
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
        # 只要不在下列的条件里，A就会发request给B：
        # age[B] <= 0.5 * age[A] + 7
        # age[B] > age[A]
        # age[B] > 100 && age[A] < 100
        # 问一共有多少条request
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

#### 833. Find And Replace in String
```
class Solution:
    def findReplaceString(self, S, indexes, sources, targets):
        """
        :type S: str
        :type indexes: List[int]
        :type sources: List[str]
        :type targets: List[str]
        :rtype: str
        """
        s = list(S)
        for inx, src, tgt in sorted(zip(indexes, sources, targets), reverse=True):
            if s[inx:inx + len(src)] == list(src):
                s = s[:inx] + list(tgt) + s[inx + len(src):]
        return ''.join(s)
```

#### 834. Sum of Distances in Tree
```
from collections import defaultdict

class Solution:
    def sumOfDistancesInTree(self, N: 'int', edges: 'List[List[int]]') -> 'List[int]':
        tree = defaultdict(set)
        # res[i]表示的是i点到其他所有点的距离总和
        res = [0] * N
        # count[i]表示以i为根一共有多少个子孙节点
        count = [0] * N
        for i, j in edges:
            tree[i].add(j)
            tree[j].add(i)
        
        self._dfs1(root=0, seen=set(), res=res, count=count, tree=tree)
        self._dfs2(root=0, N=N, seen=set(), res=res, count=count, tree=tree)
        
        return res
    
    def _dfs1(self, root, seen, res, count, tree):
        seen.add(root)
        for i in tree[root]:
            if i not in seen:
                self._dfs1(i, seen, res, count, tree)
                count[root] += count[i]
                # 核心之一：
                # i是root的孩子
                # 所以首先res[root]可以先加上root到i节点直接距离
                # 由于i一共有count[i]个子孙节点
                # 则每个子孙节点到root的距离都相当于i到root的距离加1
                # 则总体加起来的距离就是子孙节点的个数（count[i]）
                res[root] += res[i] + count[i]
        count[root] += 1
    
    def _dfs2(self, root, N, seen, res, count, tree):
        seen.add(root)
        for i in tree[root]:
            if i not in seen:
                res[i] = (res[root] - count[i]) + N - count[i]
                self._dfs2(i, N, seen, res, count, tree)
```

#### 835. Image Overlap
```
from collections import Counter

class Solution:
    def largestOverlap(self, A: 'List[List[int]]', B: 'List[List[int]]') -> 'int':
        n = len(A)
        
        ones_A = []
        ones_B = []
        for i in range(n * n):
            if A[i // n][i % n] == 1:
                ones_A.append(i // n * 100 + i % n)
            if B[i // n][i % n] == 1:
                ones_B.append(i // n * 100 + i % n)
        
        temp = []
        for i in ones_A:
            for j in ones_B:
                # i - j代表坐标平移向量
                # 则最终最多的相同的平移向量就是平移后重叠的1个个数
                temp.append(i - j)
        
        counter = Counter(temp)
        if counter.most_common(1):
            return counter.most_common(1)[0][1]
        return 0
```

#### 837. New 21 Game
```
class Solution:
    def new21Game(self, N: 'int', K: 'int', W: 'int') -> 'float':
        # 这道题是说每次抽牌的值都在1到W之间
        # 当抽到总和大于等于K的时候停止
        # 问此时总和小于等于N的概率
        if K == 0:
            return 1

        # 这个条件是说每次刚刚超过或者等于K的时候，总和肯定是小于N的
        # 换句话说，W（作为抽牌的上限）的值一定小于等于N - K
        if W <= N - K:
            return 1
        
        # dp[i]的含义是得分为i的概率
        # dp[i]又应该为前W个的dp的平均值
        # 比如W=10，那么dp[20]=(dp[10] + dp[11] + ...dp[19]) / W
        dp = [0] * (N + 1)
        dp[0] = 1
        
        # W_sum表示前W个dp之和
        W_sum = 1.0
        for i in range(1, N + 1):
            dp[i] = W_sum / W
            if i < K:
                W_sum += dp[i]
            # 假设K=15，W=10
            # 在上面的if中dp[20]=(dp[10] + ... + dp[14]) / W
            # 因为我们拿到15以后不能再拿了
            if i >= W:
                # 这里就像窗口一样
                # 这个窗口的size就是W
                # 开始时候起点就是0
                # 当终点大于等于W的时候
                # 就要把开头给减掉
                W_sum -= dp[i - W]

        return sum(dp[K:])
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
from collections import Counter
from itertools import permutations

class Solution:
    def findSecretWord(self, wordlist, master):
        """
        :type wordlist: List[Str]
        :type master: Master
        :rtype: None
        """
        # 高频好题！多看看
        # itertools.permutations(iterable, n)
        # 是从iterable从取出来n个元素
        n = 0
        while n < 6:
            # count的含义是指将wordlist中的词两两比较
            # 看有多少个词跟其他的词完全不一样
            # Counter实例是一个字典，但是支持defaultdict的特性（访问不存在的元素会返回默认值）
            diff_words = []
            for w1, w2 in permutations(wordList, 2):
                if slef._match(w1, w2) == 0:
                    diff_words.append(w1)
            count = Counter(diff_words)
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
            # 为什么要选两两不一样最少的词？
            # 原因就在下面的循环，这样可以最大的去除一大部分词（因为和这个词一样的词很少，
            # 反过来说和这个词不一样的词很多，在if的条件里就能尽可能的去除）
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
        # 这道题是说输入的S和T都有可能有回车（用"#"表示）
        # 问S和T是不是同一个字符串
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

#### 845. Longest Mountain in Array
```
class Solution:
    def longestMountain(self, A: 'List[int]') -> 'int':
        # 这道题一个标准的思路
        # 是建立数组up和down
        # 分别统计从左到右的上升的大小和从右到左下降的大小
        # 最终再遍历一遍求下max即可
        # 但是可以用两个变量up和down进行空间优化
        
        up_length = down_length = 0
        res = 0
        for i in range(1, len(A)):
            if down_length != 0 and A[i - 1] < A[i]:
                up_length = down_length = 0
            if A[i - 1] == A[i]:
                up_length = down_length = 0
            if A[i - 1] < A[i]:
                up_length += 1
            if A[i - 1] > A[i]:
                down_length += 1
            # 此时已经是爬过了刚才的山
            # 到达了山的右侧山脚下
            if up_length > 0 and down_length > 0:
                res = max(res, up_length + down_length + 1)
        
        return res
```

#### 846. Hand of Straights
```
from collections import Counter

class Solution:
    def isNStraightHand(self, hand: 'List[int]', W: 'int') -> 'bool':
        counter = Counter(hand)
        for curr in sorted(counter):
            if counter[curr] > 0:
                for j in range(W - 1, -1, -1):
                    # 比方说当前有三张1，要求凑成三组，每组都是1,2,3
                    # 则当前面值为3个牌也必定要有3张
                    counter[curr + j] -= counter[curr]
                    if counter[curr + j] < 0:
                        return False
        return True
```

#### 849. Maximize Distance to Closest Person
```
class Solution:
    def maxDistToClosest(self, seats: 'List[int]') -> 'int':
        # 这道题关键就是找有多少个连续的0
        # left表示最左边的1的位置
        left = -1
        max_dis = 0
        
        for i in range(len(seats)):
            if seats[i] == 0:
                continue
            # 此时下面所有的逻辑说明seats[i] == 1
            if left == -1:
                max_dis = max(max_dis, i)
            else:
                max_dis = max(max_dis, (i - left) // 2)
            left = i
        
        # corner case:
        # 如果最右边是0
        # 需要再检查一遍
        if seats[-1] == 0:
            max_dis = max(max_dis, len(seats) - 1 - left)
        
        return max_dis
```

#### 850. Rectangle Area II
```
class Solution:
    def rectangleArea(self, rectangles):
        """
        :type rectangles: List[List[int]]
        :rtype: int
        """
        # 扫描线好题！多看
        # 给定一系列点（左下角和右上角）代表一个矩阵
        # 求出整体被覆盖的面积
        # 扫描线经典题
        # 这道题是用水平线从下往上扫
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
    # 相当于去除了覆盖的区间求了水平方向上的并集
    def _query(self, active):
        res = 0
        curr = -2 ** 31
        for x1, x2 in active:
            # 核心之一！多多体会
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
                # 通常还剩下两种情况：
                # 要么 A[mid - 1] < A[mid] > A[mid] （出现mountain）
                # 要么 A[mid - 1] > A[mid] < A[mid] 比如[1,2,0,1,0]
                # 但是这道题是只有一个山峰！！！（审题！）即这个山峰左边一定是递减，右边一定是递增
                # 所以后面的情况肯定不会出现
                # 所以只剩下第一种情况，即此时就是mountain点
                return mid
        if A[start] > A[end]:
            return start
        
        return end
```

#### 853. Car Fleet
```
class Solution:
    def carFleet(self, target: 'int', position: 'List[int]', speed: 'List[int]') -> 'int':
        # 这道题是说如果后面的车到达了前面的车的同样位置
        # 就会减速并且并行然后一起撞线
        # 这样就形成了一个车队
        # 问最后有多少个车队撞线
        time = [(target - p) / s for p, s in sorted(zip(position, speed))]
        res = 0
        curr_slow = 0
        for t in time[::-1]:
            # 由于time是按照最远距离到最近距离遍历的
            # 比如当前curr_slow是1小时到达终点
            # 遍历到t的时候，发现需要2小时到达终点
            # 这就说明形成了一个新的车队
            # 所以res要加1
            # 反之，比如遍历的t是半小时，就会和当前的curr_slow时间的车共同形成同一个车队
            # 一起撞线，所以此时res就不用加1了，并且curr_slow仍然是curr_slow time
            if t > curr_slow:
                res += 1
                curr_slow = t
        return res
```

#### 855. Exam Room
```
from heapq import heapify
from heapq import heappop
from heapq import heappush

class ExamRoom:
    # 这道题是间隔问题（interval）
    def __init__(self, N: 'int'):
        self._N = N
        self._max_heap = [(self._dist(-1, N), -1, N)]

    def seat(self) -> 'int':
        _, x, y = heappop(self._max_heap)
        if x == -1:
            seat = 0
        elif y == self._N:
            seat = self._N - 1
        else:
            seat = x + (y - x) // 2
        heappush(self._max_heap, (self._dist(x, seat), x, seat))
        heappush(self._max_heap, (self._dist(seat, y), seat, y))
        return seat

    def leave(self, p: 'int') -> 'None':
        start = end = None
        for interval in self._max_heap:
            if interval[1] == p:
                end = interval
            if interval[2] == p:
                start = interval
            if start and end:
                break
        self._max_heap.remove(start)
        self._max_heap.remove(end)
        heapify(self._max_heap)
        heappush(
            self._max_heap,
            (self._dist(start[1], end[2]), start[1], end[2]),
        )

    # dist函数作用：
    # 给定一个interval的起点x和终点y
    # 求出最大的间隔距离
    def _dist(self, x, y):
        if x == -1:
            return -y
        elif y == self._N:
            return -(self._N - 1 - x)
        else:
            return -(abs(x - y) // 2)

# Your ExamRoom object will be instantiated and called as such:
# obj = ExamRoom(N)
# param_1 = obj.seat()
# obj.leave(p)
```

#### 857. Minimum Cost to Hire K Workers
```
# 理解的不好，回头再看！
from heapq import heappush
from heapq import heappop

class Solution:
    def mincostToHireWorkers(self, quality, wage, K):
        """
        :type quality: List[int]
        :type wage: List[int]
        :type K: int
        :rtype: float
        这道题是说有N个工人，i-th工人有quality[i]的工作质量和要求的wage[i]的薪水
        需要hire K个人（K <= N）凑成一个group，要求这个group满足：
        1. 每个工人的薪水要按照他和其他工人的工作质量比率来pay
        2. 每个工人的薪水要满足最少wage[i]的薪水
        求最少的cost能凑成这样一个group
        """
        # 这道题理解上有个误区  并不是要找最高质量的group
        q_w_ratio = sorted((w / q, q, w) for q, w in zip(quality, wage))
        
        max_heap = []
        sum_q = 0
        res = float('inf')
        for ratio, q, w in q_w_ratio:
            heappush(max_heap, -q)
            sum_q += q
            
            if len(max_heap) > K:
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

#### 862. Shortest Subarray with Sum at Least K
```
from collections import deque

class Solution:
    def shortestSubarray(self, A, k):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        # 这道题是要求返回A中sum大于k的最短的非空子数组长度
        # A中元素是可正可负的
        n = len(A)
        pre_sum = [0]
        for a in A:
            pre_sum.append(pre_sum[-1] + a)
        
        # 核心思路：
        # 保持一个pre_sum中值递增的队列
        # 里面值存pre_sum的索引
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
        # 这个节点是树中所有deepest node的根节点
        # 初始用一个None节点表示root的父亲
        # 注意普通定义的class可以hash的：hash(root)是可以的
        # 但是list这种是disable了hash接口
        # 这道题思路就是先普通的dfs先序遍历更新所有节点的深度
        # 然后求出全局最大深度
        # 最后以这个全局的最大深度为基准
        # 求出LCA
        depth = {None: -1}
        self._dfs(root, parent=None, depth=depth)
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
# 看不懂了
class Solution:
    def profitableSchemes(self, g, p, groups, profits):
        """
        :type G: int
        :type P: int
        :type group: List[int]
        :type profit: List[int]
        :rtype: int
        """
        # 这道题是说一共有g个人
        # 要求最少获取p的收益
        # 现在有多种方案
        # 其中group[i]表示i方案需要多少个人
        # profits[i]表示该方案可以获得多少收入
        # 问一共有多少种方案
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
        
        color_map = dict()
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
            curr_color = False
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

#### 890. Find and Replace Pattern
```
from collections import defaultdict

class Solution:
    def findAndReplacePattern(self, words: 'List[str]', pattern: 'str') -> 'List[str]':
        # 这道题是判断words中那些词和pattern一样
        # 比如abb和opp就是一个pattern
        res = []
        for word in words:
            if self._helper(word, pattern):
                res.append(word)
        return res
    
    def _helper(self, word, pattern):
        word_map = dict()
        pattern_map = dict()
        
        for w, p in zip(word, pattern):
            if w not in word_map:
                word_map[w] = p
            if p not in pattern_map:
                pattern_map[p] = w
            if (word_map[w], pattern_map[p]) != (p, w):
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
        # A里假设没有重复的元素
        if not A or len(A) <= 2:
            return True
        
        asc = True
        desc = True
        for i in range(len(A) - 1):
            if A[i] > A[i + 1]:
                asc = False
            if A[i] < A[i + 1]:
                desc = False
        
        # 如果两个都是False
        # 说明既出现过递增的情况，又出现过递减的情况
        # 这时候就不是单调的Array
        return asc or desc
```

#### 900. RLE Iterator
```
class RLEIterator:

    def __init__(self, A: 'List[int]'):
        self._nums = A[:]
        self._count_nums = 0

    def next(self, n: 'int') -> 'int':
        while self._count_nums < len(self._nums) and self._nums[self._count_nums] < n:
            n -= self._nums[self._count_nums]
            self._count_nums += 2
        if self._count_nums >= len(self._nums):
            return -1
        self._nums[self._count_nums] -= n
        return self._nums[self._count_nums + 1]


# Your RLEIterator object will be instantiated and called as such:
# obj = RLEIterator(A)
# param_1 = obj.next(n)
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
        # 熟练掌握! 
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

#### 911. Online Election
```
from bisect import bisect_right
from collections import defaultdict

class TopVotedCandidate:

    def __init__(self, persons: 'List[int]', times: 'List[int]'):
        self._leads = []
        self._times = []
        count = defaultdict(int)
        lead_person = -1
        for time, person in zip(times, persons):
            count[person] += 1
            if count[person] >= count[lead_person]:
                lead_person = person
            self._leads.append(lead_person)
            # 核心之一：
            # 注意这里就可以将leads和times的index对齐了！
            # 实际上直接在外面初始化一样的
            # 这里为了更好读
            self._times.append(time)

    def q(self, t: 'int') -> 'int':
        # bisect就是二分查找可以插入的位置！！！
        return self._leads[bisect_right(self._times, t) - 1]
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
        # 假设S中只有"("或者")"
        if not S:
            return 0
        
        stack = []
        res = 0
        for ch in S:
            if ch == '(':
                stack.append(ch)
            else:
                # 实际上就是贪心的思路
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
        # "@"之前的部分的"."忽略（指的是忽略这一个字符）
        # "+"号后面全部的字符（在"@"之前的）全部忽略
        # 如果点和加号同时出现
        # 加号后的全部忽略（包括点）
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
        # 这道题说的是假设我们有个目标字符串ababc（5个字符）
        # 初始就是？？？？？5个问号
        # 然后给我们一个stamp假设是abc
        # 问的是如果可以通过多次stamp将？？？？？变成答案
        # 返回一个数组，数组里是每次stamp的起始下标

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
            # 这里的turn_stamp指的是每次用几个*盖掉了target
            turn_stamp = 0
            for sz in range(len(stamp), 0, -1):
                for i in range(len(stamp) - sz + 1):
                    # 这一步就是用来生成abc这个stamp两头不同个数的*
                    new_stamp = '*' * i + stamp[i:i + sz] + '*' * (len(stamp) - sz - i)
                    pos = target.find(new_stamp)
                    while pos != -1:
                        res.append(pos)
                        turn_stamp += sz
                        # 这一步用来更新更新target
                        target = target[:pos] + '*' * len(stamp) + target[pos + len(stamp):]
                        pos = target.find(new_stamp)
            # 则最终如果用长度为len(target)的*了，说明target已经全部被盖掉了
            total_stamp += turn_stamp
        res = res[::-1]
        return res if total_stamp == len(target) else []
```

#### 939. Minimum Area Rectangle
```
class Solution:
    def minAreaRect(self, points: 'List[List[int]]') -> 'int':
        # 这道题问的是points里的点能够组成的最小长方形面积
        # 直接brute force就能解
        visited = set()
        res = 2 ** 31 - 1
        for x1, y1 in points:
            for x2, y2 in visited:
                # 打个比方
                # 此时的x1 y1是左下角的点
                # x2 y2是右上角的点
                # 则x1 y2是右下角的点
                # x2 y1 是左上角的点
                # 这样就凑成了一个矩形
                # 必须4个点都出现过 才能凑成一个矩形
                if (x1, y2) in visited and (x2, y1) in visited:
                    area = abs(x1 - x2) * abs(y1 - y2)
                    if area < res:
                        res = area
            visited.add((x1, y1))
        
        return res if res < 2 ** 31 - 1 else 0
```

#### 940. Distinct Subsequences II
```
class Solution:
    def distinctSubseqII(self, S):
        """
        :type S: str
        :rtype: int
        """
        # 这道题是求S中有多少个非空的子序列
        # 当然S中可以有重复的字符，对这道题的思路并没有影响
        # 最intuitive的解法就是暴力枚举（每个字符都有两种情况选或者不选）
        # 然后用一个hash set去重
        # 但是时间空间复杂度都太高了

        # 这道题的思路非常巧妙！！！
        endswith = [0] * 26
        # 核心之一：
        # 思路是寻找以某一个字符为结尾的字符
        # 按照顺序遍历S，这样保证此时遍历的字符ch的意义很明确
        # 就是以当前字符ch结尾的子序列
        # 核心之二：
        # 然后用endswith的和（正好此时和就是前(重点就是这个"前"字！！！)一个字符结尾的所有子序列的和）
        # 感觉变量的定义的意义很重要！
        for ch in S:
            endswith[ord(ch) - ord('a')] = sum(endswith) + 1
        return sum(endswith) % (10 ** 9 + 7)
```

#### 943. Find the Shortest Superstring
```
from collections import deque

class Solution:
    def shortestSuperstring(self, A):
        """
        :type A: List[str]
        :rtype: str
        """
        def _get_distance(word1, word2):
            # 比如输入abc和bcd
            # 返回2也就是bc这个子串的长度
            for i in range(1, len(word1)):
                if word2.startswith(word1[i:]):
                    return len(word1) - i
            return 0
        
        def _path_to_str(A, graph, path):
            res = A[path[0]]
            for i in range(1, len(path)):
                start_inx = graph[path[i - 1]][path[i]]
                res += A[path[i]][start_inx:]
            return res

        # 这道题说的是找到一个最短的string
        # 使的A中所有的sting都是这个string的子串
        n = len(A)
        # graph[i][j]表示A中的i节点到j节点之间重复的字符串长度
        graph = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                graph[i][j] = _get_distance(A[i], A[j])
                graph[j][i] = _get_distance(A[j], A[i])
        
        # 已知n <= 12所以空间不会太大
        # dist[i][j]表示当已经遍历过的mask(比如11001, 即遍历过了4, 3, 1这三个点)
        # 之后再到j节点的距离
        dist = [[0] * n for _ in range(1 << n)]
        global_repeat_length = -1
        path = []
        # queue里面4个值
        # 分别表示
        # node, mask(即当前已经遍历过的点，用bit位表示), curr_path, curr_repeat_length
        # 其中curr_path初始化为[i]表示以某个节点开始走
        # 相应的curr_repeat_length也就是初始化为0了
        queue = deque([(i, 1 << i, [i], 0) for i in range(n)])
        while queue:
            node, mask, curr_path, curr_repeat_length = queue.popleft()
            if curr_repeat_length < dist[mask][node]:
                continue
            # 1 << n - 1 == 111111....
            if mask == (1 << n) - 1 and curr_repeat_length > global_repeat_length:
                # 此时mask表示我们已经访问过了所有的点
                # 可以更新path和global_repeat_length了
                path, global_repeat_length = curr_path, curr_repeat_length
                continue
            for i in range(n):
                next_mask = mask | (1 << i)
                if next_mask != mask and \
                    dist[mask][node] + graph[node][i] >= dist[next_mask][i]:
                    dist[next_mask][i] = dist[mask][node] + graph[node][i]
                    queue.append([i, next_mask, curr_path + [i], dist[next_mask][i]])
        
        return _path_to_str(A, graph, path)
```

#### 947. Most Stones Removed with Same Row or Column
```
class UnionFind:
    def __init__(self, n):
        self._father = [i for i in range(n)]
    
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self._father[root_a] = root_b
    
    def find(self, a):
        if self._father[a] != a:
            self._father[a] = self.find(self._father[a])
        return self._father[a]

class Solution:
    def removeStones(self, stones):
        """
        :type stones: List[List[int]]
        :rtype: int
        """
        # 这道题是说石头只有同一行或者同一列有其他石头的情况下
        # 才能被remove掉
        # 问最多能remove多少块石头
        # 典型并查集问题 用总的石头数目减去集合的数目
        # 就是能remove掉的石头
        # 因为最终每个集合都会剩下一块石头
        n = len(stones)
        uf = UnionFind(20000)
        uf.count = n
        for x, y in stones:
            # 核心之一
            # 这道题很巧妙的一个思路是
            # 尽管坐标是二维的
            # 但是由于限制了平面一共就10000这么大
            # 所以可以用一个20000长度的一维数组来表示这个点!!!
            uf.union(x, y + 10000)
        
        return n - len(set(uf.find(x) for x, _ in stones))
```

#### 951. Flip Equivalent Binary Trees
```
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def flipEquiv(self, root1: 'TreeNode', root2: 'TreeNode') -> 'bool':
        if root1 is root2:
            return True
        if not root1 or not root2 or root1.val != root2.val:
            return False
        
        # 此时保证了root1.val是等于root2.val的
        return (
            self.flipEquiv(root1.left, root2.left)
            and self.flipEquiv(root1.right, root2.right)
            or self.flipEquiv(root1.left, root2.right)
            and self.flipEquiv(root1.right, root2.left)
        )
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
        # 这道题是说给定一个order
        # 判断words里面的词的顺序是不是按照这个order来的
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
                    # 出现第一个不一样的字符，判断就结束了
                    # 可以去判断下一对而单词
                    break
            else:
                # 核心（corner case之一）
                # 如果没有找到不一样的字符
                # 则理论上短的word应该小于长的word(比如在python中'aa' < 'aaa')
                # 就是说last_word的长度在这个else block里应该是小于curr_word的长度
                # 比如 "app"是应该小于"apple"的
                # 所以如果出现"app" > "apple"
                # 可以直接返回False
                if len(last_word) > len(curr_word):
                    return False

        return True
```

#### 975. Odd Even Jump	
```
class Solution:
    def oddEvenJumps(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        # 这道题就是说奇数次就是在所有大于当前值的里面挑选最小的
        # 偶数次跳就是在所有小于当前值的里面挑选最大的
        # 返回如果想要能跳到终点，有几种start inx的选法
        # 注意只能往右看
        n = len(A)
        next_higher = [0] * n
        next_lower = [0] * n
        
        # 核心：
        # 在最大值里找最小的这种问题，考虑使用单调栈
        # 比如在这道题里
        # 因为[a, i]是先按照a的值排过序的
        # 则某次遍历中，当前的index对应的a肯定比之前的在stack中的index对应的a大
        # 而且当前的a肯定是比之前的a大的值中最小的
        stack = []
        for a, i in sorted([a, i] for i, a in enumerate(A)):
            while stack and stack[-1] < i:
                next_higher[stack.pop()] = i
            stack.append(i)
        
        stack = []
        for a, i in sorted([-a, i] for i, a in enumerate(A)):
            while stack and stack[-1] < i:
                next_lower[stack.pop()] = i
            stack.append(i)
        
        higher = [0] * n
        lower = [0] * n
        higher[-1] = 1
        lower[-1] = 1
        for i in range(n - 2, -1, -1):
            higher[i] = lower[next_higher[i]]
            lower[i] = higher[next_lower[i]]
        
        return sum(higher)
```
