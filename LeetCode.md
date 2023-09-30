# 算法题
## 1.Leetcode 003无重复字符的最长子串
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
方法：【双指针】+【哈希】
```java
public class E03lengthOfLongestSubstring {
    /*
    [(a,3) (b,1) (c,2)]

     b
       e
    abcabcbb
    要点：
        1.用 begin(b) 和 end(e) 表示子串开始和结束位置
        2.用 hash 表检查重复字符
        3.从左向右查看每个字符, 如果:
         - 没遇到重复字符，end++
         - 遇到重复的字符，调整 begin
         - 将当前字符放入 hash 表
        4.end - begin + 1 是当前子串长度
     */

      public int lengthOfLongestSubstring(String s) {
        HashMap<Character,Integer> map = new HashMap<>();
        int begin = 0;
        int maxLength = 0;

        for(int end = 0; end < s.length();end++){//遇到不相同的end往后移动
            char ch = s.charAt(end);
            //遇到相同的begin往后移动
            if(map.containsKey(ch)){//遇到相同字母，取大的角标重新作为begin
                begin = Math.max(begin, map.get(ch) +1);
                map.put(ch,end);
            }else{
                map.put(ch,end);
            }
            maxLength = Math.max(end-begin+1,maxLength);z
        }
        return maxLength;
    }
}
```

## 2.Leetcode 206反转单向链表
**方法1**

构造一个新链表，从**旧链表**依次拿到每个节点，创建新节点添加至**新链表**头部，完成后新链表即是倒序的

```java
public ListNode reverseList(ListNode o1) {
    ListNode n1 = null;
    ListNode p = o1;
    while (p != null) {
        n1 = new ListNode(p.val, n1);
        p = p.next;
    }
    return n1;
}
```
**方法2 递归**
时间复杂度$O(n)$,空间复杂度$O(n)$
```java
public ListNode reverseList(ListNode p) {    
    if (p == null || p.next == null) { // 不足两个节点
        return p; // 最后一个节点
    }
    ListNode last = reverseList(p.next);
    p.next.next = p;//让p=4,5=p.next，反转；p.next.next=p,最后p=1，p.next=null
    p.next = null;//需要一个人指向null，就不会死循环
    return last;
}
```
**方法3 栈**
时间复杂度$O(n)$,空间复杂度$O(n)$
```java
public ListNode ReverseList(ListNode head) {
    Stack<ListNode> stack = new Stack<>();
    //把链表节点全部摘掉放到栈中
    while (head != null) {
        stack.push(head);
        head = head.next;
    }
    if (stack.isEmpty())
        return null;
    ListNode node = stack.pop();
    ListNode dummy = node;
    //栈中的结点全部出栈，然后重新连成一个新的链表
    while (!stack.isEmpty()) {
        ListNode tempNode = stack.pop();
        node.next = tempNode;
        node = node.next;
    }
    //最后一个结点就是反转前的头结点，一定要让他的next
    //等于空，否则会构成环
    node.next = null;
    return dummy;
}
```
## 3.Leetcode 146 LRU缓存
【问题】：请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
1. LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
2. int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
3. void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
4. 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
【哈希】
```java
class LRUCache {
        int cap;
        LinkedHashMap<Integer, Integer> cache=new LinkedHashMap<>();
        public LRUCache(int capacity) {
            this.cap = capacity;
        }


        public int get(int key) {
            if(!cache.containsKey(key)){
                return -1;
            }

            //将key变为最近使用
            makeRecently(key);
            return cache.get(key);

        }

        private void makeRecently(int key) {
            int val =cache.get(key);

            cache.remove(key);//先删除再放入到末尾，末尾是最近使用的。
            cache.put(key,val);
        }

        public void put(int key, int value) {
            if(cache.containsKey(key)){
                //有就更新key的值
                cache.put(key,value);
                //更新一下成为最近使用
                makeRecently(key);
                return;
            }

            if(cache.size()>= cap){
                //删除链表头部最久未使用的key
                int oldestKey=cache.keySet().iterator().next();//cache.keySet()返回缓存中所有key的集合，然后通过iterator()方法获取到对应的迭代器对象
                cache.remove(oldestKey);
            }
            cache.put(key,value);// 将新的 key 添加链表尾部

        }
    }
```

## 4.Leetcode 215 数组中的第K个最大元素
【方法一】：快排
```java
/*
输入: [3,2,1,5,6,4], k = 2
输出: 5
*/
class Solution {
    public int findKthLargest(int[] nums, int k) {
        quickSort(nums,0,nums.length-1);
        return nums[nums.length-k];
    }
    private void quickSort(int[] nums, int left, int right) {
        int i =left;
        int j = right;
        // 子数组长度为 1 时终止递归
        if (i >= j) return;
        while(i<j){//左指针小于右指针时循环
            while(i<j && nums[j]>=nums[left]){//只要右指针指向的值一直大于基准值，就j--
                j--;
            }
            while(i<j && nums[i] <=nums[left]){
                i++;//只要左指针指向的值一直小于基准值，就i++
            }
            swap(nums,i,j);
        }
        swap(nums,left,i);//最后交换基准值和左指针i指向的值

        //左右分区继续排
        quickSort(nums,left,i-1);
        quickSort(nums,i+1,right);
    }
    private static void swap(int[] nums, int i, int j) {
        int tmp= nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```
【方法二】：小根堆
```java
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> queue = new PriorityQueue<>();//系统默认即为小根堆
        int i = 0;//[3,2,1,5,6,4]
        for(;i < k;i++){//[2,3]
            queue.add(nums[i]);//前k个元素建堆，也可以用offer
        }
        for(;i < nums.length;i++)//
            if(queue.peek()<nums[i]){//peek拿出堆顶元素比大小,2<5,3<6
                queue.poll();
                queue.add(nums[i]);//java中没有replace操作，拆分成两步[3,5],[5,6]
            }
        return queue.peek();//5
    }
```
## 5.Leetcode 025 K个一组翻转链表
【问题】：给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
```java
/**
输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
 * 第一步： 数组的反转（前后指针反转）
 * 第二步： 区间的反转 （多个while控制条件）
 * 第三步： 控制反转区间，并连接区间 （设定区间，并连接结果）
 * @author Etta
 * @create 2023-08-24 15:18
 */
class E025reverseKGroup {
   public ListNode reverseKgroup(ListNode head,int k ){
        if(head == null){
            return null;
        }
        ListNode a,b;
        a=b=head;
        for (int i = 0; i < k; i++) {
            //不足k个不许反转
            if(b==null){
                return head;
            }
            b=b.next;
        }

        ListNode newHead = reverse(a,b);

        a.next = reverseKgroup(b,k);//以b为新head，进行下一组反转
        return newHead;
    }
    
   ListNode reverse(ListNode a ,ListNode b){//左闭右开[a,b)
        ListNode pre,cur,nxt;
        pre = null;
        cur =a ;
        nxt =a;//最初nxt和cur一起出发
        while (cur !=b){//直到cur==null
            nxt = cur.next;//定义，指针nxt继续前进
            cur.next =pre;// cur.next指向pre，反转
            //指针pre,cur继续前进
            pre =cur;
            cur=nxt;
        }
        return pre;//最后pre到达终点
    }
}
```

## 6.Leetcode 015 三数之和=target【存疑】

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<List<Integer>>();
        // 枚举 a
        for (int first = 0; first < n; ++first) {
            // 需要和上一次枚举的数不相同
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            // c 对应的指针初始指向数组的最右端
            int third = n - 1;
            int target = -nums[first];
            // 枚举 b
            for (int second = first + 1; second < n; ++second) {
                // 需要和上一次枚举的数不相同
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                // 需要保证 b 的指针在 c 的指针的左侧
                while (second < third && nums[second] + nums[third] > target) {
                    --third;
                }
                // 如果指针重合，随着 b 后续的增加
                // 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == target) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }
}


class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                return result;
            }

            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int left = i + 1;
            int right = nums.length - 1;
            while (right > left) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum > 0) {
                    right--;
                } else if (sum < 0) {
                    left++;
                } else {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));

                    while (right > left && nums[right] == nums[right - 1]) right--;
                    while (right > left && nums[left] == nums[left + 1]) left++;
                    
                    right--; 
                    left++;
                }
            }
        }
        return result;
    }
}
```

## 7.Leetcode 053 最大子数组和
```java
/*
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
 */
public class E053maxSubArray {
    public int maxSubArray(int[] nums) {
        int res = nums[0];
        for(int i = 1; i < nums.length; i++) {
            nums[i] += Math.max(nums[i - 1], 0);//num[i-1]为负数，就为负增益，不如直接+0；
            res = Math.max(res, nums[i]);
        }
        return res;
    }
}

//==================================
public int maxSubArray(int[] nums) {
        int dp[] = new int[nums.length];
        dp[0] = nums[0];
        int max = dp[0];
        for(int i = 1 ; i < nums.length ; i++){
            if(dp[i - 1] <= 0) {
                dp[i - 1] = 0; 
            }
            dp[i] = dp[i - 1] + nums[i] ; 

            max = Math.max(max , dp[i]);
        }

        return max ; 
    }
```

## 8.Leetcode 912. 排序数组（手撕快排）
```java
public class QuickSortHoare {
    public static void sort(int[] a){
        quick(a,0,a.length-1);
    }

    private static void quick(int[] a, int left, int right) {
        if (left >= right) {
            return;
        }
        int p = partition(a, left, right);
        quick(a, left, p - 1);//分别排左边
        quick(a, p + 1, right);//分别排右边
    }

    private static int partition(int[] a,int left, int right){
        int pv =a[left];
        int i = left;
        int j = right;
        while(i<j){
            //1.j从右向左，找比基准点小的值
            while(i<j && a[j] > pv){
                j--;
            }
            //2.i从左到右，找大的值
            while(i<j && a[i] < pv){
                i++;
            }
            //3.交换位置
            swap(a,i,j);

        }
        swap(a,left,i);
        return i;
    }

    private static void swap(int[] a, int i, int j) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] =tmp;
    }

}
```

## 9.Leetcode 021. 合并两个有序链表
【方法一】：三指针
时间复杂度$O(N+M)$：M N分别表示pHead1, pHead2的长度
空间复杂度$O(1)$：常数级空间
```java
public ListNode mergeLists1(ListNode p1, ListNode p2) {
        ListNode s = new ListNode(-1, null);//哨兵节点
        ListNode p = s;//初始指针p在s

        while (p1 != null && p2 != null) {//直到两个list1和llist2都等于null
            if (p1.val < p2.val) {//谁小谁连接到新链表p
                p.next = p1;//新链表list3开始，p从s 指向 下一个 p1
                p1 = p1.next;//list1的p1接着移动
            } else {
                p.next = p2;//新链表list3开始，p从s 指向 下一个 p2
                p2 = p2.next;//list2的p2接着移动
            }
            p = p.next;//新链表list3开始，p向前进一个
        }
        if (p1 != null && p2==null) {//如果p2==null了，就把p指向剩下的p1
            p.next = p1;
        }
        if (p2 != null && p1==null) {//同理
            p.next = p2;
        }
        return s.next;
    }
    /*
    输入：l1 = [1,2,4], l2 = [1,3,4]
    输出：[1,1,2,3,4,4]
     */
```
   
【方法二】：**递归**   
时间复杂度：$O(n+m)$,M N分别表示pHead1, pHead2的长度
空间复杂度：$O(n+m)$
```java
    public ListNode mergeLists(ListNode p1, ListNode p2) {
        if(p2 == null ){
            return p1;
        }
        if(p1 == null){
            return p2;
        }
       if(p1.val < p2.val) {
           p1.next = mergeLists(p1.next,p2);
           return p1;
       }else{
           p2.next = mergeLists(p1,p2.next);
           return p2;
       }
    }
```

## 10.Leecode 001.两数之和
```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        
        HashMap<Integer,Integer> map= new HashMap<>();
        for (int i = 0; i<nums.length; ++i) {
            int x = nums[i];
            int y = target - x;
            if (map.containsKey(y)) {// 在左边找 nums[i]，满足 nums[i]+x=target
                return new int[] {map.get(y), i};// 返回两个数的下标
            }
            map.put(x, i);
        }
        
        return new int[]{};
    }
}
```

## 11.LeetCode 102 二叉树的层序遍历
```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
            List<List<Integer>> result = new ArrayList<>();//new一个数组[[],[]]存储
            if (root == null) {
                return result;
            }
            LinkedList<TreeNode> queue = new LinkedList<>();//new一个队列进行添加每层，然后弹出
            queue.offer(root);
            int c1 = 1;//当前层节点数
            while (!queue.isEmpty()) {//队列非空就一直循环
                List<Integer> level = new ArrayList<>();
                int c2 = 0;//下一层节点数
                for (int i = 0; i < c1; i++) {
                    TreeNode p = queue.poll();//弹出来
                    level.add(p.val);
                    if (p.left != null) {
                        queue.add(p.left);//有左节点就递归
                        c2++;
                    }
                    if (p.right != null) {
                        queue.add((p.right));//有右节点就递归
                        c2++;
                    }
                    }
                    result.add(level);
                    c1 = c2;// 更新循环次数
            }
            return result;
        }
}
```

## 12.LeetCode 005最长回文子串【存疑】
```java
/*
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
*/
class Solution {
    public String longestPalindrome(String s) {
        if(null == s ) return "";
        String max = "";
        int left = 0 ;
        int right = 0;
        for(int i = 0 ; i < s.length() ; i++){
            String one = len(s,i,i); // 奇数
            String two = len(s,i,i+1) ; //偶数
            max = max.length() < one.length()? one : max;
            max = max.length() < two.length()? two : max;
        }
        
        return max;
    }

    /*
        len() 以指针为中心向两边扩展回文
        s 入参
        left 左边界指针 左移
        right 右边界指针 右移
    */
    public String len(String s , int left ,int right){
         while(left >= 0 && right < s.length()){
             if(s.charAt(left) != s.charAt(right) ) break;
             left -- ;
             right ++;
         }
         return s.substring(left + 1,right);//左闭右开
    }
}
```

##  13.LeetCode 033搜索旋转排序数组【存疑】
对于旋转数组 nums = [4,5,6,7,0,1,2] 首先根据 nums[0] 与 target 的关系判断 target 是在左段还是右段。

例如 target = 5, 目标值在左半段，因此在 [4, 5, 6, 7, inf, inf, inf] 这个有序数组里找就行了；
例如 target = 1, 目标值在右半段，因此在 [-inf, -inf, -inf, -inf, 0, 1, 2] 这个有序数组里找就行了。

```java
class Solution {
    public int search(int[] nums, int target) {
        int left = 0 ;
        int right = nums.length - 1;
        while(left <= right){
            int mid = (left + right ) / 2;
            if(nums[mid] == target) {
               return mid;
            }
            // 先根据 nums[0] 与 target 的关系判断目标值是在左半段还是右半段
            if(target>=nums[0]){// 目标值在左半段时，若 mid 在右半段，则将 mid 索引的值改成 inf
                if(nums[mid]<nums[0]){
                    nums[mid] =Integer.MAX_VALUE;
                }
            }else{// 目标值在右半段时，若 mid 在左半段，则将 mid 索引的值改成 -inf
                if(nums[mid]>=nums[0]){
                    nums[mid] =Integer.MIN_VALUE;
                }
            }
            if(nums[mid] < target){
                left = mid+1;
            }
            if(nums[mid]>target ){
                right = mid-1;
            }
        }
        return - 1;
    }
}
```

## 14.LeetCode 020有效的括号

```java
class Solution {
   public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(')');//相应配对情况的放入栈顶
            } else if (c == '[') {
                stack.push(']');
            } else if (c == '{') {
                stack.push('}');
            } else {//遇到右括号，与栈顶元素对比。成对，就从栈顶弹出；不等，无效括号直接返回false；
                if (!stack.isEmpty() && c == stack.peek()) {
                    stack.pop();
                } else {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }
}
```

## 15.LeetCode 141判断是否有环
龟兔赛跑
本题以及下题，实际是 Floyd's Tortoise and Hare Algorithm （Floyd 龟兔赛跑算法）[^15]

> 除了 Floyd 判环算法外，还有其它的判环算法，详见 https://en.wikipedia.org/wiki/Cycle_detection

如果链表上存在环，那么在环上以不同速度前进的两个指针必定会在某个时刻相遇。算法分为两个阶段

阶段1

* 龟一次走一步，兔子一次走两步
* 当兔子能走到终点时，不存在环
* 当兔子能追上龟时，可以判断存在环

阶段2

* 从它们第一次相遇开始，龟回到起点，兔子保持原位不变
* 龟和兔子一次都走一步
* 当再次相遇时，地点就是环的入口

为什么呢？

* 设起点到入口走 a 步（本例是 7），绕环一圈长度为 b（本例是 5），
* 那么**从起点开始，走 a + 绕环 n 圈，都能找到环入口**
* 第一次相遇时
  * 兔走了 a + 绕环 n 圈（本例 2 圈） + k，k 是它们相遇距环入口位置（本例 3，不重要）
  * 龟走了 a + 绕环 n 圈（本例 0 圈） + k，当然它绕的圈数比兔少
  * 兔走的距离是龟的两倍，所以**龟走的** = 兔走的 - 龟走的 = **绕环 n 圈**
* 而前面分析过，如果走 a + 绕环 n 圈，都能找到环入口，因此从相遇点开始，再走 a 步，就是环入口


阶段1 参考代码（判断是否有环）
```java
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode h =head;
        ListNode t = head;

        while( h != null && h.next != null){
            h = h.next.next;
            t= t.next;
            if(h == t){
                return true;
            }
        }
        return false;
        
    }
}
```

阶段2 参考代码（找到环入口）

* 从它们第一次相遇开始，龟回到起点，兔子保持原位不变
* 龟和兔子一次都走一步
* 当再次相遇时，地点就是环的入口

```java
    public ListNode detectCycle(ListNode head) {
    ListNode t = head; // 龟
    ListNode h = head; // 兔
    while (h != null && h.next != null) {
        t = t.next;
        h = h.next.next;
        if (h == t) {
            t = head;
            while (true) {
                if (h == t) {
                    return h;
                }
                h = h.next;
                t = t.next;
            }
        }
    }
    return null;
}
```

## 16.LeetCode 200岛屿数量【存疑】
>岛屿类问题的通用解法、DFS 遍历框架

```java
//网格 DFS 遍历
void dfs(int[][] grid, int r, int c) {
    // 判断 base case
    // 如果坐标 (r, c) 超出了网格范围，直接返回
    if (!inArea(grid, r, c)) {
        return;
    }
    // 访问上、下、左、右四个相邻结点
    dfs(grid, r - 1, c);
    dfs(grid, r + 1, c);
    dfs(grid, r, c - 1);
    dfs(grid, r, c + 1);
}

// 判断坐标 (r, c) 是否在网格中
boolean inArea(int[][] grid, int r, int c) {
    return 0 <= r && r < grid.length 
        	&& 0 <= c && c < grid[0].length;
}

```

本题解法                                    
```java
class Solution {
     public int numIslands(char[][] grid) {
        int count = 0;
        for(int i = 0; i < grid.length; i++) {
            for(int j = 0; j < grid[0].length; j++) {
                if(grid[i][j] == '1'){
                    dfs(grid, i, j);
                    count++;
                }
            }
        }
        return count;
    }
    private void dfs(char[][] grid, int i, int j){
        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] == '0') return;
        grid[i][j] = '0';
        dfs(grid, i + 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i - 1, j);
        dfs(grid, i, j - 1);
    }
}
```

## 17.LeetCode 088 合并两个有序数组
```java
/*
 * [1, 5, 6, 2, 4, 10, 11]
 * 可以视作两个有序区间
 * [1, 5, 6] 和 [2, 4, 10, 11]
 * 合并后，结果仍存储于原有空间
 * [1, 2, 4, 5, 6, 10, 11]
*/
    class Solution {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        for(int i =m-1,j=n-1,k= m+n -1;j>=0;--k){
            nums1[k] = i>=0 && nums1[i]>nums2[j]? nums1[i--]:nums2[j--];
        }
    }
}
```

## 18.LeetCode 046 全排列I
【问题】：`输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]`
【方法一】：回溯+交换
时间复杂度为：$O(n!)$
当函数helper生产排列下标为i的数字时，$[0,...,i,...,n]$中$0\sim i-1$的数字都已经选定，所以继续排列$i\sim n-1$的数字，一旦等于数组长度n，保存一个完整的全排列，在result里add一遍subset循环。
```java
class Solution {
    public List<List<Integer>> permute(int[] nums) {
       LinkedList<Integer> subset = new LinkedList<>();
       List<List<Integer>> res =  new LinkedList<>();
       helper(nums,0,res);
       return res;
    }

    private void helper(int[] nums,int index,List<List<Integer>> result){
        if(index==nums.length){
            LinkedList<Integer> subset = new LinkedList<>();
            for(int num :nums){
                subset.add(num);
            }
            result.add(subset);
        }else if(index<nums.length ){
            for(int j = index;j<nums.length;j++){
                swap(nums,j,index);
                helper(nums,index+1,result);
                swap(nums,j,index);//函数退出前需要清除对排列状态的修改
            }
        }
    }
    private void swap(int[] nums, int i ,int j){
        int tmp = nums[i];
        nums[i]=nums[j];
        nums[j]=tmp;
    }
}
```
【方法二】：回溯+boolean
```java
public class E046permute {
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        dfs(nums,new boolean[nums.length],new LinkedList<>(),result);
        return result;

    }

    static void dfs(int[] nums, boolean[] visited, LinkedList<Integer> stack,List<List<Integer>> result){
        if (stack.size() == nums.length) {
            result.add(new ArrayList<>(stack));
            return;
        }
        //ex:[1,2,3]
        //比那里nums数组，发现没有被使用的数字，则将其标记为使用，并加入stack
        for (int i = 0; i < nums.length; i++) {
            //一开始都是false，[false,false,false] ()
            if (!visited[i]) {
                stack.push(nums[i]);//(1)
                visited[i] =true;//[true,false,false]
                dfs(nums,visited,stack,result);//[true,true,false]->//[true,true,true]
                //回溯
                visited[i]=false;//[false,false,false]
                stack.pop();//()
            }
        }
    }
}
```

## 【关联题】LeetCode47. 全排列 II
【问题】：
`输入：nums = [1,1,2]
输出：
[[1,1,2],
 [1,2,1],
 [2,1,1]]`
 
```java
 
```
## 19.LeetCode 236二叉树的最近公共祖先
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) {
            //只要当前根节点是p和q中的任意一个，就返回（因为不能比这个更深了，再深p和q中的一个就没了）
            return root;
        }
        //根节点不是p和q中的任意一个，那么就继续分别往左子树和右子树找p和q
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        //p和q都没找到，那就没有
        if(left == null && right == null) {
            return null;
        }
        //左子树没有p也没有q，就返回右子树的结果
        if (left == null) {
            return right;
        }
        //右子树没有p也没有q就返回左子树的结果
        if (right == null) {
            return left;
        }
        //左右子树都找到p和q了，那就说明p和q分别在左右两个子树上，所以此时的最近公共祖先就是root
        return root;
    }
}
```
## 20.LeetCode 二叉树的锯齿层次遍历【存疑】
```java
public class E103zigzagLevelOrder {
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.offer(root);

        int c1 = 1;//当前层节点数
        boolean odd = true;

        while (!queue.isEmpty()) {
            LinkedList<Integer> level = new LinkedList<>();//level保存树中每层的节点结果
            int c2 = 0;//下一层节点数
            for (int i = 0; i < c1; i++) {

                TreeNode n = queue.poll();
//                level.add(n.val);//偶数层逆序，奇数层尾部添加，偶数层头部添加
                if(odd){
                    level.offerLast(n.val);
                }else{
                    level.offerFirst(n.val);
                }
                if (n.left != null) {
                    queue.offer(n.left);
                    c2++;
                }
                if (n.right != null) {
                    queue.offer(n.right);
                    c2++;
                }
            }
            odd = !odd;
            result.add(level);
            c1 = c2;//让内层循环的值作为c1
        }

        return result;
    }

```

## 21.LeetCode 054螺旋矩阵【存疑】
【方法一】：递归
```java
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix.length == 0)
            return new ArrayList<Integer>();
        int l = 0, r = matrix[0].length - 1, t = 0, b = matrix.length - 1, x = 0;
        Integer[] res = new Integer[(r + 1) * (b + 1)];
        while (true) {
            for (int i = l; i <= r; i++) res[x++] = matrix[t][i]; // left to right
            if (++t > b) break;
            for (int i = t; i <= b; i++) res[x++] = matrix[i][r]; // top to bottom
            if (l > --r) break;
            for (int i = r; i >= l; i--) res[x++] = matrix[b][i]; // right to left
            if (t > --b) break;
            for (int i = b; i >= t; i--) res[x++] = matrix[i][l]; // bottom to top
            if (++l > r) break;
        }
        return Arrays.asList(res);
    }
}
```
【方法二】：按方向模拟
```java
class Solution {
    int INF= Integer.MAX_VALUE;
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> que = new ArrayList<>();
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] dirs =new int[][]{{0,1},{1,0},{0,-1},{-1,0}};//右，下，左，上
        for(int x = 0,y=0,d=0,i=0;i<m * n;i++){
            que.add(matrix[x][y]);
            matrix[x][y]=INF;
            //下一步的位置
            int nextPosX = x+dirs[d][0];
            int nextPosY = y+dirs[d][1];
            //如果溢出、或者visited
            if(nextPosX<0||nextPosY<0||nextPosX>=m || nextPosY>= n|| matrix[nextPosX][nextPosY]==INF){
                d=(d+1)%4;
                nextPosX = x+dirs[d][0];
                nextPosY = y+dirs[d][1];
            }
            x = nextPosX;
            y = nextPosY;
        }
        return que;
    }
}
```
## 22.LeetCode 092反转链表II
【问题】：*给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。*
![](vx_images/139403822230966.png =732x)
【方法一】：【双指针】头插法
```java
//1->2->3->4->5，m=2,n=4
//输出：1->4->3->2->5
class Solution {
    public ListNode reverseBetween(ListNode head, int m, int n) {
        // 定义一个dummyHead, 方便处理
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        // 初始化指针
        ListNode g = dummyHead;//守卫指针，即left的前一个（m-1）
        ListNode p = dummyHead.next;//pointer,要转的第一个

        // 将指针移到相应的位置
        for(int step = 0; step < m - 1; step++) {
            g = g.next; p = p.next;
        }

        // 头插法插入节点
        for (int i = 0; i < n - m; i++) {
            ListNode removed = p.next;//removed=3
            p.next = p.next.next;//让2->4

            removed.next = g.next;//让3->2
            g.next = removed;//让guard->3
        }

        return dummyHead.next;
    }
}

作者：贾卷积
链接：https://leetcode.cn/problems/reverse-linked-list-ii/
```
【方法二】：递归（绕进去了）,但可以通用法？
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
     public ListNode reverseBetween(ListNode head, int left, int right) {
        if(left==1){
            return  reverseN(head,right);
        }
        head.next = reverseBetween(head.next,left-1,right-1);
        return head;
    }

    ListNode successor = null;
    private ListNode reverseN(ListNode p,int n) {
        if (p == null || p.next == null) { // 不足两个节点
            return p; // 直接返回最后一个节点
        }
        if(n==1){//反转前1个节点
            successor = p.next;//后继节点仍然是原后继节点
            return p;//相当于返回原链表
        }
        //以p.next为起点，反转前n-1个节点
        ListNode last = reverseN(p.next,n-1);
        successor = p.next.next;//此时后继节点应当是递归节点的下一个节点

        p.next.next = p;//前进
        p.next = successor;//后继节点指向后面不循环部分
        return last;
    }
}
```

## 23.LeetCode160相交链表
时间复杂度：$O(m+n)$。链表1和链表2的长度之和。
空间复杂度：$O(1)$。常数的空间。
判断a+(b-c) == b+(a-c)
若两链表 有 公共尾部 (即 c>0) ：指针 A , B 同时指向「第一个公共节点」node 。
若两链表 无 公共尾部 (即 c=0) ：指针 A , B 同时指向 null。
```java
public class E160getIntersectionNode {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA;
        ListNode B =headB;
        while(A != B){
            A = (A!=null) ? A.next:headB;
            B = (B!=null) ? B.next:headA;
        }
        return A;
    }
}
```
【栈】

## 24.LeetCode023合并K个升序链表
【方法】：合并后的第一个节点$first$，一定是某个链表的头节点（因为链表已按升序排列）。

合并后的第二个节点，可能是某个链表的头节点，也可能是$first$ 的下一个节点。

例如有三个链表 $1->2->5$, $3->4->6$,$4->5->6$，找到第一个节点 1 之后，第二个节点不是另一个链表的头节点，而是节点 1 的下一个节点 2。
按照这个过程继续思考，每当我们找到一个节点值最小的节点 $x$，就把节点$x.next$加入「可能是最小节点」的集合中。
因此，我们需要一个数据结构，它支持：
* 从数据结构中找到并移除最小节点。
* 插入节点。
这可以用最小堆实现。初始把所有链表的**头节点**入堆，然后不断弹出堆中最小节点 xxx，如果 $x.next$不为空就加入堆中。循环直到堆为空。把弹出的节点按顺序拼接起来，就得到了答案。

作者：灵茶山艾府
链接：https://leetcode.cn/problems/merge-k-sorted-lists/

【时间复杂度】：$O(NlogK)$
```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        Queue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);//最小堆
/*
使用了Lambda表达式这个堆会按照ListNode对象的val属性的值进行排序，最小的值会排在队列的前面。
PriorityQueue<ListNode> heap = new PriorityQueue<>((a, b) -> b.val - a.val);最大堆
*/

//1.链表中的头节点加入小顶堆
        for (ListNode node: lists) {
            if (node != null) {
                pq.offer(node);
            }
        }
 //2.不断从堆顶移除最小元素，加入新链表
        ListNode dummyHead = new ListNode(-1);
        ListNode cur = dummyHead;
        while (!pq.isEmpty()) {//循环到堆位空
            ListNode minNode = pq.poll();//剩余节点中的最小节点
            if (minNode.next != null) {//下一个节点不为空
                pq.offer(minNode.next);//下一个节点有可能是最小节点，入堆？？？
            }
            cur.next = minNode;//合并到新链表中
            cur = cur.next;//cur到达下一个位置方便指向下一个
            
        }
        return dummyHead.next;
    }
}

```
## 24.LeetCode 415字符串相加
```java
/**
 * 给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和并同样以字符串形式返回。
 * 
 * 你不能使用任何內建的用于处理大整数的库（比如 BigInteger）， 也不能直接将输入的字符串转换为整数形式。
 * 输入：num1 = "11", num2 = "123"
 * 输出："134"
 *
 * @author Etta
 * @create 2023-08-29 17:36
 */
public class E415addStrings {
    public String addStrings(String num1, String num2) {
        StringBuilder res = new StringBuilder("");
        /*
        1 8 9
      +   9 5
      tmp=14
      carry=1
      res=4
         */
        int i = num1.length() - 1, j = num2.length() - 1, carry = 0;//两个尾指针
        while (i >= 0 || j >= 0) {
            int n1 = i >= 0 ? num1.charAt(i) - '0' : 0;//用进位就用 carry 来记录进位值，无则为 0。
            int n2 = j >= 0 ? num2.charAt(j) - '0' : 0;
            int tmp = n1 + n2 + carry;//9+5=14 | 8+9+1=18 | 1+0+1=2
            carry = tmp / 10;//14/10=1 | 18/10=1    | 2/10=0
            res.append(tmp % 10);//14%10=4,尾追加（4）| 18%10=8 （84） |（284）
            i--;//倒着前进一位
            j--;
        }
        if (carry == 1) res.append(1);
        return res.reverse().toString();

    }
}

```
## 【关联题】字符串相乘
```java
/**
输入: num1 = "123", num2 = "456"
输出: "56088"
*/
class Solution {
    public String multiply(String num1, String num2) {
        //base case
         if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int m = num1.length();
        int n= num2.length();
        int carry =0;
        int sum =0;
        StringBuilder sb = new StringBuilder();
        for(int i =m+n-2;i>=0;i--){//倒着循环
            sum =carry;
            int left =0;
            int right = i;
            while(left<=i && right>=0){
                if(left<m && right<n){
                int a = num1.charAt(left)-'0';
                int b = num2.charAt(right)-'0';
                sum += a*b;
                }
                left++;
                right--;
            }
            sb.insert(0,String.valueOf(sum%10));
            carry = sum/10;
        }
        if(carry!=0){
            sb.insert(0,String.valueOf(carry));
        }
        return sb.toString();
    }
}
```
## 25.LeetCode 300最长递增子序列【存疑】
```java
/*
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。
子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。
*/
class Solution {
    public int lengthOfLIS(int[] nums) {
        //tails数组是以当前长度连续子序列的最小末尾元素
        //如tail[0]是求长度为1的连续子序列时的最小末尾元素
        //例：在 1 6 4中 tail[0]=1 tail[1]=4 没有tail[2] 因为无法到达长度为3的连续子序列
        int tails[] = new int[nums.length];
        //注意：tails一定是递增的 因为看题解那个动画 我们最开始的那个元素一定找的是该数组里最小的 不然如果不是最小 由于我们需要连续 后面的数一定会更大（这样不好的原因是 数越小 我们找到一个比该数大的数的几率肯定会更大）
        int res = 0;
        for(int num:nums){
            //每个元素开始遍历 看能否插入到之前的tails数组的位置 如果能 是插到哪里
            int i = 0,j = res;
            while(i < j){
                int m = (i+j)/2;
                if(tails[m] < num) i = m+1;
                else j = m;
            }
             //如果没有到达j==res这个条件 就说明tail数组里只有部分比这个num要小 那么就把num插入到tail数组合适的位置即可 但是由于这样的子序列长度肯定是没有res长的 因此res不需要更新
            tails[i] = num;
            //j==res 说明目前tail数组的元素都比当前的num要小 因此最长子序列的长度可以增加了 
            if(j == res) res ++; 
        }
        return res;
    }
}

```

## 26.LeetCode 042 接雨水
![](vx_images/4883611230967.png)
*输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 *
```java
public class E042trap {
    public int trap(int[] height) {
        //按列算
        int sum = 0;
        int[] maxLeft = new int[height.length];
        int[] maxRight = new int[height.length];
        //最两端的列不用考虑，因为一定不会有水。
        for (int i = 1; i < height.length - 1; i++) {
            //代表第 i 列左边最高的墙的高度
            maxLeft[i] = Math.max(maxLeft[i - 1], height[i - 1]);
        }
        for (int j = height.length - 2; j >=0; j--) {
            //代表第 i 列右边最高的墙的高度
            maxRight[j] = Math.max(maxRight[j + 1], height[j + 1]);
        }
        //只有较小的一段大于当前列的高度才会有水，其他情况不会有水
        for (int i = 1; i < height.length - 1; i++) {
            int min = Math.min(maxLeft[i], maxRight[i]);
            if (min > height[i]) {
                sum = sum + (min - height[i]);
            }
        }
        return sum;
    }
}
```

## 27.LeetCode143 重排链表
*给定一个单链表 L 的头节点 head ，单链表 L 表示为：
L0 → L1 → … → Ln - 1 → Ln
请将其重新排列后变为：
L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。*
快慢指针+ 反转链表+ 合并链表
```java

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
        public void reorderList(ListNode head) {
        // 快慢指针找到链表中点
        ListNode fast = head, slow = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        //例子：ABC-123;
        // newHead 指向右半部分链表(123)
        ListNode newHead = slow.next;
        slow.next = null;//断干净，左半边链条(ABC),C指向一个null

        // 反转右半部分链表
        newHead = reverseList(newHead);
        
        // 合并:链表节点依次连接
        while (newHead != null) {//head=A,newHead=1
            ListNode tmp = newHead.next;//存储临时newHead.next =2，否则会被覆盖|||newHead.next =3|||newHead.next=null
            
            newHead.next = head.next;//将1->B    |||  2->C ||| 3->null
            head.next = newHead;//将A->1，此时形成A->1->B|||  B->2,形成B->2->C，合起来即A->1->B->2->C ||| C->3, A->1->B->2->C->3
            
            head = newHead.next;//此时head=B  ||| 此时head = C
            newHead = tmp; //newHead=2  ||| newHead = 3
        }
    }

    private ListNode reverseList(ListNode p) {
        if (p == null || p.next == null) { // 不足两个节点
            return p; // 最后一个节点
        }
        //有两个以上节点的情况
        ListNode last = reverseList(p.next);//返回值用来拿到最后一个节点，假设p为倒数第二节点
        p.next.next = p;//最后p=1，p.next=null
        // 此时p是2, p.next是3, 要让3指向2,代码写成 p.next.next=p
        //                 还要注意2要指向 null, 否则就死链了
        p.next = null;
        return last;
    }
}
```

## 28.LeetCode 124 二叉树中的最大路径和
```java
public class E14maxPathSum {
    int res = Integer.MIN_VALUE;//定义一个绝对小值拿来比较
    public int maxPathSum(TreeNode root) {
        helper(root);
        return res;
    }

    private int helper(TreeNode root) {
        if(root==null){
            return 0;
        }
        int left = helper(root.left);
        int right = helper(root.right);
        res = Math.max(left+right+root.val,res);
        return Math.max(0,Math.max(left,right)+root.val);
    }

}
```


## 29.LeetCode 019 删除链表中的倒数第N个结点
```java
public class E019DeleteNthNode {
    /*
        输入：head = [1,2,3,4,5], n = 2
        输出：[1,2,3,5]
     */
    //方法1
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode s = new ListNode(-1,head);//否则1不知道下一个节点是谁
        recursion(s,n);//从哨兵出发
        return s.next;//返回哨兵的下一个，即头结点
    }

    private int recursion(ListNode p, int n){
        //结束条件
        if(p==null){
            return 0;//返回倒数第0个
        }
        int Nth = recursion(p.next, n);//Nth：下一个节点的倒数位置
        if(Nth == n){
            //删除 假设p=3,p.next=4,p.next.next=5
            p.next = p.next.next;//删除4之后,需要将3->5
        }
        return Nth+1;//当前节点的倒数位置
    }

    //方法2：快慢指针
    public ListNode removeNthFromEnd2(ListNode head,int n){
        ListNode s = new ListNode(-1,head);//哨兵结点
        ListNode p1=s;
        ListNode p2=s;

        for (int i = 0; i < n+1; i++) {//让快指针先走n+1步，
            p2 =p2.next;
        }
        while(p2 !=null){
            p1=p1.next;
            p2=p2.next;
        }//此时p1代表要删除节点的上一个节点
        p1.next = p1.next.next;//把删除节点直接跳过了
        return s.next;
    }
}
```

## 30.LeetCode 094 二叉树中序遍历 
【递归】
```java
public class E094inorderTraversal {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        dfs(result,root);
        return result;
    }

    private void dfs(List<Integer> result, TreeNode root) {
        if (root==null){
            return;
        }
        dfs(result,root.left);
        result.add(root.val);
        dfs(result,root.right);
    }

}
```
【栈】

```
public class E094inorderTraversal {
public List<Integer> inorderTraversal(TreeNode root) {\\前序
        LinkedList<Integer> result = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();


        TreeNode cur =root;
        while (cur!=null || !stack.isEmpty()){
            while (cur.left!=null){
                stack.push(cur);
            }
            result.add(stack.pop().val);
            cur =cur.right;
        }
        return result;
    }

 public List<Integer> preorderTraversal(TreeNode root) {\\中序
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> result = new LinkedList<>();
        TreeNode cur =root;
        while (cur !=null|| !stack.isEmpty()){
            while (cur!=null){
                result.add(cur.val);
                stack.push(cur);
                cur = cur.left;
            }
            cur =stack.pop();
            cur =cur.right;
        }
        return result;

    }
    

public List<Integer> postorderTraversal(TreeNode root) {\\后序
        Stack<TreeNode> stack = new Stack<>();
        List<Integer> result = new LinkedList<>();
        TreeNode cur =root;
        TreeNode prev = null;

        while (cur != null || !stack.isEmpty()){
            while (cur!=null){
                stack.push(cur);
                cur = cur.left;
            }

            if(cur.right!=null && cur.right != prev){
                cur=cur.right;
            }else{
                stack.pop();
                result.add(cur.val);
                prev = cur;
                cur=null;
            }
        }
        return result;

    }
}
```
## 31.LeetCode 072  编辑距离 【存疑】
【问题】：给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数  。
你可以对一个单词进行如下三种操作：
* 插入一个字符
* 删除一个字符
* 替换一个字符

## 32.LeetCode 056 合并区间
时间复杂度：遍历区间数组的时间为$O(n)$，对区间数组进行排序的时间复杂度为$O(nlogn)$,因此总的时间复杂度为$O(nlogn)$.

```java
class Solution {
/**
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
*/
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a1, a2) -> a1[0] == a2[0] ? a1[1] - a2[1] : a1[0] - a2[0]); // 对区间进行升序排序
        List<int[]> mergeRes = new ArrayList<>();    // 结果列表
        int start = intervals[0][0];    // 初始化合并区间的起点为首个区间的起点
        int end = intervals[0][1];      // 初始化合并区间的终点为首个区间的终点
        int n = intervals.length;
        for(int i = 1; i < n; i++){
            // 判断每一个区间能否加入当前合并区间，由于首个区间已经是初始的合并区间，因此从第二个区间开始判断
            if(intervals[i][0] > end){
                // 当前区间不能加入当前的合并区间，记录当前合并区间。以当前区间作为新的合并区间
                mergeRes.add(new int[]{start, end});
                start = intervals[i][0];
                end = intervals[i][1];
            }else{
                // 当前区间加入当前的合并区间，更新合并区间的终点
                end = Math.max(end, intervals[i][1]);
            }
        }
        mergeRes.add(new int[]{start, end});    // 补充加入最后一个合并区间
        // 转换结果列表为结果数组
        return res.toArray(new int[0][]);;
    }
}

```

## 33.LeetCode 704 二分查找
```java
    public static int search(int[] a,int target){
        int i = 0,j=a.length-1;//设置指针和初值
        while(i<=j){//范围内有东西
            int m = (i+j)/2;
            if(target<a[m]){//目标在中间值左边
                j=m-1;
            }else if(target>a[m]){//目标在中间值右边
                i=m+1;
            }else{
                return m;//找到了，索引的位置
            }
        }
        return -1;
    }
```

## 34.LeetCode 232 用栈实现队列
```java

/**
 * 请你仅使用两个栈实现先入先出队列。队列应当支持一般队列支持的所有操作（push、pop、peek、empty）：
 *
 * 实现 MyQueue 类：
 *
 * void push(int x) 将元素 x 推到队列的末尾
 * int pop() 从队列的开头移除并返回元素
 * int peek() 返回队列开头的元素
 * boolean empty() 如果队列为空，返回 true ；否则，返回 false
 *
 * @author Etta
 * @create 2023-07-28 19:42
 */
public class E232stackToQueue {
    /*
       队列头      队列尾
       a b          a(pop) b(pop)
       顶   底   底   顶
       s1            s2

       s2.push(a)
       s2.push(b)
       先把s2的所有元素移动到s1
       s1.pop()

    */
    Stack<Integer> s1 = new Stack<>();
    Stack<Integer> s2 = new Stack<>();


    public void push(int x) {//向队列尾部添加
        s2.push(x);//向s2栈顶压入元素x

    }

    public int pop() {//向队列头部移除
        if(s1.isEmpty()){//如果s1为空，s2不为空
            while(!s2.isEmpty()){
                s1.push(s2.pop());//将s2的元素移动至s1后，再去pop()
            }
        }x
        return s1.pop();//如果s1不为空，直接从s1pop()
    }

    public int peek() {//从队列头获取
        if(s1.isEmpty()){
            while(!s2.isEmpty())
            s1.push(s2.pop());
        }
        return s1.peek();
    }

    public boolean empty() {
        return(s1.isEmpty() && s2.isEmpty());
    }
}

```

## 35.LeetCode 1143 最长公共子序列

时间复杂度分析： $O(nm)$，其中n 和 m 分别是字符串 text1 和 text2的长度。
```java
/**
输入：text1 = "abcde", text2 = "ace" 
输出：3  
解释：最长公共子序列是 "ace" ，它的长度为 3 。
*/
public class E1143longestCommonSubsequence {
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(),n=text2.length();
        int[][] f = new int[m+1][n+1];

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i-1) == text2.charAt(j-1)){
                    /*
                    也就是说两个字符串的最后一位相等，那么问题就转化成了
                    字符串text1的[1,j-1]区间和字符串text2的[1,j-1]区间的最长公共子序列长度再加上一，
                    即f[i][j] = f[i - 1][j - 1] + 1。
                     */
                    f[i][j] = f[i-1][j-1]+1;
                }else{
                    /*
                    两个字符串的最后一位不相等，
                    那么字符串text1的[1,i]区间和字符串text2的[1,j]区间的最长公共子序列长度无法延长，
                    因此f[i][j]就会继承f[i-1][j]与f[i][j-1]中的较大值，即f[i][j] = max(f[i - 1][j],f[i][j - 1]) 。 
                     */
                    f[i][j] = Math.max(f[i-1][j],f[i][j-1]);
                }
            }
        }
        return f[m][n];

    }
}
```
## 36.LeetCode 082 删除排序链表中的重复元素
p1 是待删除的上一个节点，每次循环对比 p2、p3 的值

* 如果 p2 与 p3 的值重复，那么 p3 继续后移，直到找到与 p2 不重复的节点，p1 指向 p3 完成删除
* 如果 p2 与 p3 的值不重复，p1，p2，p3 向后平移一位，继续上面的操作
* p2 或 p3 为 null 退出循环
  * p2 为 null 的情况，比如链表为 1 1 1 null
  
```
p1 p2 p3
s, 1, 1, 1, 2, 3, null

p1 p2    p3
s, 1, 1, 1, 2, 3, null

p1 p2       p3
s, 1, 1, 1, 2, 3, null

p1 p3
s, 2, 3, null

p1 p2 p3
s, 2, 3, null

   p1 p2 p3
s, 2, 3, null
```
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode deleteDuplicates(ListNode head) {
       if (head == null || head.next == null) {
            return head;
        }

        ListNode s = new ListNode(-1, head);
        ListNode p1 = s;
        ListNode p2, p3;

        while ((p2 = p1.next) != null && (p3 = p2.next) != null) {
            if (p2.val == p3.val) {
                while ((p3 = p3.next) != null && p3.val == p2.val) {
                } //p3找到了不重复的值
                p1.next = p3;

            } else {
                p1 = p1.next;
            }
        }
        return s.next;

    }
}
```
**【扩展】有序链表去重-力扣 83 题**
例如
```
输入：head = [1,1,2]
输出：[1,2]

输入：head = [1,1,2,3,3]
输出：[1,2,3]
```
注意：**重复元素保留一个**
```
p1   p2
1 -> 1 -> 2 -> 3 -> 3 -> null
```
* p1.val == p2.val 那么删除 p2，注意 p1 
```
p1   p2
1 -> 2 -> 3 -> 3 -> null
```
* p1.val != p2.val 那么 p1，p2 向后移动
```java
     p1   p2
1 -> 2 -> 3 -> 3 -> null
         
          p1   p2
1 -> 2 -> 3 -> 3 -> null     
```

* p1.val == p2.val 那么删除 p2

```
          p1   p2
1 -> 2 -> 3 -> null   
```

* 当 p2 == null 退出循环

代码

```java
public ListNode deleteDuplicates(ListNode head) {
    // 链表节点 < 2
    if (head == null || head.next == null) {
        return head;
    }
    // 链表节点 >= 2
    ListNode p1 = head;
    ListNode p2;
    while ((p2 = p1.next) != null) {
        if (p1.val == p2.val) {
            p1.next = p2.next;
        } else {
            p1 = p1.next;
        }
    }
    return head;
}
```
## 37.LeetCode 004 寻找两个正序数组的中位数【存疑】
时间复杂度：$O((m+n)/2)$
```java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
   /*
    给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
    算法的时间复杂度应该为 O(log (m+n))，即要用二分法
     */
        int len1 = nums1.length;
        int len2 = nums2.length;
        int mid = (len1 + len2) / 2;
        int temp = Integer.MIN_VALUE;
        int temp2 = Integer.MIN_VALUE;
        int i1 = 0, i2 = 0;
        // 标识长度和奇偶
        boolean flag = (len1 + len2) % 2 == 0;
        for (int i = 0; i <= mid; i++) {
            // 从两个端点各取一个值
            int n1 = i1 < len1? nums1[i1]: Integer.MAX_VALUE;
            int n2 = i2 < len2? nums2[i2]: Integer.MAX_VALUE;
            // 取其中小的一个
            if(n1 < n2){
                // 取n1
                i1++;
                if(flag)
                    temp2 = temp;
                temp = n1;
            }else {
                i2++;
                if(flag)
                    temp2 = temp;
                temp = n2;
            }
        }
        return flag?(double)(temp+temp2)/2.0:(double)temp;
    }

```

## 38.LeetCode 199 二叉树右视图
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
```java
public class E199rightSideView {
    List<Integer> result = new ArrayList<>();
    public List<Integer> rightSideView(TreeNode root) {

        dfs(root,0); // 从根节点开始访问，根节点深度是0
        return result;

    }
    private void dfs( TreeNode root,int depth) {
        if (root==null){
            return;
        }
        // 先访问 当前节点，再递归地访问 右子树 和 左子树。
        if(depth ==result.size()){
            // 如果当前节点所在深度还没有出现在result里，说明在该深度下当前节点是第一个被访问的节点，因此将当前节点加入result中。
            result.add(root.val);
        }
        depth++;
        dfs(root.right,depth);
        dfs(root.left,depth);
    }
}
```

## 39.LeetCode 093 复原IP地址【存疑】
**思路**
* 字符串的长度小于 4 或者大于 12 ，一定不能拼凑出合法的 ip 地址
* 每一个结点可以选择截取的方法只有 3 种：截 1 位、截 2 位、截 3 位（不可能大于255）
* 由于 ip 段最多就 4 个段，因此这棵三叉树最多 4 层，这个条件作为递归终止条件之一；
* 每一个结点表示了求解这个问题的不同阶段，需要的状态变量有：
    * splitTimes：已经分割出多少个 ip 段；
    * begin：截取 ip 段的起始位置；
    * path：记录从根结点到叶子结点的一个路径（回溯算法常规变量，是一个栈）；
    * res：记录结果集的变量，常规变量。

```java
class Solution {
    //画图理解
    public List<String> restoreIpAddresses(String s) {
        //定义表示一个字符长度的变量
        int len = s.length();
        //定义一个返回结果的集合
        List<String> res = new ArrayList<>();
        //如果当前字符长度大于12或者小于4都不满足
        if(len > 12 || len <4){
            return res;
        }
        //定义一个保存路径上的变量
        Deque<String> path = new ArrayDeque<>();
        //深度优先搜索
        dfs(s,len, 0, 4, path, res);
        //返回结果
        return res;
    }

    public void dfs(String s, int len, int begin, int residue, Deque<String> path, List<String> res){
    //residue 需要一个变量记录剩余多少段还没被分割
        //如果字符串已经遍历到最后了，并且已经切分为4段了，
        //就把当前路径上的元素加入到返回的结果集中
        if(begin == len){
            if(residue ==0){
                res.add(String.join(".", path));
            }
            return;
        }
        //begin表示遍历字符串从哪里开始
        for(int i = begin; i < begin+3; i++){
            //如果超出字符串的长度，就直接退出
            //begin，每次选择都是从之前选择的元素的下一个元素开始，
            if(i >= len){
                break;
            }
            //如果剩余元素大于ip最多能容纳的个数，就剪枝。
            if(len -i > residue * 3){
                continue;
            }
            //判断当前截取字符是否是小于0或者大于255
            //这里的begin和i，代表的是，这时候截取了几个字符
            //begin从哪里开始，i到哪里结束
            if(judgeIpSegment(s, begin, i)){
                //保留当前截取字符
                String currentIpSegment = s.substring(begin, i+1);
                //将当前路径上的元素加入到路径队列中
                path.addLast(currentIpSegment);
                //递归下一层
                dfs(s, len, i+1,residue -1, path, res);
                //剪枝
                path.removeLast();
            }
        }
    }
    private boolean judgeIpSegment(String s, int left, int right){
        //定义一个表示整个字符的长度
        int len = right - left +1;
        //如果截取的大于等于2的字符的开头为0，就直接false
        if(len > 1 && s.charAt(left) == '0'){
            return false;
        }
        //定义返回结果的集合
        int res = 0;
        //拼接字符
        while(left <= right){
            //res*10 是为了将先加的字符默认比后面的字符大10倍，也就是位数是从小到大
            res = res * 10 + s.charAt(left) - '0';
            left++;
        }
        return res >= 0 && res <= 255;
    }
}
```
## 40.LeetCode 031下一个排列【存疑】
```java
class Solution {
    public void nextPermutation(int[] nums) {
        if(nums.length == 0){
            return;
        }
        int len = nums.length;
        //从后向前遍历
        for(int i = len-1;i >= 0;i--){
            //如果i为0，说明数组从后到前是递增（654321）的,没有更大的了
            //直接重排序变成一个递减的（123456）符合题意
            if(i == 0){
                Arrays.sort(nums);
                return;
            }else if(nums[i] > nums[i-1]){
            //从右往左找到第一个降序对后，要再从右往左找到第一个大于左指针的数，这样交换得到数最小
                //一旦出现后一个数字nums[i]比前一个大，说明存在更大的整数
                //对nums[i]及后面的数组排序，从小到大
                Arrays.sort(nums,i,len);
                for(int j = i;i < len;j++){
                    //由于从i开始后面已经排序
                    //那么保证获得比nums[i-1]大的数，是[i,...,len-1]中最小的,交换即可
                    if(nums[j] > nums[i-1]){
                        swap(nums,j,i-1);
                        return;
                    }
                }
            }
        }
    }
    public void swap(int[] nums,int i,int j){
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
作者：Lan
链接：https://leetcode.cn/problems/next-permutation/
```

## 41.LeetCode148. 排序链表
```java

/**
 * 输入：head = [-1,5,3,4,0]
 * 输出：[-1,0,3,4,5]
 * @author Etta
 * @create 2023-08-31 21:13
 */
public class E148sortList {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        //快慢指针法分割中点
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;//将链表切断

        ListNode left = sortList(head);//左半截的开始
        ListNode right = sortList(tmp);//右半截的开始
        ListNode h = new ListNode(0);//哨兵结点作为开始，比较两指针处节点值大小，由小到大加入合并链表头部，指针交替前进，直至添加完两个链表。
        ListNode res = h;
        while (left != null && right != null) {
            if (left.val < right.val) {
                h.next = left;
                left = left.next;
            } else {
                h.next = right;
                right = right.next;
            }
            h = h.next;
        }
        //提前比较结束，剩下结点全部来自右半截
        h.next = left != null ? left : right;
        return res.next;
    }
}
```

## 42.LeetCode 070.爬楼梯||兔子生兔子||斐波那契
```java
class Solution {
    public int climbStairs(int n) {
        int p =0;
        int q=0;
        int r=1;
        for(int i=1;i<=n;++i){
            p=q;
            q=r;
            r=p+q;
        }
        return r;
    }
}
```

## 43.LeetCode 069.x的平法根
```java
/**
 * 输入：x = 8
 * 输出：2
 * 解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。
 * 注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。
 * @author Etta
 * @create 2023-09-05 17:39
 */
public class E069Mysqrt {
    public int mySqrt(int x) {
//        如果这个整数的平方 恰好等于 输入整数，那么我们就找到了这个整数；
//        如果这个整数的平方 严格大于 输入整数，那么这个整数肯定不是我们要找的那个数；
//        如果这个整数的平方 严格小于 输入整数，那么这个整数 可能 是我们要找的那个数（重点理解这句话）。
            int l = 0, r = x, ans = -1;//x=8
            while (l <= r) {//3<=4
                int mid = l + (r - l) / 2;//mid=4 mid =3 mid =3 mid=2 mid=3
                if ( mid <= x/mid) {//4<=8
                    ans = mid; //ans=2
                    l = mid + 1;//l=3
                } else {//16>8 9>8
                    r = mid - 1;//r=7 r=6 r=5 r=2
                }
            }
            return ans;

    }
}
```

## 44.LeetCode 002两数相加（链表）【存疑】
作者：画手大鹏
链接：https://leetcode.cn/problems/add-two-numbers/solutions/7348/hua-jie-suan-fa-2-liang-shu-xiang-jia-by-guanpengc/
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        //定义一个新联表伪指针，用来指向头指针，返回结果
        ListNode dummy = new ListNode(0);
        //定义一个进位数的指针，用来存储当两数之和大于10的时候，
        int carry = 0;
        //定义一个可移动的指针，用来指向存储两个数之和的位置
        ListNode cur = dummy;
        //当l1 不等于null或l2 不等于空时，就进入循环
        while(l1!=null || l2!=null){
            //如果l1 不等于null时，就取他的值，等于null时，就赋值0，保持两个链表具有相同的位数
            int x= l1 !=null ? l1.val : 0;
             //如果l1 不等于null时，就取他的值，等于null时，就赋值0，保持两个链表具有相同的位数
            int y = l2 !=null ? l2.val : 0;
            //将两个链表的值，进行相加，并加上进位数
            int sum = x + y + carry;
            //计算进位数
            carry = sum / 10;
            //计算两个数的和，此时排除超过10的请况（大于10，取余数）
            sum = sum % 10;
            //将求和数赋值给新链表的节点，
            //注意这个时候不能直接将sum赋值给cur.next = sum。这时候会报，类型不匹配。
            //所以这个时候要创一个新的节点，将值赋予节点
            cur.next = new ListNode(sum);
            //将新链表的节点后移
            cur = cur.next;
            //当链表l1不等于null的时候，将l1 的节点后移
            if(l1 !=null){
                l1 = l1.next;
            }
            //当链表l2 不等于null的时候，将l2的节点后移
            if(l2 !=null){
                l2 = l2.next;
            } 
        }
        //如果最后两个数，相加的时候有进位数的时候，就将进位数，赋予链表的新节点。
        //两数相加最多小于20，所以的的值最大只能时1
        if(carry == 1){
            cur.next = new ListNode(carry);
        }
        //返回链表的头节点
        return dummy.next;
    }
}
```


## 45.LeetCode 022生成有效括号
![](vx_images/347052116230947.png =1201x)
```java
public class E022generateParenthesis {

    // 做减法
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        // 特判
        if (n == 0) {
            return res;
        }

        // 执行深度优先遍历，搜索可能的结果
        dfs("", n, n, res);
        return res;
    }

    /**
     * @param curStr 当前递归得到的结果
     * @param left   左括号还有几个可以使用
     * @param right  右括号还有几个可以使用
     * @param res    结果集
     */
    private void dfs(String curStr, int left, int right, List<String> res) {
        // 因为每一次尝试，都使用新的字符串变量，所以无需回溯
        // 在递归终止的时候，直接把它添加到结果集即可，注意与「力扣」第 46 题、第 39 题区分
        if (left == 0 && right == 0) {
            res.add(curStr);
            return;
        }

        // 剪枝（如图，左括号可以使用的个数严格大于右括号可以使用的个数，才剪枝，注意这个细节）
        if (left > right) {
            return;
        }
        //可以生成左枝叶的条件
        if (left > 0) {
            dfs(curStr + "(", left - 1, right, res);
        }
        //可以生成右枝叶的条件：左括号数量小于右括号数量，并且右括号剩余数量>0
        if (left <right && right > 0) {
            dfs(curStr + ")", left, right - 1, res);
        }
    }
}
```

## 46.剑指offer 022链表中倒数第k个节点
【快慢指针法】
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        ListNode s = new ListNode(-1);
        s.next=head;
        ListNode p1 = s;
        ListNode p2 = s;
        if (head == null) {
            return null;
        }
        for (int i = 0; i <k+1; i++) {
            p2=p2.next;

        }
        while (p2!=null){
            p1= p1.next;
            p2=p2.next;
        }
        s.next= p1.next;
        return s.next;

    }
}
```

## 47.LeetCode 165 比较版本号【存疑】
【问题】：给你两个版本号 version1 和 version2 ，请你比较它们。

版本号由一个或多个修订号组成，各修订号由一个 '.' 连接。每个修订号由 多位数字 组成，可能包含 前导零 。每个版本号至少包含一个字符。修订号从左到右编号，下标从 0 开始，最左边的修订号下标为 0 ，下一个修订号下标为 1 ，以此类推。例如，2.5.33 和 0.1 都是有效的版本号。

比较版本号时，请按从左到右的顺序依次比较它们的修订号。比较修订号时，只需比较 忽略任何前导零后的整数值 。也就是说，修订号 1 和修订号 001 相等 。如果版本号没有指定某个下标处的修订号，则该修订号视为 0 。例如，版本 1.0 小于版本 1.1 ，因为它们下标为 0 的修订号相同，而下标为 1 的修订号分别为 0 和 1 ，0 < 1 。
返回规则如下：
* 如果 version1 > version2 返回 1，
* 如果 version1 < version2 返回 -1，
* 除此之外返回 0。
* 
【方法】：【双指针】 两个字符串各遍历一遍: O(n+m）
```java
class Solution {
    public int compareVersion(String v1, String v2) {
        int i = 0, j = 0;
        int n = v1.length(), m = v2.length();
        while (i < n || j < m) {
            int num1 = 0, num2 = 0;
            //这样做可以直接去前导0，同时将字符串转换成数字也便于比较大小。
            while (i < n && v1.charAt(i) != '.') num1 = num1 * 10 + v1.charAt(i++) -'0';
            while (j < m && v2.charAt(j) != '.') num2 = num2 * 10 + v2.charAt(j++) -'0';
            if (num1 > num2) return 1;
            else if (num1 < num2) return -1;
            i++;
            j++;
        }
        return 0;
    }
}
```
## 48.LeetCode 08字符串转换为整数【存疑】

## 49.LeetCode 239滑动窗口最大值【存疑】
作者：Krahets
链接：https://leetcode.cn/problems/sliding-window-maximum/solutions/2361228/239-hua-dong-chuang-kou-zui-da-zhi-dan-d-u6h0/
```java
public class E239maxSlidingWindow {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums.length == 0 || k == 0) return new int[0];
        Deque<Integer> deque = new LinkedList<>();//先进先出
        int[] res = new int[nums.length - k + 1];
        for (int j = 0, i = 1 - k; j < nums.length; i++, j++) {
            // 删除 deque 中对应的 nums[i-1]
            if (i > 0 && deque.peekFirst() == nums[i - 1]) {
                deque.removeFirst();
            }
            // 保持 deque 递减
            while (!deque.isEmpty() && deque.peekLast() < nums[j]) {
                deque.removeLast();
            }
            deque.addLast(nums[j]);
            // 记录窗口最大值
            if (i >= 0)
                res[i] = deque.peekFirst();
        }
        return res;
    }
}
```

## 50.LeetCode 041缺失的第一个整数
作者：BugTime
链接：https://leetcode.cn/problems/first-missing-positive/solutions/1142914/duo-tu-yu-jing-xiang-jie-yuan-di-ha-xi-b-se25/
```java
public class E041firstMissingPositive {
    public int firstMissingPositive(int[] nums) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            while (nums[i] > 0 && nums[i] <= len && nums[nums[i] - 1] != nums[i]) {
                // 满足在指定范围内、并且没有放在正确的位置上，才交换
                // 例如：数值 3 应该放在索引 2 的位置上
                swap(nums, nums[i] - 1, i);
            }
        }
        // [1, -1, 3, 4]
        for (int i = 0; i < len; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        // 都正确则返回数组长度 + 1
        return len + 1;
    }
    private void swap(int[] nums, int a, int b) {
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
        
    }
}
```
## 51.LeetCode 076最小覆盖子串【存疑】
【问题】：给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
* 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
* 如果 s 中存在这样的子串，我们保证它是唯一的答案。
```java
/**
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
*/
```
## 52.LeetCode 064最小路径和
时间复杂度：$O(mn)$
空间复杂度：$O(mn)$
```java
    public int minPathSum(int[][] grid) {
    int[][] dp = new int[grid.length][grid[0].length];
        dp[0][0]=grid[0][0];
        for (int j = 1; j < grid[0].length; j++) {
            dp[0][j] =  dp[0][j-1]+grid[0][j];
        }

        for (int i = 1; i < grid.length; i++) {
            for (int j = 1; j < grid[0].length; j++) {
                dp[i][0]= dp[i-1][0]+grid[i][0];
                int prev = Math.min(dp[i-1][j],dp[i][j-1]) ;
                dp[i][j] = grid[i][j]+prev;
                
            }
        }
        return dp[grid.length-1][grid[0].length-1];

    }
```
## 53.LeetCode 105从前序与中序遍历序列构造二叉树
```java
/*
        preOrder = {1,2,4,3,6,7}
        inOrder = {4,2,1,6,3,7}

        根 1
            pre         in
        左  2,4         4,2
        右  3,6,7       6,3,7

        根 2
        左 4

        根 3
        左 6
        右 7
     */
 public TreeNode buildTree(int[] preOrder, int[] inOrder) {
        if(preOrder.length==0 ||inOrder.length==0){
            return null;
        }
        //创建根节点
        int rootValue = preOrder[0];//根节点的值一定是前序的第一个值
        TreeNode root = new TreeNode(rootValue);
        //区分左右子树
        for (int i = 0; i < inOrder.length; i++) {
            if(inOrder[i] == rootValue){//找到根节点
                //0 ~ i-1 即左子树，i+1 ~ inorder.length-1即右子树
                //copyofRange是左闭右开的
                int[] inLeft = Arrays.copyOfRange(inOrder, 0, i);//4,2
                int[] inRight = Arrays.copyOfRange(inOrder, i+1, inOrder.length);//6,3,7

                int[] preLeft = Arrays.copyOfRange(preOrder, 1, i + 1);//2,4
                int[] preRight = Arrays.copyOfRange(preOrder, i+1, inOrder.length);//3,6,7

                root.left= (TreeNode) buildTree(preLeft,inLeft);//2
                root.right= (TreeNode) buildTree(preRight,inRight);//3
                break;
            }
        }
        return root;
    }
```
## 54.LeetCode 078子集
【回溯法】
```java
public class E078subsets {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new LinkedList<>();
        backtrack(nums,0,new LinkedList<>(),result);
        return result;
    }

    private void backtrack(int[] nums, int index, LinkedList<Integer> subset, List<List<Integer>> result) {
        if(index == nums.length){
            result.add (new LinkedList<> (subset));
        }
        else if(index < nums.length){

            backtrack(nums,index+1,subset,result);
            subset.add(nums[index]);
            backtrack(nums,index+1,subset,result);
            subset.removeLast();
        }
    }
}
```
## 【关联题】77.组合
【问题】：`给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
你可以按 任何顺序 返回答案。`
```java
class Solution {
    public List<List<Integer>> combine(int n, int k) {
        LinkedList<Integer> subset = new LinkedList<>();
        List<List<Integer>> res = new LinkedList<>();
        helper(n,k,1, subset, res);
        return res;

    }

    private void helper(int maxNum,int k,int i,
    LinkedList<Integer> subset,List<List<Integer>> result){
        if(subset.size() == k){
            result.add(new LinkedList<>(subset));
        }else if(i<=maxNum){
            helper(maxNum,k,i+1,subset,result);

            subset.add(i);
            helper(maxNum,k,i+1,subset,result);
            subset.removeLast();
        }
    }
}
```
## 55.LeetCode 155最小栈
两个栈模拟最小栈
```java
class MinStack {

    Stack<Integer> stackData = new Stack<>();
    Stack<Integer> stackMin = new Stack<>();

    public MinStack() {
        return;
    }
    
    public void push(int val) {
        if(this.stackMin.isEmpty()) {
            this.stackMin.push(val);
        } else if(val <= this.getMin()) {
            this.stackMin.push(val);
        }
        this.stackData.push(val);
    }
    
    public void pop() {
        if(this.stackData.isEmpty()) {
            throw new RuntimeException("your stack is empty!");
        }
        int value = this.stackData.pop();
        if(value == this.getMin()) {
            this.stackMin.pop();
        }
    }
    
    public int top() {
        if(this.stackData.isEmpty()) {
            throw new RuntimeException("your stack is empty!");
        }
        return this.stackData.peek();
    }
    
    public int getMin() {
        if(this.stackMin.isEmpty()) {
            throw new RuntimeException("your stack is empty!");
        }
        return this.stackMin.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(val);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
```

```java

import java.util.Stack;

public class Solution {
    //用于栈的push 与 pop
    Stack<Integer> s1 = new Stack<Integer>(); 
    //用于存储最小min
    Stack<Integer> s2 = new Stack<Integer>(); 
    public void push(int node) {
        s1.push(node);  
        //空或者新元素较小，则入栈
        if(s2.isEmpty() || s2.peek() > node)  
            s2.push(node);
        else
            //重复加入栈顶
            s2.push(s2.peek());  
    }
    
    public void pop() {
        s1.pop();
        s2.pop();
    }
    
    public int top() {
        return s1.peek();
    }
    
    public int min() {
        return s2.peek();
    }
}

```
【关联题】最大栈
```java
class MaxStack {

    private final Stack<Integer> s1;
    private final Stack<Integer> s2;

    public MaxStack() {
        s1 = new Stack<>();
        s2 = new Stack<>();
    }
    
    public void push(int x) {
        s2.push(x);
        if(this.s1.isEmpty()){
            this.s1.push(x);
        }else{
             if(x > s1.peek()){
            this.s1.push(x);
             }else{
            this.s1.push(s1.peek());
        }
        }
    }
    
    public int pop() {
        s1.pop();
        return this.s2.pop();
    }
    
    public int top() {
        return this.s2.peek();
    }
    
    public int peekMax() {
        return this.s1.peek();
    }
    
    public int popMax() {
        // 找到要弹出的元素的值
        int max = this.pop();
        // 暂时存储在最大元素上方的元素
        Stack<Integer> tmp = new Stack<>();
        while(!s2.peek().equals(s1.peek())){
            tmp.push(this.pop());
        }
// 再把temp中的元素放进stack1，同时更新stack2即可，使用上面写好的方法直接调用即可。
        while(!tmp.isEmpty()){
            this.push(tmp.pop());
        }
        return max;
    }
}
/**
 * Your MaxStack object will be instantiated and called as such:
 * MaxStack obj = new MaxStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.peekMax();
 * int param_5 = obj.popMax();
 */
```
## 56.LeetCode 322 零钱问题
【问题】：给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
你可以认为每种硬币的数量是无限的。
```java
class Solution {
    public int coinChange(int[] coins, int amount) {
      int[] dp = new int[amount+1];//amount+1是因为最多由amount+1个1元硬币组成
      Arrays.fill(dp,amount+1);
      dp[0]=0;
      //外层for循环遍历所有状态的所有取值
        for (int i = 0; i < dp.length; i++) {
            //内层for循环求所有选择的最小值
            for (int coin: coins
                 ) {
                //子问题无解，跳过
                if(i-coin<0) continue;
                dp[i] = Math.min(dp[i],dp[i-coin]+1);
            }
        }
        return (dp[amount] ==amount+1) ? -1:dp[amount];
    }
}
```
## 57.LeetCode 151反转字符串
【双指针】
```java
/*
输入：s = "the sky is blue"
输出："blue is sky the"
*/
class Solution {
    public String reverseWords(String s) {
        s = s.trim();                                    // 删除首尾空格
        int j = s.length() - 1, i = j;
        StringBuilder res = new StringBuilder();
        while (i >= 0) {//i、j都从尾部出发
            while (i >= 0 && s.charAt(i) != ' ') i--;     // 搜索首个空格
            res.append(s.substring(i + 1, j + 1) + " "); // 添加单词
            while (i >= 0 && s.charAt(i) == ' ') i--;     // 跳过单词间空格
            j = i;                                       // j 指向下个单词的尾字符
        }
        return res.toString().trim();                    // 转化为字符串并返回
    }
}

```
## 58.LeetCode 014最长公共前缀
```java
class Solution {
    public String longestCommonPrefix(String[] strs) {
        String start = strs[0];
        for (int i = 0; i < start.length(); i++) {//strs[0]开始横向循环每个char
            char ch = start.charAt(i);
            for (int j = 1; j <strs.length ; j++) {//循环strs每个string内是否有char(i)
                //case1:当遇到strs[j]等于循环到i了，不能继续了就返回（0，i）
                //case2:当遇到strs[j]等于循环到不匹配char(i)了，不能继续了就返回（0，i）
                if(strs[j].length() == i||ch!=strs[j].charAt(i)){
                    return new String(start.toCharArray(),0,i);
                }
            }
            
        }
        //case3::外层循环自然结束，即数组第一个循环完了，minLen
        return strs[0];

    }
}
```

## 59.LeetCode 121股票买卖问题
**【问题1】：121买卖股票的最佳时机 I**
你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
【方法1】：动态规划
```java
class Solution {
    public int maxProfit(int[] prices) {
        //暴力解法
        int cost = Integer.MAX_VALUE,profit=0;//初始化
        for(int price : prices){
            cost =Math.min(cost,price);
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }
}
```
【方法2】：贪心
```java
public int maxProfit(int[] prices) {
       int buyday = 0;
       int sellday =1;
       int max=0;
       while(sellday<prices.length){
           if(prices[sellday]-prices[buyday]>0){//涨
               max = Math.max(max,prices[sellday]-prices[buyday]);
               sellday++;
           }else{//跌
               buyday=sellday;
               sellday++;
           }
       }
       return max;
    }
```
**【问题2】：122买卖股票的最佳时机 II**
在每一天，你可以决定是否购买和/或出售股票。你在任何时候 最多 只能持有 一股 股票。你也可以先购买，然后在 同一天 出售。

返回 你能获得的 最大 利润 。
/*输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3 。
     总利润为 4 + 3 = 7 。*/
【方法】：贪心算法（不管未来如何，当下有钱就赚）
```java
class Solution {
    public int maxProfit(int[] prices) {
    int buyday = 0;
       int sellday =1;
       int max=0;
       while(sellday<prices.length){
           if(prices[sellday]-prices[buyday]>0){//涨
               max += prices[sellday]-prices[buyday];
               buyday=sellday;
               sellday++;
           }else{//跌
               buyday=sellday;
               sellday++;
           }
       }
       return max;
    }
}
```

**【问题3】：714买卖股票的最佳时机含手续费**
/*你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。
返回获得利润的最大值。
注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费.*/
【方法】：动态规划
```java
    public int maxProfit(int[] prices, int fee) {
    //初始化    
        int[] buy =new int[prices.length];
        int[] sell= new int[prices.length];

        buy[0] = -prices[0];
        sell[0] = 0;

        for (int i = 1; i <prices.length ; i++) {
            //buy选max：1.延续上次买的利润不变；2.在上次卖的利润基础上买
            buy[i] = Math.max(buy[i-1],sell[i-1]-prices[i]);
            //sell选max：1.延续上次卖的利润不变；2.在上次买的利润基础上卖
            sell[i] = Math.max(sell[i-1],buy[i-1] +prices[i]-fee);
        }
        return sell[prices.length-1];
    }
```
## 60.LeetCode 240搜索二维矩阵

/*编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。*/
```java
    public boolean searchMatrix(int[][] matrix, int target) {
        int row = matrix.length;
        int col = matrix[0].length;//
        //从左下角开始找
        // 如果target>matrix[row][0],向右走
        // 如果target<matrix[row][0],向上走
        int i =row-1;
        int j =0;
        while(i>=0 && j<col){
                if(matrix[i][j] == target){
                    return true;
                }else if(matrix[i][j] > target){
                    i--;
                }else{
                    j++;
                }
            
        }
        return false;
    }
```
## 61.LeetCode 221最大正方形【存疑】
【方法】：动态规划

## 62.LeetCode 034在排序数组中查找元素的第一个和最后一个位置
【方法】：二分法（leftmost+rightmost）
/*输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]*/

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[]{-1,-1};
        res[0] = leftmost(nums,target);
        res[1] = rightmost(nums,target);
        return res;

    }
    private int leftmost(int[] nums, int target){
       int candidate=-1;
       int l = 0;
       int r = nums.length-1;
       while(l<=r){
           int m = (l+r)>>>1;
         if(nums[m]> target){
               r=m-1;
           }else if(nums[m]< target){
               l=m+1;
           }else{
               candidate=m;
               r=m-1;
           }
       }
       return candidate;
    }
    private int rightmost(int[] nums, int target){
        int candidate=-1;
        int l = 0;
        int r = nums.length-1;
        while(l<=r){
            int m = (l+r)>>>1;
            if(nums[m]> target){
                r=m-1;
            }else if(nums[m]< target){
                l=m+1;
            }else{
                candidate=m;
                l=m+1;
            }
        }
        return candidate;
        }
}
```
## 63.LeetCode112.路径总和（树）
【方法】：递归
/*输入：root = [1,2,3], targetSum = 5
输出：false*/
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        //base case
        if(root == null){
            return false;
        }
        if(root.left ==null&& root.right==null){
            return targetSum -root.val ==0;
        }
        return hasPathSum(root.left,targetSum-root.val)||hasPathSum(root.right,targetSum-root.val);

    }
}
```
## 64.LeetCode234.回文链表
【方法】：栈
```java
    public boolean isPalindrome(ListNode head) {
        //base case
         if (head == null) {
            return true;
        }
        ListNode p = head;
        Stack<ListNode> stack = new Stack<>();

        while(p!=null){
            stack.push(p);
            p=p.next;
        }
        p =head;
        while(p!=null){
            if(stack.pop().val != p.val){
                return false;
            }
            p=p.next;
        }
        
        return stack.isEmpty();
}
```
## 65.LeetCode013.罗马数字转整数
/*例如， 罗马数字 2 写做 II ，即为两个并列的 1 。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。*/
```java
class Solution {
     public int romanToInt(String s) {
        int sum =0;
        int preNum= getValue(s.charAt(0));
        for (int i = 1; i < s.length(); i++) {
            int cur = getValue(s.charAt(i));
            if(preNum<cur){
                sum -= preNum;
            }else {
                sum += preNum;
            }
            preNum=cur;
        }
        sum+=preNum;
        return sum;
    }
    private  int getValue(char ch){
      int i =  switch (ch){
            case 'I'-> 1;
            case 'V'->5;
          case 'X'->10;
          case 'L'-> 50;
          case 'C'->100;
          case 'D'-> 500;
          case 'M'-> 1000;
          default -> 0;
        };

        return i;
    }
}
```
## 66.LeetCode 283.移动零
```java
class Solution {
	public void moveZeroes(int[] nums) {
		if(nums==null) {
			return;
		}
		//第一次遍历的时候，j指针记录非0的个数，只要是非0的统统都赋给nums[j]
		int j = 0;
		for(int i=0;i<nums.length;++i) {
			if(nums[i]!=0) {
				nums[j] = nums[i];
				j++;
			}
		}
		//非0元素统计完了，剩下的都是0了
		//所以第二次遍历把末尾的元素都赋为0即可
		for(int i=j;i<nums.length;++i) {
			nums[i] = 0;
		}
	}
}	

```
## 67.LeetCode 394. 字符串解码【存疑】

## 68.LeetCode 110. 平衡二叉树
【方法】：递归
多益网络笔试题
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    boolean flag =true;
    public boolean isBalanced(TreeNode root) {

        maxDepth(root);
        return flag;

    }

    private int maxDepth(TreeNode root){
        if(root ==null){
            return 0;
        }
        int l = maxDepth(root.left);
        int r = maxDepth(root.right);

        if(Math.abs(l-r)>1){
            flag =false;
        }
        return Math.max(l,r)+1;
    }
}
```
## 【关联题】101. 对称二叉树
给你一个二叉树的根节点 root ， 检查它是否轴对称。
```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return root == null || recur(root.left,root.right);
    }

    private boolean recur(TreeNode L,TreeNode R){
        if( L ==null && R==null){
            return true;
        }
        if(L==null || R==null || L.val !=R.val){
            return false;
        }

        return recur(L.left,R.right) && recur (L.right , R.left);
    }
}
```
## 【关联题】543. 二叉树的直径
```java
class Solution {
    private int ans; 
    public int diameterOfBinaryTree(TreeNode root) {
       maxDepth(root);
        return  ans ;

    }

    private int maxDepth(TreeNode root){
        if(root == null){
            return 0;
        }
        int l = maxDepth(root.left);
        int r = maxDepth(root.right);
        ans = Math.max(ans,l+r);
        return Math.max(l,r)+1;
    }

}
```
## 69.LeetCode 009.回文数字
【知识点】：整型转string—— `String str = String.valueOf(x);`
string转char——`char[] array = str.toCharArray();`
```java
/*
        输入：x = -121
        输出：false
        
        输入：x = 121
        输出：true
         */
        public boolean isPalindrome(int x) {
            String str = String.valueOf(x);
            char[] array = str.toCharArray();
            if(array[0] == '-'){
                return false;
            }
            if(array.length==1){
                return true;
            }
            int i = 0;
            int j = array.length-1;
            while(i<j){
                if(array[i] != array[j]){
                    return false;
                }
                i++;
                j--;
            }
            return true;
        }
```
## 70.LeetCode 027.原地移除元素
```java
/*
输入：nums = [3,2,2,3], val = 3
输出：2, 因为nums = [2,2]
*/
class Solution {
    public int removeElement(int[] nums, int val) {
    
    int ans =0;
    for(int num:nums){
        if(num!=val){
            nums[ans]=num;
            ans++;
        }
    }
    return ans;

    }
}
```

## 71.LeetCode 217.存在重复元素
【问题】：给你一个整数数组 nums 。如果任一值在数组中出现 至少两次 ，返回 true ；如果数组中每个元素互不相同，返回 false 。
【方法一】：Hashtable
```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        HashMap<Integer,Integer> mp = new HashMap<>();
        for(int i = 0; i<nums.length;i++){
            if(!mp.containsKey(nums[i])){
                mp.put(nums[i],1);
            }else{
                mp.put(nums[i],mp.get(nums[i])+1);
            }
        }
        for(int i = 0; i<nums.length;i++){
            if(mp.get(nums[i])>=2){
                return true;
            }
        }
        return false;
    }
}
```
【方法二】：排序比较相邻元素
```java
    public boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums);
        for(int i =0;i<nums.length-1;i++){
            if(nums[i]==nums[i+1]){
                return true;
            }
        }
        return false;
    }
```

## 72.LeetCode 159. 库存管理 III
【问题】：仓库管理员以数组 stock 形式记录商品库存表，其中 stock[i] 表示对应商品库存余量。请返回库存余量最少的 cnt 个商品余量，返回 顺序不限。
【方法】：排序+取值；**快排**
```java
/*
输入：stock = [0,2,3,6], cnt = 2
输出：[0,2] 或 [2,0]
*/
class Solution {
    public int[] inventoryManagement(int[] stock, int cnt) {
        quick(stock,0,stock.length-1);
        return Arrays.copyOf(stock,cnt);

    }
     private static void quick(int[] a, int left, int right) {
        if (left >= right) {
            return;
        }
        int pv =a[left];
        int i = left;
        int j = right;
        while(i<j){
            //1.j从右向左，找比基准点小的值
            while(i<j && a[j] > pv){
                j--;
            }
            //2.i从左到右，找大的值
            while(i<j && a[i] < pv){
                i++;
            }
            //3.交换位置
            swap(a,i,j);

        }
        swap(a,i,left);
        quick(a, left, i - 1);//分别排左边
        quick(a, i + 1, right);//分别排右边
    }

    private static void swap(int[] a, int i, int j) {
        int tmp = a[i];
        a[i] = a[j];
        a[j] =tmp;
    }
}
```

## 73.LeetCode 174. 寻找二叉搜索树中的目标节点
/*
输入：root = [7, 3, 9, 1, 5], cnt = 2
       7
      / \
     3   9
    / \
   1   5
输出：7, 倒数第二大为7
*/
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    int index =0 ;
    int res =-1;
    public int findTargetNode(TreeNode root, int cnt) {
        dfs(root,cnt);
        return res;
    }

    private void dfs(TreeNode root,int cnt){
        if(root==null){
            return;
        }
        dfs(root.right,cnt);//中序遍历的倒序，即右中左
        index++;
        if(index==cnt ){
            res = root.val;
        }
        dfs(root.left,cnt);
    }
}
```
## 【关联题】98. 验证二叉搜索树
【问题】：给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
有效二叉搜索树定义如下：
* 节点的左子树只包含 小于 当前节点的数。
* 节点的右子树只包含 大于 当前节点的数。
* 所有左子树和右子树自身必须也是二叉搜索树。
```java
class Solution {
        long prev= Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root){
        if(root == null){
            return true;
        }
        boolean isLeft = isValidBST(root.left);
        if( root.val<=prev){
            return false;
        }
        prev = root.val;//中序遍历
        boolean isRight = isValidBST(root.right);
        return isLeft && isRight;
    }
}
```
## 74.LeetCode 026. 删除数组重复元素
```java
class Solution {
    public int removeDuplicates(int[] nums) {
    //双指针
    /*
    输入：nums = [0,0,1,1,1,2,2,3,3,4]
    输出：5, nums = [0,1,2,3,4]
    解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。
    */
    if(nums == null || nums.length ==0){
        return 0;
    }
    int i=0;
    int j=1;

    while(j<nums.length){
        if(nums[i]!=nums[j]){
            nums[i+1] = nums[j];
            i++;
        }
            j++;
    }
     return i+1;
    }
}
```

## 75.LeetCode 1056. 易混淆数
## 76.LeetCode 1427. 字符串的左右移
```java
class Solution {
    public String stringShift(String s, int[][] shift) {//abc,[[0,1],[1,2]]
    /*
    输入：s = "abc", shift = [[0,1],[1,2]]
    输出："cab"
    解释：
    [0,1] 表示左移 1 位。 "abc" -> "bca"
    [1,2] 表示右移 2 位。 "bca" -> "cab"
    */
        int m = shift.length;//?
        int n = s.length();//2
        int offset=0;

        for(int i =0;i<m;i++){
            offset = shift[i][1]%n;//1|2
            if(shift[i][0]==0){
                //"unhappy".substring(2) returns "happy"
                //"smiles".substring(1, 5) returns "mile"
                s =s.substring(offset)+s.substring(0,offset);
            }else{
                s=s.substring(n-offset)+s.substring(0,n-offset);
            }
        }
        return s;


    }
}
```

## 77.LeetCode 039.允许重复选择元素的组合
【回溯法】
【54.LeetCode 078子集】
```java
class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
//回溯
        List<List<Integer>> res = new LinkedList<>();
        helper(candidates,target,0,new LinkedList<>(),res);
        return res;

    }
    private void helper(int[] nums,int target,int i,LinkedList<Integer> subset,List<List<Integer>> result){
        if(target==0){
            result.add(new LinkedList<>(subset));

        }else if( target>0 && i<nums.length){
            helper(nums,target,i+1,subset,result);
            subset.add(nums[i]);
             helper(nums,target-nums[i],i,subset,result); 
             subset.removeLast();
        }
    }
}
```

## 78.LeetCode 131.分割回文字符串
【问题】：` 输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]`
```java
class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> res = new LinkedList<>();
        helper(s,0,new LinkedList<>(),res);
        return res;
    }

    private void helper(String s, int index ,LinkedList<String> subset, List<List<String>> result){
        if(index == s.length()){
            result.add(new LinkedList<>(subset));
        }else if(index < s.length()){
            for(int i = index; i<s.length();i++){
                if(isPalindrome(s,index,i)){
                subset.add(s.substring(index,i+1));
                helper(s,i+1,subset,result);
                subset.removeLast();
                }           
            }
        }
    }

    private boolean isPalindrome(String s,int start,int end){
        while(start<end){
            if(s.charAt(start)!=s.charAt(end)){
                return false;
            }
            start++;
            end--;
        }
        return true;
    }
}
```
**【关联题】132. 分割回文串 II**
【问题】:`输入：s = "aab"
输出：1
解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。`


## 【关联题】LeetCode 093. 复原 IP 地址
【问题】：`输入：s = "25525511135"
输出：["255.255.11.135","255.255.111.35"]`
```java
class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new LinkedList<>();
        helper(s,0,0,"","",res);
        return res;

    }
    private void helper(String s ,int index, int segI,String seg, String ip, List<String> result){
        if(index==s.length() && segI ==3 && isValid(seg)){
            result.add(ip+seg);
        }else if(index<s.length() && segI<=3){
            char ch = s.charAt(index);
            if(isValid(seg+ch)){
                helper(s,index+1,segI,seg+ch,ip,result);
            }
            if(seg.length()>0 && segI<3){
                helper(s,index+1,segI+1,""+ch,ip+seg+".",result);
            }
        }
    }
    private boolean isValid(String seg){
        
        return Integer.valueOf(seg) <=255
        && (seg.equals("0") || seg.charAt(0) != '0');
    }
}
```

## 79.LeetCode 32. 最长有效括号
```java
/**
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
*/
class Solution {
    public int longestValidParentheses(String s) {
      Stack<Integer> stack = new Stack<>();
        int max = 0;
        int start = 0;
        for (int i = 0; i < s.length(); i++) {//"()(()"
            if (s.charAt(i) == '(') {
                stack.push(i);//左括号压入栈
            } else {//右括号
                // 栈为空说明组成不了有效括号，寻找下一个字串
                if (stack.isEmpty()) {
                    start = i + 1;//
                } else {
                    stack.pop();//
                    max = Math.max(max, stack.isEmpty() ? i - start + 1 : i - stack.peek());
                }
            }
        }
        return max;
    }
}
```

## 80.LeetCode 247.中心对称数II
【问题】：给定一个整数 n ，返回所有长度为 n 的 中心对称数 。你可以以 任何顺序 返回答案。
中心对称数 是一个数字在旋转了 180 度之后看起来依旧相同的数字（或者上下颠倒地看）。
```java
/**
输入：n = 2
输出：["11","69","88","96"]
*/
class Solution{
    public List<String> findStrobogrammatic(int n) {
        return helper(n,n);
    }
    // n表示，当前循环中，求得字符串长度； m表示题目中要求的字符串长度
    public List<String> helper(int n, int m){
        // 第一步：判断输入或者状态是否合法
        if(n<0 || m<0 || n>m){
            throw  new IllegalArgumentException("invalid input");
        }
        // 第二步：判断递归是否应当结束
        if (n==0)
            return new ArrayList<>(Arrays.asList(""));
        if (n==1)
            return new ArrayList<>(Arrays.asList("0","1","8"));

        // 第三步：缩小问题规模
        List<String> list = helper(n-2, m);

        // 第四步：整合结果
        List<String> res = new ArrayList<>();
        for (String s : list){
            if (n!=m)
                // n=m时，表示最外层处理。
                // 例如：原始需求n=m=2, '00'不合法
                // 若原始需求n=m=4, 内层循环n=2,m=4,'00';最外层循环，n=m=4时，'1001'
                res.add("0"+s+"0");
            res.add("1"+s+"1");
            res.add("6"+s+"9");
            res.add("8"+s+"8");
            res.add("9"+s+"6");
        }
        return res;
    }
}

作者：汤圆
链接：https://leetcode.cn/problems/strobogrammatic-number-ii/
```

## 81.LeetCode 1228. 等差数列中缺失的数字
```java
//暴力法
class Solution {
    public int missingNumber(int[] arr) {
        int miss = Integer.MAX_VALUE;
        int i =0;
        int j = arr.length-1;
        int cha =0;
        if(arr[i]>arr[j]){
            cha = (arr[j]-arr[i])/arr.length;
        }else if(arr[i]<arr[j]){
            cha = (arr[j]-arr[i])/arr.length; 
        }else{
            return arr[0];
        }
   
        for(int k =1;k<=j;k++){
            if(arr[k]==arr[k-1]+cha){
                continue;
            }
            return arr[k-1]+cha;
        }
        return miss;
    }
}
```
```java
//二分法
class Solution {
    public int missingNumber(int[] arr) {
        int i = 0;
        int j = arr.length-1;
        int cha =(arr[j]-arr[i])/arr.length;

        while(i!=j-1){
            int mid =(i+j)>>>1;
            if(arr[mid]-arr[i] == cha * (mid -i)){
                i = mid;
            }else if(arr[j]-arr[mid] == cha * (j-mid)){
                j = mid;
            }

        }
        return (arr[i]+arr[j])/2;
    }
}
```