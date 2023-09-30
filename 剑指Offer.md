# 剑指Offer

## JZ4 二维数组中的查找
```java
public class JZ4Find {
    public boolean Find (int target, int[][] array) {
        // write code here
        int size = array.length;//二维数组的长度是行数
        int length = array[0].length;//二维数组第0列的长度
        int i,j;
        for ( i = size-1,j=0; i >=0 && i<size && j>=0 && j <length;) {
            if(array[i][j] == target){
                return true;
            }
            if(array[i][j] > target){
                i--;//向上走
                continue;
            }
            if(array[i][j] < target){
                j++;//向右走
                continue;
            }
        }
        return false;
    }
}
```

## JZ5 替换空格
```java
    public String replaceSpace (String s) {
        // write code here
//        String result = s.toString().replace(" ","%20");
//        return result;
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            if(s.charAt(i) ==' '){
                result.append("%20");
            }else{
                result.append(s.charAt(i));
            }
        }
        return result.toString();
    }
```

## JZ6 从尾到头打印链表
```java
public class Solution {
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {

        ArrayList<Integer> res =  new ArrayList<>();
        reverse(listNode, res);
        return res;

    }

    private void  reverse(ListNode head, ArrayList<Integer> res) {
        if (head != null) {
            reverse(head.next, res);
            res.add(head.val);
        }
    }
}
```

## JZ7 重建二叉树

## JZ8二叉树的下一节点
还原二叉树，中序遍历
```java
public class Solution {

    static ArrayList<TreeLinkNode> list = new ArrayList<>();
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        TreeLinkNode par = pNode;
        while (par.next!=null){
            par=par.next;
        }
        inorder(par);
        for (int i = 0; i < list.size(); i++) {
            if(i ==list.size()-1){
                return null;
            }
            if(pNode ==list.get(i)){
                return list.get(i+1);
            }
        }
        return null;
    }

    private void inorder(TreeLinkNode pNode){
        if(pNode !=null){
            inorder(pNode.left);
            list.add(pNode);
            inorder(pNode.right);
        }
    }
}

```
## 旋转数组的最小数字
时间复杂度O(logN)：N表示数组的长度，二分查找O(logN)
空间复杂度O(1)：仅使用常数（i, j, m）额外空间变量O(1)
```java
public class Solution {
    public int minNumberInRotateArray(int [] array) {
        // 特殊情况判断
        if (array.length== 0) {
            return 0;
        }
        // 左右指针i j
        int i = 0, j = array.length - 1;
        // 循环
        while (i < j) {
            // 找到数组的中点 m
            int m = (i + j) / 2;
            // m在左排序数组中，旋转点在 [m+1, j] 中
            if (array[m] > array[j]) i = m + 1;
            // m 在右排序数组中，旋转点在 [i, m]中
            else if (array[m] < array[j]) j = m;
            // 缩小范围继续判断
            else j--;
        }
        // 返回旋转点
        return array[i];
    }
}
```
## JZ15 二进制中1的个数
```java
 public int NumberOf1 (int n) {
        /*
        step 1：遍历二进制的32位，通过移位0-31次实现。
        step 2：将移位后的1与数字进行位与运算，结果为1就记录一次
        */
         int res = 0;
        //遍历32位
        for(int i = 0; i < 32; i++){
            //按位比较
            if((n & (1 << i)) != 0)  
                res++;
        }
        return res;
    }
```

## JZ17打印1到最大的n位数

```java
/*
输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。
*/
public class Solution {
    public int[] printNumbers (int n) {
        int len = (int) (Math.pow(10,n)-1);
        int[] res = new int[len];
        for(int i=0;i<len;i++){
            res[i]=i+1;
        }
        return res;
}
}
```

## JZ18删除链表的节点
```java
/*
输入：{2,5,1,9},5
返回值：{2,1,9}
*/
public ListNode deleteNode (ListNode head, int val) {
 if(head==null){
            return null;
        }
        if(head.val==val){
            return head.next;
        }
        // write code here
        ListNode prev = new ListNode(-1);
        prev.next=head;
        ListNode cur = head;

        while (cur!=null){
            if(cur.val == val){
                prev.next = cur.next;
                break;
            }
             prev = cur;
            cur=cur.next;

        }
        return head;
}
```

## JZ22链表中最后K个节点
```java
public ListNode FindKthToTail (ListNode pHead, int k) {
        // write code here
        ListNode p1=pHead;
        ListNode p2 = pHead;

        for (int i = 0; i < k; i++) {
            if(p2!=null){
                p2=p2.next;
            }else{
                return p1 =null;
            }
            
        }
        while (p2 != null){
            p1=p1.next;
            p2=p2.next;
        }

        return p1;
    }
```

## JZ31栈的压入、弹出序列
```java
/*
输入：[1,2,3,4,5],[4,5,3,2,1]
返回值：true
*/
    public boolean IsPopOrder (int[] pushV, int[] popV) {
        // write code here
        Stack<Integer> s1 = new Stack<>();
        int j =0;
        for (int i = 0; i < pushV.length; i++) {
            s1.push(pushV[i]);
            while(!s1.isEmpty() && s1.peek()==popV[j]){
                s1.pop();
                j++;
            }
        }

        return s1.isEmpty();
    }
```
## JZ32从上到下打印二叉树
```java
public class JZ32PrintFromTopToBottom {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
    /*
    输入：{8,6,10,#,#,2,1}
    返回值：[8,6,10,2,1]
     */
        ArrayList<Integer> list = new ArrayList<>();

        if (root == null) {
            return list;
        }
        // 1. 辅助队列
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        // 2. 根节点入队
        queue.offer(root);
        while (!queue.isEmpty()) {
            // 2.1 根节点出栈、队
            TreeNode node = queue.poll();
            list.add(node.val);
            // 2.2 坐边孩子入队
            if (node.left != null) {
                queue.offer(node.left);
            }
            // 2.2 右边孩子入队
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        return list;
    }
}
        
```
## JZ55 二叉树深度
```java
public class Solution {
    public int TreeDepth(TreeNode root) {
        if(root ==null){
            return 0;
        }
        int l = TreeDepth(root.left);
        int r = TreeDepth(root.right);
        return Math.max(l,r)+1;
    }
}

```


## JZ59 滑动窗口的最大值
【队列】：先进先出类型，可以考虑队列存储数据

## JZ68二叉搜索树的最小公共祖先
复杂度分析：基于二叉搜索策略，故时间复杂度$O(log n)$，空间复杂度$O(1）$
```java
public int lowestCommonAncestor (TreeNode root, int p, int q) {
		TreeNode curnode=root;//当前遍历结点
		while(true) {
			if(p<curnode.val&&q<curnode.val) curnode=curnode.left;//在左子树找
			else if(p>curnode.val&&q>curnode.val) curnode=curnode.right;//在右子树找
			else return curnode.val;
		}
	}

```
## JZ73翻转单词序列
public class JZ73ReverseSentence {
    /*
    输入："nowcoder. a am I"
    输出："I am a nowcoder."
     */
    public String ReverseSentence(String str) {
        Stack<String> stack = new Stack<>();
        String[] tmp = str.split(" ");

        for (int i = 0; i < tmp.length; i++) {
            stack.push(tmp[i]);
            stack.push(" ");
        }
        if (!stack.isEmpty()){
            stack.pop();//弹出最后一个空格
        }
        StringBuilder res = new StringBuilder();
        while(!stack.isEmpty()){
            res.append(stack.pop());
        }
        return res.toString();

    }
}