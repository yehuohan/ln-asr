
NGram
=====

N元语法模型（N-Gram Model）是根据前面出现的N-1个单词来猜测第N个单词的模型。

设置N个单词序列为 :math:`w_1^n = \lbrace w_1, w_2, ..., w_n \rbrace` ，其出现的概率为：

..  math::
    P(w_1^n) = \prod_{k-1}^n P(w_k | w_1^{k-1})


在N元语法模型下，认为第N个单词的概率根据与前面N-1个单词相关：

..  math::
    P(w_n | w_1^{n-1}) = P(w_n | w_{n-N+1}^{n-1})

使用最大似然估计（MLE）来估计NGram的概率，对于二元语法，概率估计为：

..  math::
    P(w_n | w_1^{n-1}) = \frac{C(w_{n-1} w_n)}
                              {\sum_w C(w_{n-1} w)}


平滑算法
--------

平滑算法是为了处理NGram模型中计数为零(:math:`c_i = 0` )的问题。

..  math::
    P(w_i) = \frac{c_i}{N}

- 加一平滑

即每个计数加一。

..  math::
    P(w_1) = \frac{c_1 + 1}{N + V}

例如，对于一元语法有w1有：

::

    w1 : 3
    w2 : 5
    w3 : 7
    w4 : 0

    ci = 3, N = 3 + 5 + 7 = 14, V = 3


例如，对于二元语法的w1w2有：

::

    w1w1 : 3
    w1w2 : 2
    w1w3 : 7
    w1w4 : 0
    w2w5 : 9

    ci = 2, N = 3 + 2 + 7 = 12, V = 4

- 打折平滑（Good-Turing）

设出现次数为 :math:`c` 的词的个数为 :math:`N_c` ，则 :math:`N_c` 的概率（出现 :math:`c` 次的所有词的概率）为 :math:`\frac{c}{N}` 。
打折即用平滑计数 :math:`c^*` 代替 :math:`c` ：

..  math::
   c^* = (c+1)\frac{N_{c+1}}{N_c}

:math:`N_c` 的概率为：

..  math::
    P_c^* =
    \begin{cases}
    N_1 / N, &c=0 \\
    c^* / N, &c>0 \\
    \end{cases}

例如，对于如下一元计数有：

::

    w1 : 10
    w2 : 3
    w3 : 2
    w4 : 1
    w5 : 1
    w6 : 1
    w7 : 0
    w8 : 0

..  math::
    P(w_7) = P(w_8) &= P_0^* / 2 &= (3 / 18) / 2 \\
    p(w_4) = P(w_5) = P(w_6) &= P_1^* &= \left[ (1+1) \frac{1}{3} \right] / 18


- 回退法（Katz Backoff）

当N元语法的的计数为零时，回到(N-1)元语法来计算：

..  math::
    P(w_n | w^{n-1}_{n-N+1}) =
    \begin{cases}
    P^*(w_n | w^{n-1}_{n-N+1}), &C(w^n_{n-N+1}) > 0 \\[2ex]
    \alpha(w^{n-1}_{n-N+1}) P(w_n | w^{n-1}_{n-N+2}), &C(w^n_{n-N+1}) = 0\\
    \end{cases}

:math:`P^*` 为打折概率，要于小于MLE概率。
由于概率之和为1，可得 :math:`\alpha(w^{n-1}_{n-N+1})` 的计算式：

..  math::
    1 &= \sum_{C(w^n_{n-N+1}) > 0} P(w_n | w^{n-1}_{n-N+1}) + \sum_{C(w^n_{n-N+1}) = 0} P(w_n | w^{n-1}_{n-N+1}) \\
    &\Downarrow \\[2ex]
    \alpha(w^{n-1}_{n-N+1}) &= \frac{1 - \sum_{w_n:C(w^n_{n-N+1}) > 0} P(w_n | w^{n-1}_{n-N+1})}
                                    {\sum_{w_n:C(w^n_{n-N+1}) = 0} P(w_n | w^{n-1}_{n-N+2})} \\
    &\Downarrow \\[2ex]
    \alpha(w^{n-1}_{n-N+1}) &= \frac{1 - \sum_{w_n:C(w^n_{n-N+1}) > 0} P^*(w_n | w^{n-1}_{n-N+1})}
                                    {1 - \sum_{w_n:C(w^n_{n-N+1}) > 0} P^*(w_n | w^{n-1}_{n-N+2})}

注意：对于相同的语料库，若有 :math:`C(w^n_{n-N+1}) > 0` 则必有 :math:`C(w^n_{n-N+2}) > 0` ，但反过来，若有 :math:`C(w^n_{n-N+2}) > 0` ，不一定有 :math:`C(w^n_{n-N+1}) > 0`

:参考:

- `Speech and Language Processing <https://web.stanford.edu/~jurafsky/slp3>`_
- `github.com/KAMI-Wei/ml-model <https://github.com/KAMI-Wei/ml-model>`_
- `github.com/daandouwe/ngram-lm <https://github.com/daandouwe/ngram-lm>`_
- `github.com/adroitous/Naive-Bayes-Classifier-with-Katz-Backoff <https://github.com/adroitous/Naive-Bayes-Classifier-with-Katz-Backoff>`_
- `github.com/zzchua/ngram-language-model <https://github.com/zzchua/ngram-language-model>`_
- `漫谈 Language Model (2): 实践篇 <http://blog.pluskid.org/?p=361>`_
- `SRILM Ngram 折扣平滑算法 <https://www.cnblogs.com/dahu-daqing/p/9759978.html>`_
- `SRILM语言模型格式解读 <https://www.cnblogs.com/dahu-daqing/p/7449200.html>`_
