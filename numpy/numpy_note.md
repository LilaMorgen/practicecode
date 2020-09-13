<center>目录</center>

[toc]

# `NumPy`是什么?

`NumPy`是python中一个科学计算的基础包。它是一个python的库，为快速操作数组而提供了`多维数组对象`、各种派生对象`(如:masked array和矩阵)`和`各种程序`，它包括了`数学`、`逻辑`、`shape manipulation`、`排序`、`选择`、`I/O`、`离散(discrete)傅里叶变换`、`基础线性代数`、`基础数据操作`、`随机模拟`等等。

`NumPy`包的核心是`ndarray`对象。这是将同类(homogeneous)数据类型的n维数组进行封装(encapsulate)，为了性能(performance)，很多操作用已经编译(compile)好的代码运行。

`NumPy`数组与标准的python序列有一些不同：

- `NumPy`数组在创建的时候是固定(fixed)大小的，而python的列表大小是可以动态变化的。改变`ndarray`对象大小将会创建一个新的数组，并且将原来的数组删除。
- `NumPy`数组中的元素类型相同而且占内存中大小也相同。例外：一个可以有对象(如：python中的对象或者`NumPy`中的对象)的数组，它可以有占不同内存大小的元素。
- `NumPy`数组有助于在大量数据下进行高级的数学运算和其他类型操作。典型的是使用它可以更有效的执行操作，并且对于实现一样的功能，它比python用内置函数写的代码量更少。
- 越来越多基于python的科学和数学计算的包是使用`NumPy`数组。尽管它们支持python序列(列表)输入，但是它们可以在处理前将输入转换为`NumPy`数组，并且一般以`NumPy`数组输出。换句话说，要想有效使用当今绝大多数基于python的科学/数学计算的软件包，仅仅只知道使用python内置序列(列表)是不够的，必须也要知道`NumPy`数组的使用。

在科学计算中，关于序列的大小和速度这两点是尤其重要的。举一个简单的例子，考虑一维序列 中的元素与相对应的另一相同长度的序列中的元素相乘的情况。如果将数据储存到python的a和b两个列表中，那么我们将可以遍历每个元素：

```python
c = []
for i in range(len(a)):
    c.append(a[i] * b[i])
```

上面将产生正确的结果，但是如果列表a和b都有上百万个数据，那么在python中我们将花费低效率的循环。我们可以通过C语言编程来更快的完成相同的任务(为了清楚，以下我们忽略了变量的声明和初始化、内存分配等)：

```c
for (i = 0; i < rows; i++): {
    c[i] = a[i] * b[i];
}
```

这将节省解释python代码和操作python对象的时间，但是牺牲了从python代码中获得的好处。此外，所需的代码工作随着我们数据的维度的增加而增加。举个例子，在二维数组情况，用C语言编写:

```c
for (i = 0; i < rows; i++): {
    for (j = 0; j < columns; j++): {
        c[i][j] = a[i][j] * b[i][j];
    }
}
```

当涉及n维数组时，`NumPy`给了我们两全其美的方法：逐一元素操作和"默认模式"。但是逐一元素操作是通过提前编译C语言代码来快速执行的。在`NumPy`中：

```python
c = a * b
```

接近C语言编写的速度，但是随着代码的简化我们期望基于python运行。这最后一个例子表明了`NumPy`的两大特点(`矢量化`和`广播`)，这两大特点是其他功能的基础。

`矢量化`描述的是在代码中不需要显示的循环和索引等。这些东西只是发生在"幕后"(优化和编译好的C语言代码中)。矢量化代码有很多的优势，如下：

- 矢量化代码更简洁(concise)并且更容易读
- 更少量的代码意味着更少的BUG
- 这些代码更像是标准的数学代号(notation)(一般这样可以更轻松正确地编写数学表达式)
- 矢量化会产生更多的"`pythonic`"代码。没有矢量化，我们的代码将会因为循环的低效和难以阅读而被丢弃。

`广播`这一术语是用来描述隐式的逐一元素操作。通常来说，`NumPy`所有操作(不仅仅只有算数操作，还有逻辑操作、位操作和功能操作等等)都是以隐式的逐一元素操作运行的。此外，在上面的例子中，a和b可以是两个相同形状的数组，也可以是一个标量和一个数组，也可以是两个不同形状的数组，条件是较小形状的数组可以扩展到较大形状的数组的形状，这样才能保证广播结果正确。

`NumPy`完全支持面向对象方法(object-oriented approach)。举个例子,`ndarray`是一个类，它处理各种方法和属性。它的许多方法在`NumPy`的外层命名空间都表明了功能，给予了程序员完全自由的选择自己喜欢的范例进行或者选择最适合手头任务的方式进行编写代码。

# 安装`NumPy`

使用pip进行安装

```shell
pip install numpy
```

# `NumPy`教程

## 基础

`NumPy`的主要对象是同类的(homogeneous)多维数组。它是一个元素表(里面元素通常是数字)，所有元素都是同一类型，里面元素通过正整数索引。在`NumPy`中，维数(dimensions)被称为轴(axes)。

举个例子，在三维空间中点的坐标(coordinates)[1, 2, 3]有1个轴。这个轴有3个元素，所以说它的长为3。在下面的例子中，该数组有2个轴。第一个轴长度为2，第二个轴长度为3。

```python
[[1, 0, 0],
 [0, 1, 2]]
```

`NumPy`的数组类被称为`ndarray`。它也可以叫别名数组(`alias array`)。注意`numpy.array`类与python库中`array.array`类不同，`array.array`类只能处理一维数组并且只能提供很少的功能。以下是`ndarray`对象最重要的几个属性：

- `ndarray.ndim`

数组的轴数(维数)。

- `ndarray.shape`

数组的大小。这是一个整数元组，表明数组的大小。

对于一个n行(rows)，m列(columns)的矩阵，它的形状大小是(n, m)。因此它的形状元组的长度是轴数。

- `ndarray.size`

数组的元素总数。形状元组元素的乘积。

- `ndarray.dtype`

在数组中，描述元素类型的对象。可以使用标准的python类型创建或者指定数组元素的类型。另外，`NumPy`中提供了它自己的类型，如：`numpy.int32`、`nimpy.int16`、`numpy.float64`等。

- `ndarray.itemsize`

数组每个元素的所占字节大小。举个例子，数组元素的元素类型是`float64`，那么它所占字节大小是`8(=64/8)bytes`。当元素类型是`complex32`，那么它所占字节大小是`4(=32/8)bytes`。h换一种求法是`numpy.dtype.itemsize`。

- `ndarray.data`

包含了数组实际元素的缓冲区(内存地址，其地址一直在变动)。通常，我们不需要使用该属性，因为我们用索引工具就可以获取数组的元素。

### An example

```python
import numpy as np
a = np.arange(15).reshape(3, 5)
print(a)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
print(a.shape)  # (3, 5)
print(a.ndim)  # 2
print(a.dtype)  # class dtype
print(a.dtype.name)  # class str('int32')
print(a.itemsize)  # 4
print(a.size)  # 15
print(type(a))  # numpy.ndarray
```

### 数组创建

有几种方式去创建数组。

- `array`
- `zeros\ones\empty`
- `arange`

举个例子，通过使用`array`函数方法，你可以用常规的python列表或元组进行创建数组。这样数组元素的类型就由序列元素类型决定了。

```python
import numpy as np
a = np.array([2, 3, 4])
print(a)
print(a.dtype)  # int32
b = np.array([1.2, 3.5, 5.1])
print(b)
print(b.dtype)  # float64
```

数组的类型可以在创建的时候指定：

```python
c = np.array([[1, 2], [3, 4]], dtype=complex)
print(c)
# array([[1.+0.j, 2.+0.j],
#        [3.+0.j, 4.+0.j]])
```

通常我们有时候需要创建一个数组，虽然还没想好放什么元素，都是可以先定义大小。因此，`NumPy`提供了几个函数用初始占位符内容创建数组。这减小了成长数组的必要。

函数`zeros`创建一个全是0的数组，函数`ones`创建一个全是1的数组，函数`empty`创建一个初始内容是随机的数组并且依赖与内存状态。默认创建类型是`float64`。

```python
# zeros函数
z = np.zeros((2, 2))
print(z)
# array([[0., 0.],
#        [0., 0.]])
print(z.dtype)  # float64

# ones函数
o = np.ones((2, 3))
print(o)
# array([[1., 1., 1.],
#        [1., 1., 1.]])
print(o.dtype)  # float64

# empty函数
e = np.empty((2, 3))
print(e)
# array([[5.e-324, 5.e-324, 5.e-324],
#        [0.e+000, 0.e+000, 0.e+000]])
print(e.dtype)  # float64
```

创建一个数字序列，`NumPy`提供了类似与`range`的函数来创建数组。

```python
d = np.arange(10, 30, 5)
print(d)  # array([10, 15, 20, 25])
print(d.dtype)  # int32
f = np.arange(0, 2, 0.3)
print(f)  # array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
print(f.dtype)  # float64
```

由于函数`arange`如果传入了浮点数，元素的个数不好确定。因此这时使用函数`linspace`可以传入你想创建的数组元素个数的参数。

```python
from numpy import pi
g = np.linspace(0, 2, 9)
print(g)  # array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])
print(g.dtype)  # float64
x = np.linspace(0, 2*pi, 100)
print(x)
'''
array([0.        , 0.06346652, 0.12693304, 0.19039955, 0.25386607,
       0.31733259, 0.38079911, 0.44426563, 0.50773215, 0.57119866,
       0.63466518, 0.6981317 , 0.76159822, 0.82506474, 0.88853126,
       0.95199777, 1.01546429, 1.07893081, 1.14239733, 1.20586385,
       1.26933037, 1.33279688, 1.3962634 , 1.45972992, 1.52319644,
       1.58666296, 1.65012947, 1.71359599, 1.77706251, 1.84052903,
       1.90399555, 1.96746207, 2.03092858, 2.0943951 , 2.15786162,
       2.22132814, 2.28479466, 2.34826118, 2.41172769, 2.47519421,
       2.53866073, 2.60212725, 2.66559377, 2.72906028, 2.7925268 ,
       2.85599332, 2.91945984, 2.98292636, 3.04639288, 3.10985939,
       3.17332591, 3.23679243, 3.30025895, 3.36372547, 3.42719199,
       3.4906585 , 3.55412502, 3.61759154, 3.68105806, 3.74452458,
       3.8079911 , 3.87145761, 3.93492413, 3.99839065, 4.06185717,
       4.12532369, 4.1887902 , 4.25225672, 4.31572324, 4.37918976,
       4.44265628, 4.5061228 , 4.56958931, 4.63305583, 4.69652235,
       4.75998887, 4.82345539, 4.88692191, 4.95038842, 5.01385494,
       5.07732146, 5.14078798, 5.2042545 , 5.26772102, 5.33118753,
       5.39465405, 5.45812057, 5.52158709, 5.58505361, 5.64852012,
       5.71198664, 5.77545316, 5.83891968, 5.9023862 , 5.96585272,
       6.02931923, 6.09278575, 6.15625227, 6.21971879, 6.28318531])
'''
y = np.sin(x)
print(y)
'''
array([ 0.00000000e+00,  6.34239197e-02,  1.26592454e-01,  1.89251244e-01,
        2.51147987e-01,  3.12033446e-01,  3.71662456e-01,  4.29794912e-01,
        4.86196736e-01,  5.40640817e-01,  5.92907929e-01,  6.42787610e-01,
        6.90079011e-01,  7.34591709e-01,  7.76146464e-01,  8.14575952e-01,
        8.49725430e-01,  8.81453363e-01,  9.09631995e-01,  9.34147860e-01,
        9.54902241e-01,  9.71811568e-01,  9.84807753e-01,  9.93838464e-01,
        9.98867339e-01,  9.99874128e-01,  9.96854776e-01,  9.89821442e-01,
        9.78802446e-01,  9.63842159e-01,  9.45000819e-01,  9.22354294e-01,
        8.95993774e-01,  8.66025404e-01,  8.32569855e-01,  7.95761841e-01,
        7.55749574e-01,  7.12694171e-01,  6.66769001e-01,  6.18158986e-01,
        5.67059864e-01,  5.13677392e-01,  4.58226522e-01,  4.00930535e-01,
        3.42020143e-01,  2.81732557e-01,  2.20310533e-01,  1.58001396e-01,
        9.50560433e-02,  3.17279335e-02, -3.17279335e-02, -9.50560433e-02,
       -1.58001396e-01, -2.20310533e-01, -2.81732557e-01, -3.42020143e-01,
       -4.00930535e-01, -4.58226522e-01, -5.13677392e-01, -5.67059864e-01,
       -6.18158986e-01, -6.66769001e-01, -7.12694171e-01, -7.55749574e-01,
       -7.95761841e-01, -8.32569855e-01, -8.66025404e-01, -8.95993774e-01,
       -9.22354294e-01, -9.45000819e-01, -9.63842159e-01, -9.78802446e-01,
       -9.89821442e-01, -9.96854776e-01, -9.99874128e-01, -9.98867339e-01,
       -9.93838464e-01, -9.84807753e-01, -9.71811568e-01, -9.54902241e-01,
       -9.34147860e-01, -9.09631995e-01, -8.81453363e-01, -8.49725430e-01,
       -8.14575952e-01, -7.76146464e-01, -7.34591709e-01, -6.90079011e-01,
       -6.42787610e-01, -5.92907929e-01, -5.40640817e-01, -4.86196736e-01,
       -4.29794912e-01, -3.71662456e-01, -3.12033446e-01, -2.51147987e-01,
       -1.89251244e-01, -1.26592454e-01, -6.34239197e-02, -2.44929360e-16])
'''
```

### `NumPy`中相关创建数组函数的详细介绍

#### `numpy.array`

功能：创建一个数组。

```python
numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
```

Parameters :

`object` : array_like(类数组)

一个数组、任何暴露数组接口的对象、一个由`__array__`方法创建的对象或者是任何(嵌套的)序列。

`dtype` : data-type, optional(数据类型，可选填)

描述了数组的数据类型。如果未给出，那么将确定它的数据类型为序列中保存数据的最小类型。这个参数通常可以用创建数组。对于复制数组，并且重新指定类型，使用`astype(t)`方法。

```python
import numpy as np
x = np.array([[1, 2, 3], [4, 5, 6]])
print(x)
# array([[1, 2, 3],
#        [4, 5, 6]])
print(x.dtype)  # int32
y = x.astype('int64')
print(y)
# array([[1, 2, 3],
#        [4, 5, 6]])
print(y.dtype)  # int64
```

`copy` : bool, optional(布尔值，可选填)

如果是真(默认)，那么这个对象被复制。否则，只有当`__array__`返回一个副本或对象是一个嵌套序列又或是需要一个副本来满足其他要求(如：dtype、order等)。

`order` : {'K', 'A', 'C', 'F'}, optional(选择集合中一种，默认是'K')

指定数组的内存布局(layout)。如果对象不是一个数组，那么新创建的数组将是'C'序列(行主要)而不是'F'序列，'F'序列即Fortran序列(列主要)。如果对象是数组，那么跟随下面表中设定：

| order | no copy |                    copy=True                     |
| :---: | :-----: | :----------------------------------------------: |
|  'K'  |    \    |      保留F & C order，否则保留最相似的order      |
|  'A'  |    \    | 如果输入是F不是C，那么是F order。否则是C order。 |
|  'C'  | C oder  |                     C order                      |
|  'F'  | F oder  |                     F order                      |

当copy=False并且因为其他原因被复制，那么它与copy=True类似。'A'是一个例外，如果order='A'，而对象既不是'C'也不是'F'的数组，并且由于dtype的改变而强制执行复制，那么order的结果不一定是预期的'C'。这可能是一个BUG。

`subok` : bool, optional(布尔值，可选填)

如果为真，那么子类将被传递。否则数组将强制为基类数组(默认)。

`ndmin` : int, optional(整型， 可选填)

指定返回数组的最小维数。可以根据需要预置数组形状。

Return :

`out` : ndarray(n维数组对象)

返回一个满足特定需要的数组对象。

##### 创建多种元素类型的列表

```python
import numpy as np
x = np.array([(1, 2), (3, 4)], dtype=[('a_type', str, 10), ('b_type', '<i4')])
# x = np.array([(1, 2), (3, 4)], dtype=[('a_type', '<U10'), ('b_type', '<i4')])  # 与上面一样意思
# dtype=[('a_type', str, 10), ('b_type', '<i4')]是自定义类型，定义每个维中数组的元组中有不同类型，这个是定义元组中的两个元素，一个是字符串长度为10的类型，另一个是int32类型
# ('a_type', str, 10)，类型定义中，第一个是类型名字，第二个才是类型

print(x)
# array([('1', 2), ('3', 4)], dtype=[('a_type', '<U10'), ('b_type', '<i4')])

print(x['a_type'])  # 选出a_type类型的元素以数组返回
# array(['1', '3'], dtype='<U10')
```

##### 由子类创建为数组

subok=False

```python
import numpy as np
x = np.array(np.mat('1, 2; 3, 4'))
print(x)  # 输出为数组
# array([[1, 2],
#        [3, 4]])
```

subok=True

```python
import numpy as np
x = np.array(np.mat('1, 2; 3, 4'), subok=True)
print(x)  # 输出为矩阵
# matrix([[1, 2],
#         [3, 4]])
```

#### `numpy.zeros`

功能：返回一个指定形状大小和元素类型，并且所有元素都为0的数组。

```python
numpy.zeros(shape, dtype=float, order='C')
```

Parameters :

`shape` : int or tuple of ints(整数或者整数元组)

新数组的大小。

`dtype` : data-type, optional(数据类型，可选填)

数组数据类型。默认是`numpy.float64`。

`order` : {'C', 'F'}, optional, default : 'C'(在集合中选择，可选填，默认值是'C')

以行优先('C')还是列优先('F')储存在内存。

Returns :

`out` : ndarray(n维数组对象)

返回给定形状大小、数据类型和储存顺序的0元素数组。

##### Examples

```python
import numpy as np
x = np.zeros(5)
print(x)
# array([0., 0., 0., 0., 0.])

y = np.zeros(5,dtype=int)
print(y)
# array([0, 0, 0, 0, 0])

m = np.zeros((2, 1))
print(m)
# array([[0.],
#        [0.]])

n = np.zeros(2,dtype=[('a_type', 'i4'), ('b_type', 'i4')])
print(n)
# array([(0, 0), (0, 0)], dtype=[('a_type', '<i4'), ('b_type', '<i4')])
```

#### `numpy.zeros_like`

功能：返回一个与给定数组相同形状大小和元素类型的0元素数组。

```python
numpy.zeros_like(a, dtype=None, order='K', subok=True)
```

Parameters :

`a` : array_like(类数组)

a的形状大小和元素类型定义了返回数组的这些类型。

`dtype` : data-type, optional(数据类型，可选填)

重定义数据类型。

`order` : {'C', 'F', 'A', or 'K'}, optional(在集合选择，可选填)

重新定义内存布局(layout)。

`subok` : bool, optional(布尔值，可选填)

如果为真，新数组使用a数组类。否则新数组为基类。默认是真。

Return :

`out` : ndarray(n维数组对象)

返回一个与给定数组相同形状大小和元素类型的0元素数组。

##### Examples

```python
import numpy as np

x = np.arange(6)
print(x)
# array([0, 1, 2, 3, 4, 5])
x = x.reshape(2, 3)
print(x)
# array([[0, 1, 2],
#        [3, 4, 5]])
print(np.zeros_like(x))
# array([[0, 0, 0],
#        [0, 0, 0]])

y = np.arange(3, dtype=float)
print(y)
# array([0., 1., 2.])
print(np.zeros_like(y))
# array([0., 0., 0.])
```

#### `numpy.ones`

功能：返回一个给定形状大小和元素类型的全是1元素的数组。

```python
numpy.ones(shape, dtype=None, order='C')
```

Parameter :

`shape` : int or tuple of int(整数或者整数元组)

新数组形状大小。

`dtype` : data-type, optional(数据类型，可选填)

定义数组元素类型。默认是`numpy.float64`。

`order` : {'C', 'F'}, optional, default : 'C'(在集合中选择，可选填，默认值为'C')

定义多维数组在内存是以行优先('C')还是列优先('F')储存。

Return :

`out` : ndarray(n维数组对象)

返回一个给定形状大、元素类型和储存顺序的全是1元素的数组。

##### Examples

```python
import numpy as np

x = np.ones(5)
print(x)
# array([1., 1., 1., 1., 1.])

y = np.ones(5, dtype=int)
print(y)
# array([1, 1, 1, 1, 1])

m = np.ones((2, 1))
print(m)
# array([[1.],
#        [1.]])
```

#### `numpy.ones_like`

功能：返回一个与给定数组形状大小和元素类型相同的全是1元素的数组。

```python
numpy.ones_like(a, dtype=None, order='K', subok=True)
```

Parameters :

`a` : array_like(类数组)

a数组的形状大小和元素类型定义了返回数组的这些属性。

`dtype` : data-type, optional(数据类型，可选填)

可重新定义返回的数组元素的类型。

`order` : {'C', 'F', 'A', or 'K'}, optional(在集合中选择，可选填)

定义内存布局(layout)。

`subok` : bool, optional(布尔值，可选填)

如果为真，新数组使用a数组类。否则新数组为基类。默认是真。

Returns :

`out` : ndarray(n维数组对象)

返回一个与给定数组形状大小和元素类型相同的全是1元素的数组。

##### Examples

```python
import numpy as np

x = np.arange(6)
x = x.reshape(2, 3)
print(x)
# array([[0, 1, 2],
#        [3, 4, 5]])
print(np.ones_like(x))
# array([[1, 1, 1],
#        [1, 1, 1]])

y = np.arange(6, dtype=float)
print(x)
# array([0., 1., 2., 3., 4., 5.])
print(np.ones_like(y))
# array([1., 1., 1., 1., 1., 1.])
```

#### `numpy.empty`

功能：返回一个指定形状大小和元素类型，并且没有初始输入元素的数组。

```python
numpy.empty(shape, dtype=flaot, order='C')
```

Parameters :

`shape` : int or tuple of int(整数或者整数元组)

空数组大小。

`dtype` : data-type, optional(数据类型，可选填)

定义数组数据类型。

`order` : {'C', 'F'}, optional, default : 'C'(在集合中选择，可选填，默认值是'C')

定义多维数组在内存中是行优先('C')还是列优先('F')储存。

Returns :

`out` : ndarray(n维数组对象)

返回一个给定形状大小、元素类型和储存顺序的未初始化数组。数组初始化是`None`。

##### Examples

```python
import numpy as np

x = np.empty((2, 2))  # 随机生产数组
print(x)
# array([[5.e-324, 5.e-324],
#        [0.e+000, 0.e+000]])

y = np.empty((2, 2), dtype=int)  # 随机生产数组
print(y)
# array([[1, 0],
#        [1, 0]])
```

#### `numpy.empty_like`

功能：返回一个与给定数组形状大小和元素类型相同的空数组。

```python
numpy.empty_like(prototype, dtype=None, order='K', subok=True)
```

Parameters :

`prototype` : array_like(类数组)

prototype的形状大小和数据类型定义了返回数组的这些属性。

`dtype` : data-type, optional(数据类型，可选填)

可重新定义数组数据类型。

`order` : {'C', 'F', 'A', or 'K'}, optional(在集合中选择，可选填)

定义内存布局。

`subok` : bool, optional(布尔值，可选填)

如果为真，新数组使用prototype数组类。否则新数组为基类。默认是真。

Returns :

`out` : ndarray(n维数组对象)

返回一个与prototype形状大小和数据类型相同的未初始化的数组。

##### Examples

```python
import numpy as np

a = ([1, 2, 3], [4, 5, 6])
x = np.empty_like(a)
print(a)
# array([[-2042526868,       32767,  1949568416],
#        [        591,           0, -2147483648]])
```

#### `numpy.arange`

功能：返回给定区间的均匀分布的值。值是通过半开区间产生[start, stop)(换句话说，区间包括开始，但是不包括结尾。)

对于整数参数，这个函数等效为python的内置函数，但是它返回的是一个`ndarray`而不是`list`。

也可以使用非整数步数，如：0.1。但是结果的数组个数是不好看出。

```python
numpy.array([start,]stop,[step,]dtype=None)
```

Parameters :

`start` : number, optional(数字，可选填)

区间的开始。区间包括开始值。默认的开始值是0。

`stop` : number(数字)

区间的结束。区间不包括结束值。除了步数不是整数和浮点数精度会影响输出长度。

`step` : number, optional(数字，可选填)

相邻两个值的间隔。默认步数是1。如果步数作为位置参数传入，那么区间开始值也必须传入。

`dtype` : dtype(元素类型)

输出数组的类型。如果元素类型未给，那么通过其他传入的参数确定元素类型。

Returns :

`arange` : ndarray(n维数组)

均匀分布的值组成的数组。

对于浮点参数，数组的长度是ceil( ( start - stop ) / step )。因为浮点溢出(overflow)，数组的最后的值可能比结束值大。

###### Examples

```python
import numpy as np

x = np.arange(3)
print(x)
# array([0, 1, 2])

y = np.arange(3.0)
print(y)
# array([0., 1., 2.])

m = np.arange(3, 7)
print(m)
# array([3, 4, 5, 6])

n = np.arange(3, 7, 2)
print(n)
# array([3, 5])
```

#### `numpy.linspace`

功能：在给定区间下，返回均匀间隔的数。

在给定区间和`num`，返回`num`个区间中均匀间隔的数。

区间结束点可以选择进行排除。

支持非标量的开始和结束。

```python
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
```

Parameters :

`start` : array_like(类数组)

序列的开始值。

`stop` : array_like(类数组)

除非结束点设置为False，否则是序列的结束值。在是结束点是False的情况，序列由除`num + 1`之外的样本组成，以至于结束值被排除。注意：当结束点为False时，步数大小改变。

`num` : int, optional(整数，可选填)

产生的样本大小。默认是50。一定要是非负整数。

`endpoint` : bool, optional(布尔值，可选填)

如果为真，结束值为样本。否则，结束值不包括。默认是True。

`retstep` : bool, optional(布尔值，可选填)

如果为真，返回(样本，步数)。

`dtype` : dtype, optional(数据类型，可选填)

输出数组的数据类型。如果数据类型为给出，那么根据其他给出的参数推断数组的数据类型。

`axis` : int, optional(整数，可选填)

轴数用于储存样本。只有开始或者结束是类数组时才相关。

Returns :

`sample` : ndarray(n维数组)

返回一个在闭区间或半开区间(依赖于`endpoint`(结束点)是真还是假)中的等间距样本。

`step` : float, optional(浮点数，可选输出)

只有当retstep为真时才输出。样本中相邻两数的间隔大小。

##### Examples

```python
import numpy as np

x = np.linspace(2.0, 3.0, num=5)
print(x)
# array([2.  , 2.25, 2.5 , 2.75, 3.  ])

y = np.linspace(2.0, 3.0, num=5, endpoint=False)
print(y)
# array([2. , 2.2, 2.4, 2.6, 2.8])

m = np.linspace(2.0, 3.0, num=5, retstep=True)
print(m)
# (array([2.  , 2.25, 2.5 , 2.75, 3.  ]), 0.25)
```

##### Graphical illustration

```python
import numpy as np
import matplotlib.pyplot as plt
N = 8
y = np.zeros(N)
x1 = np.linspace(0, 10, N,endpoint=True)
x2 = np.linspace(0, 10, N,endpoint=False)
plt.plot(x1, y, 'o')
plt.plot(x2, y + 0.5, '0')
plt.ylim([-0.5, 1])
plt.show()
```

![image-20200821210100040](C:%5CUsers%5C81283%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20200821210100040.png)

#### `numpy.random.rand`

功能：生成一个给定形状大小的任意值数组。

创建一个给定形状大小的数组并从区间[0, 1)中任意填入数组中。

```python
numpy.random.rand(d0, d1, ..., dn)
```

Parameters :

`d0, d1, ..., dn` : int, optional(整数，可选填个数)

Returns :

`out` : ndarray, shape(d0, d1, ..., dn)

输出[0,1)中任意值的数组。

```python
import numpy as np

x = np.random.rand(3, 2)
print(x)
# array([[0.95290356, 0.12842212],
#        [0.30931995, 0.88945833],
#        [0.02126127, 0.71993726]])
```

#### `numpy.random.randn`

功能：从标准状态分布(standard normal distribution)中返回一个样本。

```PYTHON
numpy.random.randn(d0, d1, ...,dn)
```

Parameters :

`d0, d1, ..., dn` : int, optional(整数，可选填)

数组形状大小。如果不给，则返回分布中的一个浮点数。

Returns :

`Z` : ndarray or float(n维数组对象或者浮点数)

注意：

对于任意的分布$N(\mu,\sigma^2)$：

```python
σ * np.random.randn(...) + μ
```

##### Examples

```python
import numpy as np

x = np.random.randn()
print(x)
# -0.4429660128658533
```

从$N(3,2.5^2)$中选择样本

```python
import numpy as np

y = 2.5 * np.random.randn(2, 4) + 3
print(y)
# array([[ 6.54012398,  3.7693524 ,  5.58930912,  0.82538004],
#        [ 2.48431295, -0.08422529,  3.17837704,  3.21625178]])
```

#### `numpy.fromfunction`

功能：通过在每个坐标上执行一个函数来构造数组。在数组(x, y, z)处有fn(x, y, z)的值。

```python
numpy.fromfunction(fuction, shape, **kwargs)
```

Parameters :

`function` : callable(可以调用)

函数可以通过n个参数调用，n是shape中参数的个数。每个参数代表沿特定轴变化的数组坐标。

`shape` : (N,) tuple of ints(整数元组)

输出数组的选择大小，也决定了通过函数形成的坐标数组的形状大小。

`dtype` : data-type, optional(数据类型，可选填)

通过函数形成的坐标数组的数据类型，默认是浮点型。

Returns :

`fromfunction` : any

调用函数的结果是直接返回。因此fromfunction的形状大小完全由函数决定。如果函数返回一个标量，那么fromfunction的形状大小不会匹配shape的参数。

注意：

除dtype以外的关键字参数将传给函数。

##### Examples

```python
import numpy as np

x = np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
print(x)
# array([[ True, False, False],
#        [False,  True, False],
#        [False, False,  True]])

y = np.fromfunction(lambda i, j: i + j, (3, 3),dtype=int)
print(y)
# array([[0, 1, 2],
#        [1, 2, 3],
#        [2, 3, 4]])
```

#### `numpy.fromfile`

功能：用文本或者二进制文件中数据创建一个数组。

一种读取已知数据类型的二进制数据和解析简单格式化文本的高效方法。使用`tofile`方法写入的数据，可以使用该方法来读取。

```python
numpy.fromfile(file, dtype=float, count=-1, sep='')
```

Parameter :

`file` : file or str(文件或者文件名字符串)

打开文件对象或者文件名。

`dtype` : data-type(数据类型)

返回数组的数据类型。对于二进制文件，它通常用于确定文件中各项的大小和字节顺序。

`count` : int(整数)

读取的项目个数。`-1`表示所有项目。

`sep` : str(字符串)

如果是文本文件，那么它是项目之间的分隔符。空('')的分隔符意味着要将文件看作二进制文件。空格(' ')匹配0或者更多的空白字符。仅由空格组成的分割符最少匹配一个空格。

##### Examples

创建一个n维数组对象

```python
import numpy as np

dt = np.dtype([('time', [('min', int), ('sec', int)]),
               ('temp', float)])
x = np.zeros(1, dtype=dt)
print(x)
# array([((0, 0), 0.)],
#       dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])

x['time']['min'] = 10
x['temp'] = 98.25
print(x)
# array([((10, 0), 98.25)],
#       dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])
```

将原始数据保存到磁盘

```python
import os
fname = 'binary'
x.tofile(fname)
```

从磁盘中读取原始数据

```python
y = np.fromfile(fname, dtype=dt)
print(y)
# array([((10, 0), 98.25)],
#       dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])
```

推荐储存和加载数据的方式(以.npy文件储存)

```python
np.save(fname, x)
np.load(fanme + '.npy')
# array([((10, 0), 98.25)],
#       dtype=[('time', [('min', '<i4'), ('sec', '<i4')]), ('temp', '<f8')])
```

### 打印输出数组

当你打印输出数组时，数组是以简单的嵌套列表来展现。

太长的数组将会省略中心部分。如果需要强行打印全部，那么通过改变打印设置来完成。

```python
# 修改打印输出设置，强制打印全部
np.set_printoptions(threshold=np.nan)
```

### 基础操作

在数组中元素逐一进行算数操作。

```python
a = np.array([20, 30, 40, 50])
b = np.arange(4)
c = a - b
print(c)
# array([20, 29, 38, 47])

print(b**2)
# array([0, 1, 4, 9], dtype=int32)

print(10 * np.sin(a))
# array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])

print(a < 35)
# array([ True,  True, False, False])
```

与矩阵的乘法不同，数组的乘法是逐个元素相乘。

```python
A = np.array([[1, 1],
             [0,1]])
B = np.array([[2, 0],
             [3, 4]])
A * B  # 对应元素相乘
# array([[2, 0],
#        [0, 4]])

A @ B  # 矩阵相乘
# array([[5, 4],
#        [3, 4]])

A.dot(B)  # 用dot方法进行矩阵相乘
# array([[5, 4],
#        [3, 4]])
```

`+=`和`*=`操作。

```python
a = np.ones((2, 3), dtype=int)
b = np.random.random((2, 3))
a *= 3
print(a)
# array([[3, 3, 3],
#        [3, 3, 3]])

b += a
print(b)
# array([[3.23633511, 3.32116339, 3.42040182],
#        [3.17732031, 3.42658345, 3.4033062 ]])

a += b  # b不能自动转换为整数类型
# TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
```

对不同类型数组进行运算操作，输出数组的类型会更倾向于运算数组的公共类型。

复数公式：$z=r(cos\theta+isin\theta)=re^{i\theta}$

```python
a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3)
print(b.dtype.name)
# 'float64'

c = a + b
print(c)
# array([1.        , 2.57079633, 4.14159265])

print(c.dtype.name)
# 'float64'

d = np.exp(c * 1j)
print(d)
# array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
#        -0.54030231-0.84147098j])

print(d.dtype.name)
# 'complex128'
```

许多数组的一元(unary)操作都是作为`ndarray`类的方法进行的。

```python
a = np.random.random((2, 3))
print(a)
# array([[0.22211365, 0.91025021, 0.84639606],
#        [0.16552852, 0.83354256, 0.62299227]])

a.sum()
# 3.6008232613521853

a.min()
# 0.16552852183741995

a.max()
# 0.9102502088496457
```

可以通过指定轴来进行操作。

```python
b = np.arange(12).reshape(3, 4)
print(b)
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]])

b.sum(axis=0)
# array([12, 15, 18, 21])

b.min(axis=1)
# array([0, 4, 8])

b.cumsum(axis=1)  # 在1轴上，逐步累加
# array([[ 0,  1,  3,  6],  # 举例第一行[0, 0+1, 0+1+2, 0+1+2+3]
#        [ 4,  9, 15, 22],
#        [ 8, 17, 27, 38]], dtype=int32)
```

### 通用函数

`NumPy`提供了熟悉的数学函数(`sin`、`cos`、`exp`)。这些函数是作为通用函数被调用。在`NumPy`中执行函数是逐个元素进行运算操作。

```python
B = np.arange(3)
print(B)
# array([0, 1, 2])

print(np.exp(B))  # 指数运算
# array([1.        , 2.71828183, 7.3890561 ])

print(np.sqrt(B))  # 算数平方根运算
# array([0.        , 1.        , 1.41421356])

C = np.array([2., -1., 4.])
print(np.add(B, C))  # 加法运算
# array([2., 0., 6.])
```

### 索引、切片和迭代

一维数组可以被索引、切片和迭代，它更像列表和其他python的序列。

```python
a = np.arange(10)**3
print(a)
# [  0   1   8  27  64 125 216 343 512 729]

print(a[2])
# 8

print(a[2:5])  # 半开区间，包括开始，不包括结束
# [ 8 27 64]

a[:6:2] = -1000
print(a)
# [-1000     1 -1000    27 -1000   125   216   343   512   729]

print(a[::-1])  # 倒序索引输出
# [  729   512   343   216   125 -1000    27 -1000     1 -1000]

for i in a:
    print(i**(1/3))
    print('=============')
    print(round(i**(1/3)), '\n')
nan
=============
nan 

1.0
=============
1.0 

nan
=============
nan 

3.0
=============
3.0 

nan
=============
nan 

5.0
=============
5.0 

5.999999999999999
=============
6.0 

6.999999999999999
=============
7.0 

7.999999999999999
=============
8.0 

8.999999999999998
=============
9.0 
```

多维数组可在每个轴上有一个索引。这些索引通过逗号分隔的元组给出。

```python
def f(x, y):
    return 10*x + y

b = np.fromfunction(f, (5, 4), dtype=int)
print(b)
# [[ 0  1  2  3]
#  [10 11 12 13]
#  [20 21 22 23]
#  [30 31 32 33]
#  [40 41 42 43]]

print(b[2, 3])
# 23

print(b[0:5, 1])
# [ 1 11 21 31 41]

print(b[:, 1])
# [ 1 11 21 31 41]

print(b[1:3, :])
# [[10 11 12 13]
#  [20 21 22 23]]
```

当索引值数小于轴数时，缺少的索引默认是全切片(:)。

```python
print(b[-1])
# [40 41 42 43]
```

表达式b[i]中的i后面可以用:来对保留轴进行占位。也可以使用点来占位，如b[i, ...]。

dots(...)代表了许多冒号(colons)作为全索引。举个例子假设x数组轴是5。

- x[1, 2, ...]等价于x[1, 2, :, :, :]
- x[..., 3]等价于x[:, :, :, :, 3]
- x[4, ..., 5, :]等价于x[4, :, :, 5, :]

```python
c = np.array([[[  0,   1,   2],
               [ 10,  12,  13]],
              [[100, 101, 102],
               [110, 112, 113]]])

print(c.shape)
# (2, 2, 3)

print(c[-1, ...])  # 等价于c[1, :, :]或者c[1]
# [[100 101 102]
#  [110 112 113]]

print(c[..., 2])  # 等价于c[:, :, 2]
# [[  2  13]
#  [102 113]]
```

对多维数组第一个轴进行迭代。

```python
for row in b:
    print(row)

# [0 1 2 3]
# [10 11 12 13]
# [20 21 22 23]
# [30 31 32 33]
# [40 41 42 43]
```

然而，如果你想要迭代数组每一个元素，你可以使用`flat`属性的迭代器对数组所有元素进行迭代。

```python
for element in b.flat:
    print(element)
    
# 0
# 1
# 2
# 3
# 10
# 11
# 12
# 13
# 20
# 21
# 22
# 23
# 30
# 31
# 32
# 33
# 40
# 41
# 42
# 43
```

### 数组形状大小操作

#### 改变数组形状大小

通过每个轴上元素的个数来确定数组的形状。

```python
a = np.floor(10*np.random.random((3, 4)))
print(a)
# [[9. 9. 8. 1.]
#  [5. 8. 8. 3.]
#  [9. 4. 0. 2.]]
print(a.shape)
# (3, 4)
```

通过各种命令可以改变一个数组的形状大小。注意，下面三个命令都是返回一个修改的数组，不是改变原数组形状大小。

```python
print(a.ravel())  # 返回展平的一维数组
# [9. 9. 8. 1. 5. 8. 8. 3. 9. 4. 0. 2.]
print(a.reshape(6, 2))  # 返回一个定义的形状
# [[9. 9.]
#  [8. 1.]
#  [5. 8.]
#  [8. 3.]
#  [9. 4.]
#  [0. 2.]]
print(a.T)  # 返回转置(transposed)数组
# [[9. 5. 9.]
#  [9. 8. 4.]
#  [8. 8. 0.]
#  [1. 3. 2.]]
print(a.T.shape)
# (4, 3)
print(a.shape)
# (3, 4)
```

由ravel()产生的数组元素顺序一般是C顺序(行主要)，即右边的索引更改最快，所以元素a[0, 0]之后是a[0, 1]。如果数组是被重新定义形状大小，那么数组还是C顺序。`NumPy`创建的数组一般是这个顺序储存，所以通常ravel()不需要复制其他参数，但是如果数组是通过切片得到的或者通过不同寻常的操作得到的，那么可能需要将其复制。函数ravel()和reshape()也可以改变order参数来改变储存顺序，如果是F顺序，那么左边将是索引的最快。

numpy.resize()函数改变数组本身形状大小。

```python
print(a)
# [[9. 9. 8. 1.]
#  [5. 8. 8. 3.]
#  [9. 4. 0. 2.]]
a.resize((2, 6))
print(a)
# [[9. 9. 8. 1. 5. 8.]
#  [8. 3. 9. 4. 0. 2.]]
```

在reshape函数中，如果一个维数传入-1，那么其维数是自动计算的。

```python
print(a)
# [[9. 9. 8. 1.]
#  [5. 8. 8. 3.]
#  [9. 4. 0. 2.]]
print(a.reshape(2, -1))  # 第一个轴长度确定，第二轴传入-1，那么第二轴将会自动计算长度
# [[9. 9. 8. 1. 5. 8.]
#  [8. 3. 9. 4. 0. 2.]]
```

#### 不同数组的堆叠

几个数组可以沿着不同轴进行堆叠。

```python
a = np.floor(10 * np.random.random((2, 2)))
print(a)
# [[8. 5.]
#  [4. 9.]]
b = np.floor(10 * np.random.random((2, 2)))
print(b)
# [[5. 0.]
#  [2. 1.]]
print(np.vstack((a, b)))  # 垂直堆叠
# [[8. 5.]
#  [4. 9.]
#  [5. 0.]
#  [2. 1.]]
print(np.hstack((a, b)))  # 水平堆叠
# [[8. 5. 5. 0.]
#  [4. 9. 2. 1.]]
```

函数`column_stack`可将一维数组堆叠为二维数组。对于二维数组的堆叠和函数`hstack`一样。

```python
from numpy import newaxis
print(np.column_stack(a, b))
# [[8. 5. 5. 0.]
#  [4. 9. 2. 1.]]

a = np.array([4., 2.])
b = np.array([3., 8.])

print(np.column_stack(a, b))  # 一维数组堆叠为二维数组
# [[4. 3.]
#  [2. 8.]]

print(np.hstack(a, b))  # 只有二维堆叠才与column_stack相同
# [4. 2. 3. 8.]

print(a[:, newaxis])  # 允许由一个二维列向量
# [[4.]
#  [2.]]

print(np.column_stack(a[:, newaxis], b[:, newaxis]))
# [[4. 3.]
#  [2. 8.]]

print(np.hstack(a[:, newaxis], b[:, newaxis]))  # 二维堆叠结果与column_stack相同
# [[4. 3.]
#  [2. 8.]]
```

在另一方面，对于任意输入数组，函数`row_stack`与函数`vstack`返回效果相同。对于二维以上的数组，函数`hstack`沿着第二轴堆叠，函数`vstack`沿着第一轴堆叠，并且函数`concatenate`提供了一个可选参数，它给出了级联的轴号。

注意：

在复杂情况下，函数`r_`和`c_`对于某一轴上堆叠数字很有用，这两个函数允许使用范围迭代(':')。

```python
print(np.r_[1:4, 0, 4])
# array([1, 2, 3, 0, 4])
```

#### 分割一个数组变成几个更小的数组

使用函数`hsplit`，你可以沿着水平轴上分割一个数组，另外通过指定返回的相同形状的数组数量，或者指定需要从哪列开始分割。

```python
a = np.floor(10 * np.random.random((2, 12)))
print(a)
# [[5. 7. 7. 0. 7. 1. 0. 7. 8. 3. 0. 5.]
#  [6. 0. 3. 4. 1. 4. 2. 2. 7. 8. 4. 6.]]

print(np.hsplit(a, 3))
# [array([[5., 7., 7., 0.],
#        [6., 0., 3., 4.]]), array([[7., 1., 0., 7.],
#        [1., 4., 2., 2.]]), array([[8., 3., 0., 5.],
#        [7., 8., 4., 6.]])]

print(np.hsplit(a, (3, 4)))
# [array([[5., 7., 7.],
#        [6., 0., 3.]]), array([[0.],
#        [4.]]), array([[7., 1., 0., 7., 8., 3., 0., 5.],
#        [1., 4., 2., 2., 7., 8., 4., 6.]])]

print(np.hsplit(a, (3,)))
# [array([[5., 7., 7.],
#        [6., 0., 3.]]), array([[0., 7., 1., 0., 7., 8., 3., 0., 5.],
#        [4., 1., 4., 2., 2., 7., 8., 4., 6.]])]
```

函数`vsplit`沿着垂直轴进行分割，并且函数`arrary_split`允许指定沿着哪一个轴进行分割

### 复制和视图

当我们操作一个数组时，它们的数据有时会复制到一个新数组，有时不会复制。

#### 不完全复制

简单任务时，不复制数组或它们的数据。

```python
a = np.arange(12)
b = a  # 不会创建新数组
print(b is a)  # a,b两变量名指向同一对象，对象储存地址并未发生改变
# True
b.shape = (3, 4)  # 改变b的形状就是改变a的形状
print(a.shape)
# (3, 4)
```

python将可变对象作为引用传递，所以函数调用不会复制对象。

```python
def f(x):
    print(id(x))

print(id(a)) # 2667002014656
f(a)  # 2667002014656
```

#### 视图(浅复制)

不同的数组对象可以分享相同数据。`view`方法创建一个查看相同数据的新数组。

```python
c = a.view()
print(c is a)
# False
print(c.base is a)  # c是a数据的一个视图
# True
print(c.flags.owndata)  # 数据不是c自己的
# False
c.shape = (2, 6)  # 改变c的形状，a的形状不改变，c只是查看了a的数据
print(a.shape)
# (3, 4)
c[0, 4] = 1234  # 改变c的数据实际就是改变a的数据
print(a)
# [[   0    1    2    3]
#  [1234    5    6    7]
#  [   8    9   10   11]]
```

切片一个数组将返回该数组的一个视图。

```python
s = a[:, 1:3]
print(s)  # s是a[:, 1:3]的视图
# [[ 1  2]
#  [ 5  6]
#  [ 9 10]]
s[:] = 10  # s[:]是s的视图，改变s[:]最终改变的是a的数据
print(a)
# [[   0   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]
```

#### 深复制

`copy`方法是完全复制数组及它的数据。

```python
d = a.copy()  # 创建带有新数据的新数组
print(d is a)
# False
print(d.base is a)  # d中的数据不与a分享
# False
d[0, 0] = 9999
print(a)
# [[   0   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]
print(d)
# [[9999   10   10    3]
#  [1234   10   10    7]
#  [   8   10   10   11]]
```



