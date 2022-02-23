# Ubuntu Library

- [Eigen](d#eigen)

---
## Eigen

*Eigen consists only of header files, hence there is nothing to compile before you can use it. Moreover, these header files do not depend on your platform, they are the same for everybody.*

### 安装方法1 终端apt命令安装   
```bash
sudo apt-get install libeigen3-dev
```
`Eigen` 只包含头文件，因此它不需要实现编译（只需要使用#include），指定好Eigen的头文件路径，编译项目即可。    

Eigen头文件的默认安装位置是：`/usr/include/eigen3` .    

这样在 `include` 的时候需要 `#include "eigen/Eigen/Dense"` 比较麻烦，将 `/usr/include/eigen3` 文件夹下的 `Eigen` 软连接到 `/usr/include/` 下就可以直接用 `#include "Eigen/Dense"`:    
```bash
sudo ln -s /usr/include/eigen3/Eigen  /usr/include/Eigen
```

### 安装方法2 源码安装
安装包[下载网址](http://eigen.tuxfamily.org/index.php?title=Main_Page)可以下载任意版本对应的文件，本例下载了Eigen 3.3.7版本的`tar.gz`格式压缩文件。

文件名：*eigen-eigen-323c052e1731.tar.gz*

运行命令：
```bash
sudo tar -xzvf eigen-eigen-323c052e1731.tar.gz -C /usr/local/include
# sudo tar -xzvf eigen-eigen-323c052e1731.tar.gz -C /usr/include
```
**注意**： `/usr/local/include` 也可以换成 `/usr/include` ，该命令 `tar -xzvf file.tar.gz` 表示解压 `tar.gz` 文件， `-C` 表示建立解压缩档案.    

这条指令将 `eigen-eigen-323c052e1731.tar.gz` 文件解压到了 `/usr/local/include` 目录下，在 `/usr/local/include` 目录下得到文件夹`eigen-eigen-323c052e1731`    

运行命令:    
```bash
sudo mv /usr/local/include/eigen-eigen-323c052e1731 /usr/local/include/eigen3
```
这条指令将 `eigen-eigen-323c052e1731` 文件 更名为 `eigen3`

运行命令：    
```bash
sudo cp -r /usr/local/include/eigen3/Eigen /usr/local/include
#sudo cp -r /usr/include/eigen3/Eigen /usr/include
```
**注意**：参考cp指令 `cp -r /usr/men /usr/zh` 将目录 `/usr/men` 下的所有文件及其子目录复制到目录 `/usr/zh` 中

上个命令的说明：
因为 `eigen3` 被默认安装到了 `usr/local/include` 里了（或者是 `usr/include` 里），在很多程序中`include`时经常使用 `#include <Eigen/Dense>` 而不是使用 `#include <eigen3/Eigen/Dense>` 所以要做下处理，否则一些程序在编译时会因找不到 `Eigen/Dense` 而报错。上面指令将` usr/local/include/eigen3` 文件夹中的 `Eigen` 文件递归地复制到上一层文件夹（直接放到 `/usr/local/include` 中，否则系统无法默认搜索到 -> 此时只能在CMakeLists.txt用include_libraries(绝对路径了)）

### 验证安装
```bash
cd ~/Downloads
mkdir eigen_tutorials
cd eigen_tutorials
gedit e1.cpp
```
添加以下内容:    
```cpp
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
int main()
{
    MatrixXd m = MatrixXd::Random(3,3);
    //定义3x3的随机矩阵m，各元素取值范围是[-1，1]
    m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
    //"MatrixXd::Constant(3,3,1.2)"定义各元素为1.2的3-by-3常数矩阵，m的取值范围变为[10，110]
    cout << "m =" << endl << m << endl;
    VectorXd v(3); //定义维度为3的列向量v
    v << 1, 2, 3;  //赋值于v
    cout << "m * v =" << endl << m * v << endl;
}
```
编译运行:    
```bash
g++ e1.cpp
./a.out
```
运行结果:    
```bash
m =
    94   89.8 43.5
    49.4 101  86.8
    88.3 29.8 37.8
m * v =
        404
        512
        261
```


### Eigen库的模块及其头文件
为了应对不同的需求，Eigen库被分为多个功能模块，每个模块都有自己相对应的头文件，以供调用。 其中，Dense模块整合了绝大部分的模块，而Eigen模块更是整合了所有模块（也就是整个Eigen库）。

![eigen](../img/eigen.svg)     

**参考资料**
> [Eigen是个什么鬼？](https://zhuanlan.zhihu.com/p/36706885)    

---
