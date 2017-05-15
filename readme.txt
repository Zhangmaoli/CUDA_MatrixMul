关于cuda的安装、配置和编译：

####安装####

Windows下安装：
在设备管理器中查看GPU是否支持CUDA。
在CUDA下载页面选择合适的系统平台，下载对应的开发包。
安装开发包，需要预先安装Visual Studio 2010 或者更高版本。
验证安装，打开命令提示框，输入命令nvcc – V。

Linux下安装：
使用lspci |grep nvidia –I 命令查看GPU型号。
在CUDA下载页面选择合适的系统平台，下载对应的开发包(*.run)。
安装：使用：
       chmod a+x cuda_7.0.28_linux.run sudo    
      ./cuda_7.0.28_linux.run。
设置环境变量：
      PATH=/usr/local/cuda/bin:$PATH export PATH
      source /etc/profile

####创建和调试####

Windows下创建及调试：
新建项目-CUDA 7.0 Runtime。
调试：使用Nsight 进行调试：
            Nsight->start CUDA debugging

Linux下创建及调试：
创建*.cu以及*.cuh文件，需包含<cuda_runtime.h>头文件。
调试：使用cuda-gdb进行调试：
              nvcc–g –G *.cu –o binary
nvcc为cuda程序编译器。
-g 表示可调试。
*.cu 为cuda源程序。
-o 生成可执行文件。

####编译####

编译：
Windows下可直接使用Windows Microsoft Visual Studio等集成开发环境。
Linux下编译：nvcc cuda.cu。



####代码####
kernel.cu:矩阵链乘实现
vec.cu:矩阵向量乘实现
