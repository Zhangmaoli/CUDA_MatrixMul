����cuda�İ�װ�����úͱ��룺

####��װ####

Windows�°�װ��
���豸�������в鿴GPU�Ƿ�֧��CUDA��
��CUDA����ҳ��ѡ����ʵ�ϵͳƽ̨�����ض�Ӧ�Ŀ�������
��װ����������ҪԤ�Ȱ�װVisual Studio 2010 ���߸��߰汾��
��֤��װ����������ʾ����������nvcc �C V��

Linux�°�װ��
ʹ��lspci |grep nvidia �CI ����鿴GPU�ͺš�
��CUDA����ҳ��ѡ����ʵ�ϵͳƽ̨�����ض�Ӧ�Ŀ�����(*.run)��
��װ��ʹ�ã�
       chmod a+x cuda_7.0.28_linux.run sudo    
      ./cuda_7.0.28_linux.run��
���û���������
      PATH=/usr/local/cuda/bin:$PATH export PATH
      source /etc/profile

####�����͵���####

Windows�´��������ԣ�
�½���Ŀ-CUDA 7.0 Runtime��
���ԣ�ʹ��Nsight ���е��ԣ�
            Nsight->start CUDA debugging

Linux�´��������ԣ�
����*.cu�Լ�*.cuh�ļ��������<cuda_runtime.h>ͷ�ļ���
���ԣ�ʹ��cuda-gdb���е��ԣ�
              nvcc�Cg �CG *.cu �Co binary
nvccΪcuda�����������
-g ��ʾ�ɵ��ԡ�
*.cu ΪcudaԴ����
-o ���ɿ�ִ���ļ���

####����####

���룺
Windows�¿�ֱ��ʹ��Windows Microsoft Visual Studio�ȼ��ɿ���������
Linux�±��룺nvcc cuda.cu��



####����####
kernel.cu:��������ʵ��
vec.cu:����������ʵ��
