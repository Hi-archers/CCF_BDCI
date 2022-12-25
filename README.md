## 预训练基准模型知识量度量复现代码说明



#### 1. 复现镜像和代码链接为

<https://pan.baidu.com/s/1n3YoBrrzJ9il7yfyOhK8DQ?pwd=i9dv>
 

#### 2. 模型checkpoint文件链接为: 
(1)以下链接中共含有一个BERT分类模型参数，三个T5模型压缩包其中共有7个T5模型。
https://pan.baidu.com/s/1OnkepRTBc8suyOY4uXSqJg?pwd=hghs

**note: 模型下载完成后, 需要放到data/user_data/model文件夹里**


#### 3. docker中的运行命令为:(详情见image文件夹中的README)
```shell
nvidia-docker run --entrypoint=/bin/sh -v </data_in_host>:/data ccir:final /data/run.sh
```

**note**: 

(1)**run.sh在data路径下**(直接执行下面命令即可)

(2)如果运行sh脚本有报错,请联系我。


#### 4. 复现代码分为5个模块, 分别如下:

#### 4.1 数据预处理
```shell
log打印出 "Finish data preprocessing..." 为数据预处理结束标志
```

#### 4.2 使用bert进行inference

```shell
log打印出 "Finish bert inference..." 为使用bert模型inference结束
```


#### 4.3 使用T5模型进行inference
```shell
log打印出 "Finish T5 inference..." 为T5模型inference结束
```

#### 4.4 根据生成的答案合并得到最终结果
```shell
"Finish generate result..." 为生成results结束标志
```

