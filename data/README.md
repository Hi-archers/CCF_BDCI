## 预训练基准模型知识量度量复现代码说明



#### 1. 复现镜像和代码链接为

<https://pan.baidu.com/s/1FvouIH6TfzLtjrQEzkwHcw>
 密码: fja8

#### 2. 模型checkpoint文件链接为: 
(1)Bert-checkpoint
<https://pan.baidu.com/s/1GJfY_1nqNc24XwakzYHHdA>
密码: da68

(1)T5-checkpoint-split_1 <https://pan.baidu.com/s/1ZoI_tWu5JVqTGKX1Wh3cBw> 密码: o6w7


(2)T5-checkpoint-split_2
<https://pan.baidu.com/s/1iD5mRz4Ca_u0LDq8BAEk6A> 
密码: dvux

(3)T5-checkpoint-split_3
<https://pan.baidu.com/s/1xoPxRVutVEdkc1MSNgze3w>
密码: k85x 

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

