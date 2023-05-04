# README

- 数据下载链接：

  随机动作：
  
  链接：https://pan.baidu.com/s/1f5mbX2oV0WLt84PvQpLydw 
  提取码：xs9u 
  
  预训练模型：
  
  链接：https://pan.baidu.com/s/1rgmI22KOKCgexsgCRVWdIQ?pwd=9jg0 
  提取码：9jg0 
  
- 数据大约有20G，包括300局比赛共900条轨迹，共三百多万个状态、动作及其奖励，数据存储格式是张量Orderdict

- 音频数据还未收集，音频数据的收集格式还需要再进行讨论，如果直接收集原始数据估计数据量会接近100G

- 将数据加载进datasets.py需要大概两分多钟，以及需要20G左右的运行内存。

- datasets.py每次返回长度为32的状态、动作、奖励序列
