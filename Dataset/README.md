# README
  
- 数据包括50局比赛共150条轨迹，共50万个状态、动作及其奖励，数据存储格式是张量Orderdict

- 数据读进来有五个量，对于一个轨迹s~1~-s~n~

  - observations：当前状态，s~1~-s~n-1~
  - next_observations：下一个时刻状态，s~2~-s~n~
  - actions：动作，a~1~-a~n-1~
  - rewards：奖励，r~1~-r~n-1~
  - terminals：终止与否，对应next_observations，只有最后一个状态s~n~为1，其他为0

  数据集里的量是所有轨迹拼接起来的结果。
