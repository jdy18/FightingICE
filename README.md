# fightingice 

## Download sample dataset 

` wget https://huggingface.co/datasets/iamjinchen/FightingICE/resolve/main/Sample_data.pth.zip  `             
` unzip Sample_data.pth.zip `               
`mv Sample_data.pth /Dataset`        

## Download full dataset

- 6G data collected by pretrained policy: 

  链接：https://pan.baidu.com/s/1GOLz2K8D-cHGnt0Os7fdlg?pwd=ufa8 
  提取码：ufa8 
- 6G data collected by random policy:

  链接：https://pan.baidu.com/s/1YiNqLo25tvATJRBOWKZT-Q?pwd=396u 
  提取码：396u 

## Prepare

you need to install tianshou package first. https://github.com/thu-ml/tianshou

## Run TD3+BC
- prepare
    - you need to install tianshou package first. https://github.com/thu-ml/tianshou
    -  `pip install tianshou`
- run the code
    - ` python cogftg_td3_bc.py `

## Run CQL
- train model
  - ` python cogftg_cql.py
- test model
  - save the parameter in /tianshou_cql/test/model
  - ` python test.py
 ## Run BCQ

### there are 2 versions of BCQ. 
- continuous version
    - which i change action to a 40-dim one-hot tensor. 
    - in training phrase, the loss goes higher and higher :/
    - To RUN:  
    `cd /BCQ/continuous`      
    `python cogftg_bcq_continuous.py`
- discrete version
    - which should work. 
    - TO RUN:    
`cd /BCQ/discrete   `   
`python cogftg_bcq_discrete.py`

## Useful links
[audio-only RL model : BlindAI](https://github.com/TeamFightingICE/BlindAI) \
[RL platform easy to use : tianshou](https://github.com/thu-ml/tianshou)
