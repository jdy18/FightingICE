# fightingice 

## Download sample dataset 

` wget https://huggingface.co/datasets/iamjinchen/FightingICE/resolve/main/Sample_data.pth.zip  `             
` unzip Sample_data.pth.zip `               
`mv Sample_data.pth /Dataset`             

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
 
## Useful links
[audio-only RL model : BlindAI](https://github.com/TeamFightingICE/BlindAI) \
[RL platform easy to use : tianshou](https://github.com/thu-ml/tianshou)
