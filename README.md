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
  - ` python cogftg_cql.py `
- test model
  - save the model in /tianshou_cql/test/model
  - ` python test.py `
 ## Run BCQ

### there are 2 versions of BCQ. 
- continuous version
    - which i change action to a 40-dim one-hot tensor. 
    - in training phrase, the loss goes higher and higher :/
    - it keeps output same action in test phrase. To my understanding, \
    this is because it learned a continuous action tensor, which cannot simply be `argmax()`.
    
    - To RUN:  
       - change directory in `utils_bcq.py` to your buffer file.
       - change epoch, step_nums in `cogftg_bcq_continuous.py`\
        `cd /BCQ/continuous`      \
        `python cogftg_bcq_continuous.py`
       - the log, weights will be output under `log` directory.
    
    - To Test
       - change directory parameters in `test.py` to your `policy.pth` file.\
       i have pretrained weight files in `/results`, to which you can refer
       - the results is set to output in `/results/bcq_vs_MctsAi23i.txt`


- discrete version
    - which should work. 
     
    - TO RUN:    
       - change directory in `utils_discrete.py` to your buffer file.
       - change epoch, step_nums in `cogftg_bcq_discrete.py`\
          `cd /BCQ/discrete`      \
          `python cogftg_bcq_continuous.py`
       - the log, weights will be output under `log` directory.
    - To Test
       - change directory parameters in `test.py` to your `policy.pth` file.\
       i have pretrained weight files in `/results`, to which you can refer
       - the results is set to output in `/results/bcq_discrete_vs_MctsAi23i.txt`
       
       

 ## Run CRR

-  prepare:
  -  install tianshou package as mentioned above
  - `cd CRR`

- To train:  

  - change `path` in `utils.py` to path of your dataset.
  - change epoch, step_nums in `cogftg_crr.py`
  - run `python cogftg_crr.py`
  - the log, weights will be output under `log` directory.

- To Test

  - get the pretrained actor model

  - set the `actor_path,save_path,actor_name` in `'test.py'`. 

  - For example : `python test.py --actor_path ./model/actor.pt --actor_name CRR --save_path  ./results/ppopretrain_vs_MctsAi23i.txt`

  - path` in `utils.py` to path of your dataset.

    

    

    the log, weights will be output under `log` directory.

    
 ## Run PPO
- RNN version

- MLP version


## Useful links
[audio-only RL model : BlindAI](https://github.com/TeamFightingICE/BlindAI) \
[RL platform easy to use : tianshou](https://github.com/thu-ml/tianshou)
