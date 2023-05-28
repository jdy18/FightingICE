# BCQ

|  feature_net | train dataset | train epochs | steps per epoch | test rounds | win_ratio | hp_diff_avg | Speed | RemainHP | Advantage |  Damage |
| ----|----|----|----|----|----|----|----|----|----|----|
| DQN | pretrain | 160 | 10000 | 150 | 0.689 | 74.044 | 0.0743 | 0.2456 | 0.0925 | 0.939 |
| MLP | pretrain | 200 | 5000 | 90 | 0.767 | 67.644 | 0.002 | 0.402 |0.0846| 0.7672 |
| MLP | random | 200 | 5000 | 90 | 0.555 | 28.355 | 0.0087 | 0.2785 | 0.0354 | 0.792|
| MLP | mix | 200 | 5000 | 90 | 0.566 | 26.87 | 0.0044 | 0.295 | 0.0336 | 0.772 |





## with DQN structure as feature_net:
`
with pretrain data, 160 epochs:   
test with 90 rounds:   
 win_ratio: 74.044, 
 hp_diff_avg 0.689   
 Speed:0.07431790123456786   
RemainHP:0.24566666666666667  
Advantage:0.09255555555555556  
Damage:0.9394444444444444  
`

## with MLP  as feature_net :
`
with pretrain data, 200 epochs: 
test with 90 rounds: 
 win_ratio: 0.7666666666666667 
 hp_diff_avg: 67.64444444444445 
 Speed: 0.0020493827160493827   
 RemainHP: 0.4019166666666667   
 Advantage: 0.08455555555555555   
 Damage: 0.7671944444444445 
 `
   
 `
 with random data, 200 epochs: 
 test with 90 rounds: 
 win_ratio: 0.5555555555555556  
 hp_diff_avg: 28.355555555555554  
 Speed: 0.008712962962962962  
 RemainHP: 0.2785  
 Advantage: 0.035444444444444445  
 Damage: 0.7923888888888888  
 `
 `
 with random data, 200 epochs: 
 test with 90 rounds: 
  win_ratio: 0.5666666666666667  
 hp_diff_avg: 26.877777777777776  
 Speed: 0.004435185185185184  
 RemainHP: 0.29491666666666666  
 Advantage: 0.03359722222222222  
 Damage: 0.7722777777777778  
 `


