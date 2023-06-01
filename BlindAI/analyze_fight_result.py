import argparse
import pandas as pd
import os

def calculate(path):
    files = [path + '\\' + f for f in os.listdir(path)]
    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None)
        dfs.append(df)
    data_all = pd.concat(dfs, axis=0, ignore_index=True)
    data_all.columns = ['Round', 'P1', 'P2', 'Time']
    win_ratio = sum(data_all.P1 > data_all.P2) / data_all.shape[0]
    hp_diff = sum(data_all.P1 - data_all.P2) / data_all.shape[0]
    
    data = []
    round_data = {}
    for i in range(data_all.shape[0]):
        # print('p1_HP:',data_all.P1[i])
        # print('p2_HP:',data_all.P2[i])
        # print('elapsed_frame:',data_all.Time[i])
        
        round_data['p1_HP'] = data_all.P1[i]
        round_data['p2_HP'] = data_all.P2[i]
        round_data['time'] = data_all.Time[i]/60
        data.append(round_data)
        round_data = {}
        
    remain_time = 0
    MyHp = 0
    OppHp = 0
    for item in data:
        MyHp += item['p1_HP']
        OppHp += item['p2_HP']
        if item['p1_HP'] >= item['p2_HP']:
            remain_time += 60-item['time']
            print(remain_time)
    
    Speed = remain_time/90/60
    RemainHP = MyHp/90/400
    Advantage = 0.5*((MyHp/90-OppHp/90)/400)
    Damage = 1 - OppHp/90/400

    return win_ratio, hp_diff, Speed, RemainHP, Advantage, Damage


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='D:/works/course/reinforcement learning/DareFightingICE-6.0.2/log/point', help='The directory containing result log')
    args = parser.parse_args()
    win_ratio, hp_diff, Speed, RemainHP, Advantage, Damage = calculate(args.path)
    print('The winning ratio is:', win_ratio)
    print('The average HP difference is:', hp_diff)
    print('Speed:',Speed)
    print('RemainHP:' + str(RemainHP))
    print('Advantage:' + str(Advantage))
    print('Damage:' + str(Damage))
