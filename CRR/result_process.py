file_path = 'results/crr_pre_300wstep.txt'
with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
    res = list(f)

data = []
round_data = {}
for i in range(len(res)):
    if 'p1_HP' in res[i]:
        round_data['p1_HP'] = int(res[i].strip()[6:])
        round_data['p2_HP'] = int(res[i+1].strip()[6:])
        round_data['time'] = int(res[i+2].strip()[14:])/60
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

rounds = len(data)
Speed = remain_time/rounds/60
RemainHP = MyHp/rounds/400
Advantage = 0.5*((MyHp/rounds-OppHp/rounds)/400)
Damage = 1 - OppHp/rounds/400
win_ratio = sum([item['p1_HP'] > item['p2_HP'] for item in data])/rounds
hp_diff_avg  = sum([item['p1_HP'] - item['p2_HP'] for item in data])/rounds
print('Speed:' + str(Speed))
print('RemainHP:' + str(RemainHP))
print('Advantage:' + str(Advantage))
print('Damage:' + str(Damage))
print('win_ratio:' + str(win_ratio))
print('hp_diff_avg ' + str(hp_diff_avg ))


