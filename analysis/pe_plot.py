import json
import numpy as np
import matplotlib.pyplot as plt

def retrieve_bars_count(array):
    return np.histogram(array, bins=np.linspace(0, 1, 11))[0]

with open('sgd.json') as f:
    sgd_pe = json.load(f)

with open('adam.json') as f:
    adam_pe = json.load(f)

with open('rmsprop.json') as f:
    rmsprop_pe = json.load(f)

print('SGD pixel error mean: %.10f' % np.mean(sgd_pe))
print('Adam pixel error mean: %.10f' % np.mean(adam_pe))
print('RMSProp pixel error mean: %.10f' % np.mean(rmsprop_pe))

sgd_pe = retrieve_bars_count(np.array(sgd_pe))
adam_pe = retrieve_bars_count(np.array(adam_pe))
rmsprop_pe = retrieve_bars_count(np.array(rmsprop_pe))

plt.figure()

width=0.3

ind = np.arange(len(sgd_pe))
plt.bar(ind, sgd_pe, width, label='SGD')
plt.bar(ind + width, adam_pe, width, label='Adam')
plt.bar(ind + width * 2, rmsprop_pe, width, label='RMSProp')

plt.xlabel('Pixel Error')
plt.ylabel('Count')
plt.legend()

x_ticks = list(map(lambda x: str(x)[:3], list(np.linspace(0, 1, 11)[1:])))
plt.xticks(ind + width / 2, x_ticks)

plt.show()

