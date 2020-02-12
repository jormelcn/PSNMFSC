
from PSNMFSC import unmix, sparseness
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

R = np.random.rand(50, 500)
p = 4

M, S, hist = unmix(R, p, Sp=0.9, alpha=1, gamma_M=0.1, iterations=tqdm(range(5000)))

plt.figure()
plt.plot(hist)
plt.savefig("history.png")
plt.close()

print(np.sum(S, axis=0))
print(sparseness(S))
print(S)

plt.figure()
for i in range(M.shape[1]):
    plt.plot(M[:,i])
plt.savefig("Endmembers.png")
plt.close()

print(np.mean((R - M@S)**2)**0.5)