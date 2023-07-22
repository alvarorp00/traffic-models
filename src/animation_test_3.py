import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from celluloid import Camera

numpoints = 10
points = np.random.random((2, numpoints))
colors = cm.rainbow(np.linspace(0, 1, numpoints))
camera = Camera(plt.figure())
for _ in range(100):
    points += 0.1 * (np.random.random((2, numpoints)) - .5)
    plt.scatter(*points, c=colors, s=100)
    camera.snap()
anim = camera.animate(blit=True)
anim.save('scatter.mp4')
