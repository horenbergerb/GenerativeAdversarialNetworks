import matplotlib.pyplot as plt
import numpy as np

'''
demonstrating a support whose dimension is lower than the space in which it is embedded
'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

'''
# Cylinder
x=np.linspace(-1, 1, 100)
z=np.linspace(0, 1.0/(2.0*3.14159), 100)
Xc, Zc=np.meshgrid(x, z)
Yc = np.sqrt(1-Xc**2)

# Draw parameters
rstride = 20
cstride = 10
ax.plot_surface(Xc, Yc, Zc, alpha=0.3, rstride=1, cstride=1, cmap='coolwarm')
ax.plot_surface(Xc, -Yc, Zc, alpha=0.3, rstride=1, cstride=1, cmap='coolwarm')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
'''

theta = np.linspace(0, 2 * np.pi, 201)
x = np.cos(theta)
y = np.sin(theta)
z=((1/(2.0*3.14159)) + np.sin(0.5*theta))
ax.plot(x,y,z)


#plotting the support
#circle = plt.Circle((0,0), 1.0)
fig,ax = plt.subplots()
plt.plot(x,y)
plt.axis('scaled')

#plt.ylim((-2.0,2.0))
#plt.xlim((-2.0,2.0))
#ax.add_patch(circle)
plt.show()
