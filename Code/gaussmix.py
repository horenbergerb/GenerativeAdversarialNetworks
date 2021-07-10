import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.datasets import make_blobs

'''
We create a mixture of gaussians and generate samples
Then we get a gaussian kernel on the samples
Next we translate the gaussian kernel down and trim negative values
Finally we plot the gaussian kernel and plot its support

Reference:
https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67
'''

n_components = 3

X, truth = make_blobs(n_samples = 300, centers=n_components, cluster_std = [2,1.5,1], random_state=42)

# Extract x and y
x = X[:, 0]
y = X[:, 1]# Define the borders
deltaX = (max(x) - min(x))/10
deltaY = (max(y) - min(y))/10
xmin = min(x) - 2.2*deltaX
xmax = max(x) + 2.2*deltaX
ymin = min(y) - 2.2*deltaY
ymax = max(y) + 2.2*deltaY
print(xmin, xmax, ymin, ymax)# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]

#calculating kernel
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = st.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)


#translate and trim
f = f-0.001
f[f<0.0] = np.nan

#getting the support
vals = np.argwhere(~np.isnan(f))
print(vals)
sup_x = []
sup_y = []
for cur in range(len(vals)):
    sup_x.append(xx[vals[cur][0]][vals[cur][1]])
    sup_y.append(yy[vals[cur][0]][vals[cur][1]])

fig = plt.figure(figsize=(13, 7))
ax = plt.axes(projection='3d')
surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none', vmin=np.nanmin(f), vmax=np.nanmax(f))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
#ax.set_title('Surface plot of Gaussian 2D KDE')
fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
ax.view_init(60, 35)
plt.show()

#plt.figure(figsize=(13,7))
plt.ylim((-13, 20))
plt.xlim((-13,13))
plt.scatter(sup_x, sup_y,marker=",")
plt.show()
