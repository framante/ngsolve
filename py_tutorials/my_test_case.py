# solve the Poisson equation -Delta u = f
# with Dirichlet boundary condition u = 0

import ngsolve as ng
from netgen.geom2d import unit_square
import numpy as np

ng.ngsglobals.msg_level = 1

# generate a triangular mesh of mesh-size 0.2
mesh = ng.Mesh(unit_square.GenerateMesh(maxh=0.2))

# H1-conforming finite element space
fes = ng.H1(mesh, order=3, dirichlet=[1,2,3,4])

# define trial- and test-functions
u = fes.TrialFunction()
v = fes.TestFunction()

# the right hand side
f = ng.LinearForm(fes)
def g(x):
    return ng.exp(x**2 / 2)

#print("g(x) = ", g(np.linspace(0,10,100)))

f += 32 * g(ng.x) * v * ng.dx

# the bilinear-form 
a = ng.BilinearForm(fes, symmetric=True)
a += ng.grad(u)*ng.grad(v)*ng.dx
a += u*v*ng.dx

a.Assemble()
f.Assemble()

# the solution field 
gfu = ng.GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
print(type(gfu.vec), type(gfu.vec.data))
array = np.empty(0)
array = np.append(gfu.vec.data.FV().NumPy(), array)
print(type(array), array.shape)
array = np.append(gfu.vec.data.FV().NumPy(), array)
print(type(array), array.shape)

# plot the solution (netgen-gui only)
ng.Draw (gfu)
ng.Draw (-ng.grad(gfu), mesh, "Flux")

exact = 16 * ng.x * (1-ng.x) * ng.y * (1-ng.y)
print ("L2-error:", ng.sqrt (ng.Integrate ( (gfu-exact)*(gfu-exact), mesh)))
