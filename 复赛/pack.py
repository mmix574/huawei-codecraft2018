import os
import sys
assert(sys.argv[1]=='-name')
filename = sys.argv[2]
os.system('tar -czvf {}.tar.gz src/ecs/*.py src/ecs/*.txt src/ecs/ecs.py src/ecs/linalg/*.py  src/ecs/learn/*.py src/ecs/ecs.py'.format(filename))
